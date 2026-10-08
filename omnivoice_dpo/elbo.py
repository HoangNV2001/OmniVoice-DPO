"""Per-sample masked-diffusion ELBO scores with variance reduction (as in VRPO / LLaDA 1.5).

A candidate is the exact token grid ``X`` (``[C, T]``) generated after a fixed conditioning prefix (style, text and
reference-voice tokens, built exactly as at generation time). Its score is a Monte-Carlo ELBO estimate:

    s(X) = -(1 / n_t) * sum_k (1 / t_k) * sum_c w_c * sum_{(c, j) in M_k} CE(x_cj | X with M_k masked, prefix)

Variance reduction:
- ``n_t`` stratified timesteps ``t_k = t_min + (1 - t_min) * (k + u) / n_t`` with one shared ``u``;
- one mask ``M_k ~ Bernoulli(t_k)`` per timestep;
- the same ``(t_k, M_k)`` for chosen and rejected (same length by construction) and for policy and reference
  (antithetic sampling), drawn from a per-pair generator so both models see identical corruptions.

``weighting="mean"`` gives the draft's MRPO score instead: masked-token mean CE with OmniVoice codebook weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Corruption:
    t: torch.Tensor  # [n_t]
    masks: torch.Tensor  # [n_t, C, T] bool


def sample_corruption(C: int, T: int, n_t: int, t_min: float, generator: torch.Generator) -> Corruption:
    u = torch.rand((), generator=generator)
    t = t_min + (1 - t_min) * (torch.arange(n_t, dtype=torch.float32) + u) / n_t
    masks = torch.rand((n_t, C, T), generator=generator) < t[:, None, None]
    empty = ~masks.flatten(1).any(1)
    if empty.any():  # very short targets at small t: force one masked token so every term is defined
        masks[empty, 0, 0] = True
    return Corruption(t=t, masks=masks)


def build_batch(prefixes, audio_starts, targets, corruptions, mask_id: int, pad_id: int):
    """Rows = every (target, timestep). ``prefixes[i]``: ``[C, Lp]`` long; ``audio_starts[i]``: index in the prefix where
    audio tokens begin (reference audio, or the target when there is none); ``targets[i]``: ``[C, T]``;
    ``corruptions[i]``: its :class:`Corruption`. Returns tensors for :func:`target_logprobs` plus row bookkeeping."""
    rows = []
    for i, (p, a0, x, cor) in enumerate(zip(prefixes, audio_starts, targets, corruptions)):
        for k in range(len(cor.t)):
            m = cor.masks[k]
            xin = x.masked_fill(m, mask_id)
            lab = x.masked_fill(~m, -100)
            rows.append((i, k, torch.cat([p, xin], 1), lab, a0, p.shape[1]))
    C = rows[0][2].shape[0]
    L = max(r[2].shape[1] for r in rows)
    B = len(rows)
    ids = torch.full((B, C, L), pad_id, dtype=torch.long)
    audio_mask = torch.zeros((B, L), dtype=torch.bool)
    valid = torch.zeros((B, L), dtype=torch.bool)
    pos = torch.zeros((B, L), dtype=torch.long)
    for b, (_, _, seq, _, a0, _) in enumerate(rows):
        n = seq.shape[1]
        ids[b, :, :n] = seq
        audio_mask[b, a0:n] = True
        valid[b, :n] = True
        pos[b, :n] = torch.arange(n)
    attn = valid[:, None, None, :].expand(B, 1, L, L).contiguous()
    meta = [(i, k, lp, lab) for (i, k, _, lab, _, lp) in rows]
    return {"input_ids": ids, "audio_mask": audio_mask, "attention_mask": attn, "position_ids": pos}, meta


def _core(model):
    """The OmniVoice module under an optional PEFT wrapper (LoRA layers stay active: PEFT patches modules in place)."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def target_hidden_logits(model, batch, meta):
    """Run the backbone once and apply the audio heads only on each row's target span. Returns a list of
    ``[C, T, V]`` logits, one per row."""
    m = _core(model)
    emb = m._prepare_embed_inputs(batch["input_ids"], batch["audio_mask"])
    h = m.llm(inputs_embeds=emb, attention_mask=batch["attention_mask"], position_ids=batch["position_ids"],
              return_dict=True)[0]
    C, V = m.config.num_audio_codebook, m.config.audio_vocab_size
    out = []
    for b, (_, _, lp, lab) in enumerate(meta):
        T = lab.shape[1]
        logits = m.audio_heads(h[b, lp:lp + T])  # [T, C*V]
        out.append(logits.view(T, C, V).permute(1, 0, 2))
    return out


def score(model, batch, meta, corruptions, n_targets: int, weighting: str = "elbo", norm: str = "tokens",
          codebook_weights=None):
    """Scores ``[n_targets]`` (higher = model finds the candidate more likely) and per-codebook masked CE ``[n_targets, C]``."""
    logits = target_hidden_logits(model, batch, meta)
    device = logits[0].device
    C = logits[0].shape[0]
    w = torch.ones(C, device=device) if codebook_weights is None else torch.as_tensor(codebook_weights, device=device,
                                                                                         dtype=torch.float32)
    terms = [[] for _ in range(n_targets)]
    per_cb = torch.zeros(n_targets, C, device=device)
    cnt = torch.zeros(n_targets, C, device=device)
    for lg, (i, k, _, lab) in zip(logits, meta):
        lab = lab.to(device)
        ce = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), lab.reshape(-1), reduction="none",
                             ignore_index=-100).view_as(lab)  # [C, T], 0 where not masked
        n_c = (lab != -100).sum(1).float()
        sum_c = ce.sum(1)
        per_cb[i] += sum_c.detach()
        cnt[i] += n_c
        if weighting == "elbo":
            term = (w * sum_c).sum() / corruptions[i].t[k].to(device)
            if norm == "tokens":
                term = term / lab.numel()
        elif weighting == "mean":
            term = (w / w.sum() * sum_c / n_c.clamp_min(1)).sum()
        else:
            raise ValueError(weighting)
        terms[i].append(term)
    s = torch.stack([torch.stack(t).mean() for t in terms])
    return -s, per_cb / cnt.clamp_min(1)
