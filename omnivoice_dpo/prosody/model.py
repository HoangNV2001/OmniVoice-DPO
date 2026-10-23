"""Prosody encoder (frame features -> one quantized code per syllable) and an OmniVoice that adds it to text embeddings.

Encoder: per-frame features (:data:`omnivoice_dpo.prosody.prep.FEATS`, 25 Hz) -> linear -> residual 1-D conv blocks over
the utterance -> mean over each syllable's frames, plus log duration and log pause -> MLP -> FSQ (finite scalar
quantization, a few dims with a few levels each: the information bottleneck) -> linear to the LLM width. The last linear is
zero-initialized, and a dropped syllable contributes exactly zero, so with no codes the model input is unchanged.

A small decoder predicts simple prosodic statistics of each syllable back from its code (:data:`N_AUX`); its loss
(``aux_weight``) keeps the codes informative from the start, before the backbone has learned to read them.

Batch fields (built by :mod:`omnivoice_dpo.prosody.data`):

- ``pros_feats`` ``[U, Tmax, F]`` frame features of the U utterances that carry prosody, zero-padded; ``pros_flen`` ``[U]``
  their frame counts;
- ``pros_syl`` ``[S, 3]`` long: utterance index, first frame, end frame (exclusive) of each syllable;
- ``pros_extra`` ``[S, 2]``: log(1 + duration frames), log(1 + pause frames);
- ``pros_keep`` ``[S]``: 1 to use the syllable's code, 0 to drop it;
- ``pros_tok`` ``[B, L]`` long: for each input position, the syllable index it belongs to, else -1.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn

from omnivoice.models.omnivoice import OmniVoice

logger = logging.getLogger(__name__)

DEFAULT = {"n_feats": 11, "dim": 256, "n_blocks": 4, "kernel": 5, "levels": [8, 5, 5, 5],
           "feat_scale": [1 / 6, 1.0, 1 / 20] + [1 / 4] * 8, "aux_weight": 1.0}
# Auxiliary targets per syllable, predicted back from its code so the code has to carry them: mean F0 (semitones, voiced
# frames) / 6, voiced fraction, mean energy (dB) / 20, log(1 + duration frames), log(1 + pause frames).
N_AUX = 5


class FSQ(nn.Module):
    """Finite scalar quantization (Mentzer et al., 2023): bound each dim to ``levels[i]`` values, round with a
    straight-through gradient. Output in [-1, 1] per dim."""

    # Hyper-parameters are kept as plain lists, not buffers: checkpoints without this module load cleanly.
    def __init__(self, levels: list[int]):
        super().__init__()
        self.level_list = list(levels)

    def _levels(self, ref: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.level_list, dtype=torch.float32, device=ref.device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        lv = self._levels(z)
        half_l = (lv - 1) * (1 + 1e-3) / 2
        offset = torch.where(lv % 2 == 0, 0.5, 0.0)
        shift = torch.atanh(offset / half_l)
        bounded = torch.tanh(z.float() + shift) * half_l - offset
        q = bounded + (bounded.round() - bounded).detach()
        return q / (lv // 2)

    def indices(self, codes: torch.Tensor) -> torch.Tensor:
        """Integer level per dim, 0..levels-1."""
        half = self._levels(codes) // 2
        return (codes * half + half).round().long()

    def from_indices(self, idx: torch.Tensor) -> torch.Tensor:
        half = self._levels(idx) // 2
        return (idx.float() - half) / half


class _Block(nn.Module):
    def __init__(self, dim: int, kernel: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel, padding=kernel // 2)
        self.ff = nn.Sequential(nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x, mask):  # x [U, T, D], mask [U, T, 1]
        h = self.conv((self.norm(x) * mask).transpose(1, 2)).transpose(1, 2)
        return (x + self.ff(h)) * mask


class ProsodyEncoder(nn.Module):
    def __init__(self, hidden: int, n_feats: int, dim: int, n_blocks: int, kernel: int, levels: list[int],
                 feat_scale: list[float], aux_weight: float = 0.0):
        super().__init__()
        self.aux_weight = aux_weight
        self.feat_scale = list(feat_scale)
        self.inp = nn.Linear(n_feats, dim)
        self.blocks = nn.ModuleList(_Block(dim, kernel) for _ in range(n_blocks))
        self.head = nn.Sequential(nn.LayerNorm(dim + 2), nn.Linear(dim + 2, dim), nn.GELU(), nn.Linear(dim, len(levels)))
        self.fsq = FSQ(levels)
        self.out = nn.Linear(len(levels), hidden)
        self.aux = nn.Sequential(nn.Linear(len(levels), dim), nn.GELU(), nn.Linear(dim, N_AUX))
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def codes(self, feats, flen, syl, extra):
        """Quantized codes ``[S, n_dims]`` in [-1, 1]."""
        return self.encode(feats, flen, syl, extra)[0]

    @staticmethod
    def aux_targets(feats, syl, extra):
        """``[S, N_AUX]`` from the raw frame features of each syllable."""
        f = feats.float()
        cols = torch.stack([f[..., 0], f[..., 1], f[..., 2], (f[..., 1] > 0).float()], -1)
        cs = torch.nn.functional.pad(cols.cumsum(1), (0, 0, 1, 0))
        u, f0, f1 = syl[:, 0], syl[:, 1], syl[:, 2]
        tot = cs[u, f1] - cs[u, f0]
        n = (f1 - f0).clamp(min=1).float()
        st = tot[:, 0] / tot[:, 3].clamp(min=1)
        return torch.stack([st / 6, tot[:, 1] / n, tot[:, 2] / n / 20, extra[:, 0].float(), extra[:, 1].float()], -1)

    def encode(self, feats, flen, syl, extra):
        """``(codes [S, n_dims], aux loss)``: the codes, and how badly they predict :func:`aux_targets`."""
        U, T, _ = feats.shape
        target = self.aux_targets(feats, syl, extra)
        mask = (torch.arange(T, device=feats.device)[None] < flen[:, None]).unsqueeze(-1).to(feats.dtype)
        dt = self.inp.weight.dtype
        feats, extra, mask = feats.to(dt), extra.to(dt), mask.to(dt)
        scale = torch.tensor(self.feat_scale, dtype=dt, device=feats.device)
        x = self.inp(feats * scale) * mask
        for b in self.blocks:
            x = b(x, mask)
        cs = torch.nn.functional.pad(x.cumsum(1), (0, 0, 1, 0))  # [U, T + 1, D]
        u, f0, f1 = syl[:, 0], syl[:, 1], syl[:, 2]
        pooled = (cs[u, f1] - cs[u, f0]) / (f1 - f0).clamp(min=1).unsqueeze(-1).to(x.dtype)
        codes = self.fsq(self.head(torch.cat([pooled, extra.to(x.dtype)], -1)))
        aux = torch.nn.functional.mse_loss(self.aux(codes.to(dt)).float(), target)
        return codes, aux

    def forward(self, feats, flen, syl, extra, keep):
        """``(embeddings [S, hidden], aux loss)``; dropped syllables embed to exactly zero."""
        codes, aux = self.encode(feats, flen, syl, extra)
        return self.out(codes.to(self.out.weight.dtype)) * keep.unsqueeze(-1).to(self.out.weight.dtype), aux

    def embed_codes(self, codes, keep):
        dt = self.out.weight.dtype
        return self.out(codes.to(dt)) * keep.unsqueeze(-1).to(dt)


def scatter_to_positions(emb: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
    """``[B, L, H]``: row ``emb[tok[b, l]]`` where ``tok >= 0``, zero elsewhere."""
    out = emb.new_zeros(*tok.shape, emb.shape[-1])
    m = tok >= 0
    out[m] = emb[tok[m]]
    return out


class OmniVoiceProsody(OmniVoice):
    """OmniVoice whose text-token embeddings get a per-syllable prosody embedding added (``config.prosody`` holds the
    encoder hyper-parameters). Without ``pros_*`` inputs it is the plain model."""

    def __init__(self, config, llm=None):
        super().__init__(config, llm)
        pc = {**DEFAULT, **(getattr(config, "prosody", None) or {})}
        config.prosody = pc
        self.prosody_encoder = ProsodyEncoder(hidden=config.llm_config.hidden_size, **pc)
        self._pros_add = None
        self.infer_add = None  # [2B, L, H] set by omnivoice_dpo.prosody.infer around the upstream sampler
        self.pros_guidance = 0.0  # > 0: at inference, amplify the codes' effect (see forward)
        self._aux_log = []

    def _prepare_embed_inputs(self, input_ids, audio_mask):
        e = super()._prepare_embed_inputs(input_ids, audio_mask)
        if self._pros_add is not None:
            e = e + self._pros_add.to(e.dtype)
        return e

    def prosody_add(self, pros_feats, pros_flen, pros_syl, pros_extra, pros_keep, pros_tok):
        """``(additions [B, L, H], aux loss)``."""
        emb, aux = self.prosody_encoder(pros_feats.float(), pros_flen, pros_syl, pros_extra.float(), pros_keep.float())
        return scatter_to_positions(emb, pros_tok), aux

    def forward(self, input_ids, audio_mask, labels=None, attention_mask=None, document_ids=None, position_ids=None,
                pros_feats=None, pros_flen=None, pros_syl=None, pros_extra=None, pros_keep=None, pros_tok=None,
                pros_add=None):
        aux = None
        if pros_add is None and pros_feats is not None and pros_syl.shape[0] > 0:
            pros_add, aux = self.prosody_add(pros_feats, pros_flen, pros_syl, pros_extra, pros_keep, pros_tok)
        if pros_add is None and self.infer_add is not None and self.infer_add.shape[:2] == (input_ids.shape[0],
                                                                                            input_ids.shape[-1]):
            pros_add = self.infer_add
        if pros_add is None and self.training:
            # a batch with no prosody still touches every encoder weight (zero gradient), as DDP without
            # find_unused_parameters requires
            pros_add = sum(p.reshape(-1)[0] for p in self.prosody_encoder.parameters()) * 0.0
        self._pros_add = pros_add
        try:
            out = super().forward(input_ids=input_ids, audio_mask=audio_mask, labels=labels,
                                  attention_mask=attention_mask, document_ids=document_ids, position_ids=position_ids)
        finally:
            self._pros_add = None
        if self.pros_guidance > 0 and not self.training and pros_add is not None and labels is None:
            # prosody guidance, like CFG on text: lp + w * (lp - lp without codes), on log-probs; rows without codes
            # (the unconditional half) are unchanged
            plain = super().forward(input_ids=input_ids, audio_mask=audio_mask, attention_mask=attention_mask,
                                    document_ids=document_ids, position_ids=position_ids)
            lp, l0 = out.logits.float().log_softmax(-1), plain.logits.float().log_softmax(-1)
            out.logits = (lp + self.pros_guidance * (lp - l0)).to(out.logits.dtype)
        w = self.prosody_encoder.aux_weight
        if out.loss is not None and aux is not None and w > 0:
            out.loss = out.loss + w * aux
            self._aux_log.append(float(aux.detach()))
            if len(self._aux_log) >= 400:
                logger.info(f"prosody aux loss (mean of {len(self._aux_log)} batches): {np.mean(self._aux_log):.4f}")
                self._aux_log = []
        return out
