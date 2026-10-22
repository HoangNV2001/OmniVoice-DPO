"""Generation with per-syllable prosody embeddings, on the upstream sampler.

The target text's syllables are located in the conditional input exactly as at training time (re-tokenize the wrapped
text with offsets, check the ids), their embeddings are scattered there, and the model adds them at every sampling step
(``OmniVoiceProsody.infer_add``). The unconditional half of the CFG batch carries no text and gets nothing. One item per
call.

Embeddings for a syllable come from codes (``ProsodyEncoder.embed_codes``), e.g. codes encoded from a recording of the
same text (:func:`codes_from_store`); a syllable without a code (``keep`` 0) gets nothing, as when it was dropped in
training.
"""

from __future__ import annotations

import numpy as np
import torch

from omnivoice.models.omnivoice import _combine_text
from omnivoice_dpo.prosody.text import syllables, token_to_syllable

TEXT_START = "<|text_start|>"


def target_positions(model, task, denoise: bool = True, i: int = 0):
    """For each position of item ``i``'s conditional input, the index of the target-text syllable it belongs to (-1
    elsewhere, including the reference text), plus the number of target syllables. Uses the task exactly as the sampler
    will (same text, reference, length)."""
    text, ref_text = task.texts[i], task.ref_texts[i]
    inp = model._prepare_inference_inputs(text, task.target_lens[i], ref_text, task.ref_audio_tokens[i], task.langs[i],
                                          task.instructs[i], denoise)
    ids = inp["input_ids"][0, 0]
    tok = model.text_tokenizer
    ts, te = tok.convert_tokens_to_ids(TEXT_START), tok.convert_tokens_to_ids("<|text_end|>")
    a = int((ids == ts).nonzero()[0])
    b = int((ids == te).nonzero()[0]) + 1
    full = _combine_text(ref_text=ref_text, text=text)
    enc = tok(f"{TEXT_START}{full}<|text_end|>", return_offsets_mapping=True)
    if enc["input_ids"] != ids[a:b].tolist():
        raise ValueError("re-tokenized text does not match the sampler's tokens (tags in text?)")
    spans = syllables(full)
    n_target = len(syllables(text))
    first = len(spans) - n_target
    t2s = token_to_syllable(enc["offset_mapping"], spans, len(TEXT_START))
    pos = torch.full((ids.numel(),), -1, dtype=torch.long)
    pos[a:b] = torch.tensor([k - first if k >= first else -1 for k in t2s])
    return pos, n_target


@torch.inference_mode()
def generate(model, text: str, vcp, num_target: int, cfg, syl_emb: torch.Tensor | None = None, seed: int = 0,
             lang: str = "vi") -> torch.Tensor:
    """Audio tokens ``[C, num_target]``. ``syl_emb`` ``[n_target_syllables, H]`` or None (plain model)."""
    torch.manual_seed(seed)
    task = model._preprocess_all(text=[text], language=[lang], voice_clone_prompt=[vcp], duration=None,
                                 preprocess_prompt=cfg.preprocess_prompt)
    task.target_lens = [num_target]
    model.infer_add = None
    if syl_emb is not None:
        pos, n = target_positions(model, task, cfg.denoise)
        if n != syl_emb.shape[0]:
            raise ValueError(f"{n} syllables in text, {syl_emb.shape[0]} embeddings")
        add = torch.zeros(2, pos.numel(), syl_emb.shape[-1], device=model.device, dtype=syl_emb.dtype)
        m = pos >= 0
        add[0, m.to(model.device)] = syl_emb[pos[m].to(syl_emb.device)].to(add.device)
        model.infer_add = add
    try:
        return model._generate_iterative(task, cfg)[0]
    finally:
        model.infer_add = None


@torch.inference_mode()
def codes_from_store(model, feats: np.ndarray, syl: np.ndarray) -> torch.Tensor:
    """Codes ``[n, dims]`` of one utterance from its stored frame features and syllables (prep output)."""
    enc = model.prosody_encoder
    dev = next(enc.parameters()).device
    f = torch.from_numpy(np.asarray(feats, dtype=np.float32))[None].to(dev)
    s = torch.from_numpy(syl[:, 2:4].astype(np.int64)).to(dev)
    span = torch.cat([torch.zeros(len(s), 1, dtype=torch.long, device=dev), s], 1)
    extra = torch.log1p(torch.from_numpy(np.stack([syl[:, 3] - syl[:, 2], syl[:, 4]], 1).astype(np.float32))).to(dev)
    return enc.codes(f, torch.tensor([f.shape[1]], device=dev), span, extra)
