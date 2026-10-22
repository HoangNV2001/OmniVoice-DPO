"""Training data with prosody: a store of prep outputs, and wrappers around the upstream sample processor and collators.

The upstream processor builds ``[style][<|text_start|> text <|text_end|>][audio]``. The wrapper finds the text span in
``input_ids``, re-tokenizes the same string with character offsets (and checks the ids match), maps each text token to
its syllable, and attaches the utterance's frame features and syllable spans. Samples whose text was dropped (unconditional
training), that have no aligned entry in the store, or whose stored text / length differ, carry no prosody; so do a share
``p_utt`` of the rest, and single syllables with probability ``p_syl`` or with alignment confidence below ``min_conf``.
"""

from __future__ import annotations

import hashlib
import json
import os
import random

import numpy as np
import torch

from omnivoice_dpo.prosody.prep import FEATS
from omnivoice_dpo.prosody.text import token_to_syllable

TEXT_START = "<|text_start|>"


class ProsodyStore:
    """``(id, md5 of text)`` -> frame features ``[T, F]`` and syllables ``[n, 5]`` (char start, char end, first frame,
    end frame, pause frames) with confidences, over every ``<root>/<dataset>/index.jsonl``."""

    def __init__(self, root: str):
        self.root, self.names, self.keys = root, [], {}
        rows, confs, entries, n_rows = [], [], [], 0
        for name in sorted(os.listdir(root)):
            idx = os.path.join(root, name, "index.jsonl")
            if not os.path.isfile(idx):
                continue
            di = len(self.names)
            self.names.append(name)
            for line in open(idx, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:  # last line of a store still being written
                    continue
                if not r["syl"] or r["off"] < 0:
                    continue
                s = np.asarray(r["syl"], dtype=np.float32)
                self.keys[(r["id"], r["md5"])] = len(entries)
                entries.append((di, r["off"], r["T"], n_rows, len(s)))
                n_rows += len(s)
                rows.append(s[:, :5].astype(np.int32))
                confs.append(s[:, 5])
        self.entries = np.asarray(entries, dtype=np.int64).reshape(-1, 5)
        self.syl = np.concatenate(rows) if rows else np.zeros((0, 5), np.int32)
        self.conf = np.concatenate(confs) if confs else np.zeros(0, np.float32)
        self._mm = {}

    def __len__(self):
        return len(self.keys)

    def _feats(self, di: int) -> np.ndarray:
        if di not in self._mm:  # opened lazily, once per data-loader worker
            path = os.path.join(self.root, self.names[di], "feats.f16")
            self._mm[di] = np.memmap(path, dtype=np.float16, mode="r").reshape(-1, len(FEATS))
        return self._mm[di]

    def lookup(self, sample_id: str, text: str, T: int):
        k = self.keys.get((sample_id, hashlib.md5(text.encode()).hexdigest()[:12]))
        if k is None:
            return None
        di, off, t, so, n = self.entries[k]
        if t != T:
            return None
        return np.asarray(self._feats(di)[off:off + t], dtype=np.float32), self.syl[so:so + n], self.conf[so:so + n]


class ProsodySampleProcessor:
    def __init__(self, base, store: ProsodyStore, p_utt: float = 0.3, p_syl: float = 0.1, min_conf: float = 0.3):
        self.base, self.store, self.p_utt, self.p_syl, self.min_conf = base, store, p_utt, p_syl, min_conf
        self.text_tokenizer = base.text_tokenizer
        self.ts_id = self.text_tokenizer.convert_tokens_to_ids(TEXT_START)
        self.te_id = self.text_tokenizer.convert_tokens_to_ids("<|text_end|>")

    def __call__(self, sample):
        out = self.base(sample)
        try:
            p = self.prosody(sample, out)
        except Exception:  # never lose a training sample over its prosody
            p = None
        if p is not None:
            out["pros"] = p
        return out

    def prosody(self, sample, out):
        if out["audio_mask"].all() or random.random() < self.p_utt:
            return None
        text = sample["label"].get("text", "")
        got = self.store.lookup(sample["label"]["id"], text, sample["audio_tokens"].shape[1])
        if got is None:
            return None
        feats, syl, conf = got
        ids = out["input_ids"][0]
        a = (ids == self.ts_id).nonzero().flatten()
        b = (ids == self.te_id).nonzero().flatten()
        if len(a) != 1 or len(b) != 1:
            return None
        a, b = int(a[0]), int(b[0]) + 1
        enc = self.text_tokenizer(f"{TEXT_START}{text}<|text_end|>", return_offsets_mapping=True)
        if enc["input_ids"] != ids[a:b].tolist():
            return None
        t2s = token_to_syllable(enc["offset_mapping"], [(int(s[0]), int(s[1])) for s in syl], len(TEXT_START))
        tok = torch.full((out["length"],), -1, dtype=torch.long)
        tok[a:b] = torch.tensor(t2s, dtype=torch.long)
        keep = (torch.rand(len(syl)) >= self.p_syl) & torch.from_numpy(conf >= self.min_conf)
        span = torch.from_numpy(syl[:, 2:4].astype(np.int64))
        extra = torch.log1p(torch.from_numpy(np.stack([syl[:, 3] - syl[:, 2], syl[:, 4]], 1).astype(np.float32)))
        return {"feats": torch.from_numpy(feats), "span": span, "extra": extra, "keep": keep.float(), "tok": tok}


class ProsodyCollator:
    """Runs the upstream collator, then adds the ``pros_*`` fields (see :mod:`omnivoice_dpo.prosody.model`)."""

    def __init__(self, base):
        self.base = base

    def __call__(self, samples):
        out = self.base(samples)
        out.update(collate_prosody(samples, out["input_ids"].shape[0], out["input_ids"].shape[-1],
                                   packed="document_ids" in out))
        return out


def collate_prosody(samples, B: int, L: int, packed: bool) -> dict:
    tok = torch.full((B, L), -1, dtype=torch.long)
    feats, flen, syl, extra, keep = [], [], [], [], []
    off = n_syl = 0
    for i, s in enumerate(samples):
        row, start = (0, off) if packed else (i, 0)
        p = s.get("pros")
        if p is not None:
            u = len(feats)
            feats.append(p["feats"])
            flen.append(p["feats"].shape[0])
            syl.append(torch.cat([torch.full((len(p["span"]), 1), u, dtype=torch.long), p["span"]], 1))
            extra.append(p["extra"])
            keep.append(p["keep"])
            t = p["tok"]
            view = tok[row, start:start + len(t)]
            m = t >= 0
            view[m] = t[m] + n_syl
            n_syl += len(p["span"])
        off += s["length"]
    if feats:
        T = max(flen)
        pf = torch.zeros(len(feats), T, len(FEATS))
        for u, f in enumerate(feats):
            pf[u, :len(f)] = f
        return {"pros_feats": pf, "pros_flen": torch.tensor(flen), "pros_syl": torch.cat(syl), "pros_extra": torch.cat(extra),
                "pros_keep": torch.cat(keep), "pros_tok": tok}
    return {"pros_feats": torch.zeros(0, 1, len(FEATS)), "pros_flen": torch.zeros(0, dtype=torch.long),
            "pros_syl": torch.zeros(0, 3, dtype=torch.long), "pros_extra": torch.zeros(0, 2), "pros_keep": torch.zeros(0),
            "pros_tok": tok}
