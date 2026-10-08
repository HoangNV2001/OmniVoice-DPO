"""Per-candidate tag checks from stored features (CPU only).

The ASR transcript is aligned to the tag-free text. For each tag at word position P:

- the time region where its sound should be is between the end of word P-1 and the start of word P (±0.15 s);
- ASR words inserted at P are what the model said there.

Checks:

- ``err``: share of reference words the ASR did not find (insertions do not count; accent-free, so tone errors are
  ignored). A gross content filter, not a WER.
- NV: ``ok`` when the AudioSet classes of the tag peak above ``tau`` inside the region (``ast_pos``); ``ast_clip`` is the
  same over the whole clip. ``leak`` when the inserted words spell the tag name ("[laughs]" read as "láp").
- FIL: ``ok`` when an inserted word is one of the tag's ``heard_as`` spellings.
- EMO: only ``leak`` (whether the delivery matches is for listeners).

    python -m omnivoice_dpo.rewards.tagcheck --run run/ --feat run/feat --tags configs/tags_vi.yaml
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import os
import re
import unicodedata

import numpy as np

from omnivoice_dpo.tags import TagInventory

# What ASR writes for laughs, coughs, sighs...: inserted at an NV tag these mean the sound happened, not a leak.
ONO = {"ha", "haha", "hahaha", "he", "hehe", "hi", "hihi", "ho", "hoho", "khi", "khuc", "khich", "hu", "hum", "um",
       "hm", "oi", "o", "a", "e", "u", "uh", "ah", "oh", "ahem", "hem", "hat", "xi", "khu", "ac", "phu", "hay", "haiz",
       "hai", "tsk", "chac", "m", "ui", "ua", "om", "uhm", "hmm", "huh", "hic", "hix", "ay", "ai"}


def strip_diac(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower()).replace("đ", "d")
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def collapse(s: str) -> str:
    return re.sub(r"(.)\1+", r"\1", s)


def toks(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", strip_diac(s))


def asr_tokens(words) -> list[tuple[str, float, float]]:
    out = []
    for w, t0, t1 in words:
        for t in toks(w):
            out.append((t, t0 if t0 is not None else 0.0, t1 if t1 is not None else (t0 or 0.0)))
    return out


def ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def ast_score(win, classes, want, reg) -> float:
    """Max probability of the ``want`` classes over 1 s windows touching ``reg``."""
    if win is None or len(win) == 0:
        return 0.0
    idx = [1 + classes.index(c) for c in want if c in classes]
    if not idx:
        return 0.0
    c = win[:, 0]
    m = (c + 0.5 >= reg[0]) & (c - 0.5 <= reg[1])
    if not m.any():
        m = np.ones(len(c), bool)
    return float(win[m][:, idx].max())


def check_candidate(inv: TagInventory, text: str, asr: dict | None, win, classes: list[str], dur: float,
                    tau: float = 0.1, pad: float = 0.15) -> dict:
    spans = inv.parse(text)
    ref = toks(inv.strip(text))
    hyp = asr_tokens(asr["words"]) if asr else []
    hs = [h[0] for h in hyp]
    r2h, ins_at = {}, collections.defaultdict(list)
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ref, hs, autojunk=False).get_opcodes():
        if op == "equal":
            r2h.update({i1 + d: j1 + d for d in range(i2 - i1)})
        elif op == "insert":
            ins_at[i1] += hs[j1:j2]
        elif op == "replace" and (j2 - j1) > (i2 - i1):
            ins_at[i1] += hs[j1:j1 + (j2 - j1) - (i2 - i1)]
    err = 1 - len(r2h) / max(len(ref), 1)

    def t_start(i):
        for k in range(i, len(ref)):
            if k in r2h:
                return hyp[r2h[k]][1]
        return None

    def t_end(i):
        for k in range(i, -1, -1):
            if k in r2h:
                return hyp[r2h[k]][2]
        return None

    tags = []
    for sp in spans:
        P = len(toks(inv.strip(text[:sp.start])))
        ins = [collapse(t) for t in ins_at.get(P, [])]
        t_prev = t_end(P - 1) if P > 0 else None
        t_next = t_start(P)
        reg = ((t_prev - pad) if t_prev is not None else 0.0, (t_next + pad) if t_next is not None else dur)
        name = collapse(re.sub(r"[^a-z]", "", strip_diac(sp.tag.name)))
        joined = "".join(ins)
        leak = any(len(t) >= 3 and ratio(t, name) >= 0.7 for t in ins) or (len(joined) >= 4 and ratio(joined, name) >= 0.7)
        rec = {"tag": sp.tag.token, "cat": sp.tag.category, "pos": P, "region": [round(reg[0], 2), round(reg[1], 2)],
               "ins": ins}
        if sp.tag.category == "NV":
            rec["ast_pos"] = ast_score(win, classes, sp.tag.audioset, reg)
            rec["ast_clip"] = ast_score(win, classes, sp.tag.audioset, (0.0, dur))
            rec["ok"] = rec["ast_pos"] >= tau
            rec["leak"] = leak
            rec["chen"] = any(t not in ONO for t in ins)
        elif sp.tag.category == "FIL":
            rec["ok"] = any(t in sp.tag.heard_as for t in ins)
            rec["leak"] = False
        else:
            rec["ok"] = None
            rec["leak"] = leak
        tags.append(rec)
    return {"err": round(err, 4), "leak": any(t["leak"] for t in tags), "tags": tags,
            "asr": asr["text"] if asr else ""}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="directory with meta.jsonl and wavs")
    ap.add_argument("--feat", required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--out", help="default: <feat>/checks.jsonl")
    a = ap.parse_args()
    import soundfile as sf

    inv = TagInventory.from_yaml(a.tags)
    meta = [json.loads(l) for l in open(os.path.join(a.run, "meta.jsonl"), encoding="utf-8") if l.strip()]
    asr = {r["id"]: r for r in map(json.loads, open(os.path.join(a.feat, "asr.jsonl"), encoding="utf-8"))}
    ast = np.load(os.path.join(a.feat, "ast.npz"))
    classes = list(ast["classes"])
    with open(a.out or os.path.join(a.feat, "checks.jsonl"), "w", encoding="utf-8") as f:
        for m in meta:
            cid = f"{m['id']}.{m['k']}"
            dur = sf.info(os.path.join(a.run, m["wav"])).duration
            win = ast[cid] if cid in ast.files else None
            c = check_candidate(inv, m["text"], asr.get(cid), win, classes, dur, tau=a.tau)
            f.write(json.dumps({"id": m["id"], "k": m["k"], "dur": round(dur, 2), **c}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
