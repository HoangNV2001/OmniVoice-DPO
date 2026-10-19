"""Prosody features per candidate and prosodic distance between the candidates of a prompt (CPU only).

Prosody has no automatic judge, so these features never decide which candidate is better. They are used to:

1. drop broken candidates before listening (content error from ASR);
2. pick, among the valid candidates of a prompt, the pair that differs most in prosody, so a listening turn is not spent
   on two near-identical clips;
3. track per round whether intonation flattens (F0 spread), a known failure of preference training on prosody.

Per clip, on a 10 ms grid: F0 (pYIN) in semitones relative to the clip's median voiced F0, and frame energy in dB.

- ``err``: share of reference words the ASR did not find (accent-free, insertions ignored), as in ``tagcheck``.
- ``f0_std``, ``f0_p10``, ``f0_p90`` (semitones) and ``voiced`` (fraction of voiced frames in the speech span).
- ``pauses``: inner silences of at least 150 ms (energy 35 dB below the clip peak), as ``[start_s, dur_s]``.
- ``rate``: reference words per second of speech (pauses excluded).

Per prompt, ``dist[a][b]``: DTW distance between the [F0, energy] contours of candidates a and b (leading and trailing
silence trimmed, 20 ms frames), averaged along the warping path. Contours are kept in ``contours.npz`` so distances can
be recomputed without running pYIN again.

    python -m omnivoice_dpo.rewards.prosody --run run/cands --feat run/feat [--workers 8]
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import multiprocessing as mp
import os

import numpy as np

from omnivoice_dpo.rewards.tagcheck import asr_tokens, toks

SR, HOP = 16000, 160  # 10 ms frames
SIL_DB, MIN_PAUSE = 35.0, 15  # silence threshold below peak; shortest inner pause in frames
E_SCALE = 4.0  # dB per unit, so energy and F0 (semitones) contribute on a similar scale to the DTW cost


def content_error(text: str, asr: dict | None) -> float:
    ref = toks(text)
    hyp = [h[0] for h in asr_tokens(asr["words"])] if asr else []
    hit = sum(i2 - i1 for op, i1, i2, _, _ in difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes()
              if op == "equal")
    return 1 - hit / max(len(ref), 1)


def contours(path: str) -> tuple[np.ndarray, np.ndarray]:
    """``(st, db)`` per 10 ms frame: F0 in semitones re the clip median (NaN when unvoiced) and energy in dB."""
    import librosa

    y, _ = librosa.load(path, sr=SR)
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=500, sr=SR, frame_length=1024, hop_length=HOP)
    db = 20 * np.log10(librosa.feature.rms(y=y, frame_length=400, hop_length=HOP)[0] + 1e-5)
    n = min(len(f0), len(db))
    f0, voiced, db = f0[:n], voiced[:n], db[:n]
    st = np.full(n, np.nan, np.float32)
    if voiced.any():
        st[voiced] = 12 * np.log2(f0[voiced] / np.median(f0[voiced]))
    return st, db.astype(np.float32)


def speech_span(db: np.ndarray) -> tuple[np.ndarray, int, int]:
    loud = db >= db.max() - SIL_DB
    idx = np.flatnonzero(loud)
    return loud, (int(idx[0]) if len(idx) else 0), (int(idx[-1]) + 1 if len(idx) else len(db))


def clip_stats(st: np.ndarray, db: np.ndarray, n_words: int) -> dict:
    loud, a, b = speech_span(db)
    pauses, run = [], 0
    for i in range(a, b + 1):
        if i < b and not loud[i]:
            run += 1
            continue
        if run >= MIN_PAUSE:
            pauses.append([round((i - run) * HOP / SR, 2), round(run * HOP / SR, 2)])
        run = 0
    v = st[a:b][~np.isnan(st[a:b])]
    speech_s = max((b - a) * HOP / SR - sum(p[1] for p in pauses), 1e-3)
    return {"f0_std": round(float(v.std()), 3) if len(v) else 0.0,
            "f0_p10": round(float(np.percentile(v, 10)), 3) if len(v) else 0.0,
            "f0_p90": round(float(np.percentile(v, 90)), 3) if len(v) else 0.0,
            "voiced": round(len(v) / max(b - a, 1), 3), "pauses": pauses, "n_pause": len(pauses),
            "pause_s": round(sum(p[1] for p in pauses), 2), "rate": round(n_words / speech_s, 2)}


def dtw_features(st: np.ndarray, db: np.ndarray) -> np.ndarray:
    """``[2, T]`` at 20 ms: F0 with unvoiced frames interpolated, and energy re its median in units of ``E_SCALE`` dB."""
    _, a, b = speech_span(db)
    st, db = st[a:b], db[a:b]
    ok = ~np.isnan(st)
    f = np.interp(np.arange(len(st)), np.flatnonzero(ok), st[ok]) if ok.any() else np.zeros(len(st))
    e = (db - np.median(db)) / E_SCALE
    m = len(f) // 2 * 2
    return np.stack([f[:m].reshape(-1, 2).mean(1), e[:m].reshape(-1, 2).mean(1)])


def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    import librosa

    if x.shape[1] == 0 or y.shape[1] == 0:
        return float("nan")
    D, wp = librosa.sequence.dtw(X=x, Y=y, metric="euclidean")
    return float(D[-1, -1] / len(wp))


def _contours(path):
    return contours(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="directory with meta.jsonl and wavs from omnivoice_dpo.generate")
    ap.add_argument("--feat", required=True, help="directory with asr.jsonl from omnivoice_dpo.rewards.features")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    meta = [json.loads(l) for l in open(os.path.join(a.run, "meta.jsonl"), encoding="utf-8") if l.strip()]
    asr = {r["id"]: r for r in map(json.loads, open(os.path.join(a.feat, "asr.jsonl"), encoding="utf-8"))}
    cpath = os.path.join(a.feat, "contours.npz")
    if os.path.exists(cpath):
        z = np.load(cpath)
        cont = {k: (z[k][0], z[k][1]) for k in z.files}
    else:
        with mp.Pool(a.workers) as pool:
            got = pool.map(_contours, [os.path.join(a.run, m["wav"]) for m in meta], chunksize=4)
        cont = {f"{m['id']}.{m['k']}": c for m, c in zip(meta, got)}
        np.savez_compressed(cpath, **{k: np.stack(v) for k, v in cont.items()})
    groups = collections.defaultdict(list)
    with open(os.path.join(a.feat, "prosody.jsonl"), "w", encoding="utf-8") as f:
        for m in meta:
            key = f"{m['id']}.{m['k']}"
            st, db = cont[key]
            row = {"id": m["id"], "k": m["k"], "dur": round(len(db) * HOP / SR, 2),
                   "err": round(content_error(m["text"], asr.get(key)), 4), **clip_stats(st, db, len(toks(m["text"]))),
                   "asr": asr[key]["text"] if key in asr else ""}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            groups[m["id"]].append(m["k"])
    with open(os.path.join(a.feat, "dist.jsonl"), "w", encoding="utf-8") as f:
        for pid, ks in groups.items():
            ks = sorted(ks)
            feats = {k: dtw_features(*cont[f"{pid}.{k}"]) for k in ks}
            dist = [[0.0 if i == j else round(dtw_distance(feats[i], feats[j]), 4) for j in ks] for i in ks]
            f.write(json.dumps({"id": pid, "ks": ks, "dist": dist}) + "\n")
    print(f"prosody: {len(meta)} clips, {len(groups)} prompts -> {a.feat}/prosody.jsonl, dist.jsonl")


if __name__ == "__main__":
    main()
