"""Does the model follow its prosody codes? Generate held-out utterances several ways and compare their syllable prosody
with the original recording.

For each selected utterance (dev lists of a data config, aligned in the store, 2-15 s, no tags), with the same voice
prompt (a voicebank voice chosen by id hash), the original's length and the same seed:

- ``base``: the model without prosody (``--base``, optional);
- ``null``: the prosody model with no codes;
- ``oracle``: codes encoded from the utterance's own recording;
- ``shuffle``: the same codes in a random syllable order (right distribution, wrong place).

Every output is decoded, re-aligned and measured with the prep extractor. Per syllable: mean F0 (semitones re the
utterance median, voiced frames), log duration, pause after. Per utterance, against the original: Pearson r of F0 and of
log duration, F1 of pauses (at least ``PAUSE`` frames), mean CTC confidence (an intelligibility proxy). Summary: means and
paired bootstrap 95 % CIs (4000 resamples) of each condition minus ``null``. Tones make a syllable's F0 largely a
function of its text, so even the plain model correlates with the recording; ``follow_f0`` / ``follow_dur`` isolate the
codes' effect: per utterance, r between (condition - null) and (original - null) over syllables (0 when the condition
moves at random, as ``base`` and ``shuffle`` should).

    python -m omnivoice_dpo.prosody.evaluate --model <ckpt dir with audio_tokenizer/> [--base <ckpt>] --data-config data.json \\
        --store store/ --ctc <CTC model> --voices-dir voices/ --out out/ [--datasets a_dev,b_dev] [--n 40] [--wav 15]
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch

from omnivoice_dpo.prosody.data import ProsodyStore
from omnivoice_dpo.prosody.prep import Extractor, iter_shard
from omnivoice_dpo.prosody.text import has_unalignable

PAUSE = 3  # frames (120 ms)


def syl_stats(feats: np.ndarray, syl) -> np.ndarray:
    """``[n, 4]``: mean F0 st over voiced frames (NaN if none), log(1 + duration frames), pause frames, mean energy (dB
    re the utterance maximum)."""
    out = []
    for s in syl:
        f = feats[int(s[2]):int(s[3])]
        v = f[:, 1] > 0.5
        out.append([float(f[v, 0].mean()) if v.any() else np.nan, np.log1p(s[3] - s[2]), s[4], float(f[:, 2].mean())])
    return np.asarray(out, dtype=np.float64)


def pearson(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 4 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def pause_f1(a, b):
    pa, pb = a[:-1, 2] >= PAUSE, b[:-1, 2] >= PAUSE
    tp = (pa & pb).sum()
    if pa.sum() + pb.sum() == 0:
        return np.nan
    return float(2 * tp / (pa.sum() + pb.sum()))


def boot(d, n=4000, seed=0):
    d = np.asarray([x for x in d if not np.isnan(x)])
    if len(d) < 3:
        return [np.nan, np.nan, np.nan]
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    from omnivoice import OmniVoice, OmniVoiceGenerationConfig

    from omnivoice_dpo.generate import load_voices, pick_voice
    from omnivoice_dpo.prosody.infer import codes_from_store, generate
    from omnivoice_dpo.prosody.model import OmniVoiceProsody

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--base")
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--ctc", required=True)
    ap.add_argument("--voices-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", default="style_tts_metadata_dev,production_voice_dev")
    ap.add_argument("--n", type=int, default=40, help="utterances per dataset")
    ap.add_argument("--wav", type=int, default=15, help="write wavs for this many utterances")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)

    # utterances
    store = ProsodyStore(a.store)
    want = set(a.datasets.split(","))
    lists = []
    cfgj = json.load(open(a.data_config))
    for d in cfgj.get("dev", []) + cfgj.get("train", []):
        for lst in d["manifest_path"]:
            if lst.split("/dataset/")[1].split("/")[0] in want:
                lists.append(lst)
    per_ds = []
    for name in sorted(want):
        cands = []
        for lst in [x for x in lists if f"/dataset/{name}/" in x]:
            for line in open(lst):
                p = line.split()
                for r in map(json.loads, open(p[1], encoding="utf-8")):
                    t, T = r.get("text", ""), r.get("num_tokens", 0)
                    if "[" in t or has_unalignable(t) or not 50 <= T <= 375:
                        continue
                    got = store.lookup(r["id"], t, T)
                    if got is not None and got[2].mean() >= 0.5 and len(got[1]) >= 5:
                        cands.append((name, r["id"], t, T, p[0]))
        per_ds.append(rng.sample(cands, min(a.n, len(cands))))
    # interleave datasets, so the first --wav utterances (written as audio) cover all of them
    picked = [x for i in range(max(map(len, per_ds), default=0)) for d in per_ds if i < len(d) for x in [d[i]]]
    print(f"### {len(picked)} utterances from {sorted(want)}", flush=True)

    model = OmniVoiceProsody.from_pretrained(a.model, dtype=torch.float16).cuda().eval()
    base = OmniVoice.from_pretrained(a.base, dtype=torch.float16).cuda().eval() if a.base else None
    voices = load_voices(model, a.voices_dir)
    names = sorted(voices)
    ex = Extractor(a.model, a.ctc)
    cfg = OmniVoiceGenerationConfig(postprocess_output=False)
    import soundfile as sf

    toks = {}
    for tar in sorted({x[4] for x in picked}):
        need = {x[1] for x in picked if x[4] == tar}
        jsonl = next(line.split()[1] for lst in lists for line in open(lst) if line.split()[0] == tar)
        for key, tok, _ in iter_shard(tar, jsonl):
            if key in need:
                toks[key] = tok

    rows = []
    fo = open(os.path.join(a.out, "per_utt.jsonl"), "w", encoding="utf-8")
    for j, (name, sid, text, T, _) in enumerate(picked):
        feats, syl, _conf = store.lookup(sid, text, T)
        ref = syl_stats(feats, syl)
        codes = codes_from_store(model, feats, syl)
        keep = torch.ones(len(codes), device=codes.device)
        perm = torch.randperm(len(codes), generator=torch.Generator().manual_seed(j))
        vname = pick_voice(sid, names)
        conds = {"null": None, "oracle": model.prosody_encoder.embed_codes(codes, keep),
                 "shuffle": model.prosody_encoder.embed_codes(codes[perm.to(codes.device)], keep)}
        res = {"dataset": name, "id": sid, "voice": vname, "n_syl": len(syl), "text": text,
               "distinct_codes": int(torch.unique(codes, dim=0).shape[0])}
        wavs = {}
        if j < a.wav:
            wavs["orig"] = ex.codec.decode(torch.from_numpy(toks[sid].astype(np.int64))[None].cuda()).audio_values.reshape(-1)
        todo = ([("base", base, None)] if base is not None else []) + [(k, model, v) for k, v in conds.items()]
        for k, m, emb in todo:
            if m is model:
                out = generate(model, text, voices[vname], T, cfg, syl_emb=None if emb is None else emb.half(), seed=a.seed)
            else:
                torch.manual_seed(a.seed)
                task = base._preprocess_all(text=[text], language=["vi"], voice_clone_prompt=[voices[vname]],
                                            preprocess_prompt=cfg.preprocess_prompt)
                task.target_lens = [T]
                out = base._generate_iterative(task, cfg)[0]
            got = ex(out.cpu().numpy(), text)
            if got[0] is None:
                res[k] = None
                continue
            gf, gs, wav = got
            g = syl_stats(gf, gs)
            res[k] = {"r_f0": pearson(ref[:, 0], g[:, 0]), "r_dur": pearson(ref[:, 1], g[:, 1]),
                      "pause_f1": pause_f1(ref, g), "conf": float(np.mean([s[5] for s in gs])), "syl": g.tolist()}
            if j < a.wav:
                wavs[k] = wav
        res["orig_syl"] = ref.tolist()
        # does a condition move away from the plain model in the direction of the recording? per syllable,
        # r(cond - null, orig - null), for F0 and log duration
        if res.get("null"):
            nl = np.asarray(res["null"]["syl"], dtype=np.float64)
            for k in [k for k in res if k in ("base", "oracle", "shuffle") and res[k]]:
                c = np.asarray(res[k]["syl"], dtype=np.float64)
                res[k]["follow_f0"] = pearson(c[:, 0] - nl[:, 0], ref[:, 0] - nl[:, 0])
                res[k]["follow_dur"] = pearson(c[:, 1] - nl[:, 1], ref[:, 1] - nl[:, 1])
        for k, w in wavs.items():
            sf.write(os.path.join(a.out, f"{j:03d}_{k}.wav"), w.detach().float().cpu().numpy(), 24000)
        fo.write(json.dumps(res, ensure_ascii=False) + "\n")
        fo.flush()
        rows.append(res)
        if (j + 1) % 10 == 0:
            print(f"### {j + 1}/{len(picked)}", flush=True)

    keys = ["base", "null", "oracle", "shuffle"] if base is not None else ["null", "oracle", "shuffle"]
    summary = {}
    for met in ("r_f0", "r_dur", "pause_f1", "conf"):
        summary[met] = {}
        for k in keys:
            vals = [r[k][met] if r.get(k) else np.nan for r in rows]
            summary[met][k] = boot(vals)
            if k != "null":
                summary[met][f"{k}-null"] = boot([v - (r["null"][met] if r.get("null") else np.nan)
                                                  for v, r in zip(vals, rows)])
    for met in ("follow_f0", "follow_dur"):
        summary[met] = {k: boot([r[k].get(met, np.nan) if r.get(k) else np.nan for r in rows])
                        for k in keys if k != "null"}
    summary["n"] = len(rows)
    summary["distinct_codes_per_syl"] = float(np.mean([r["distinct_codes"] / r["n_syl"] for r in rows]))
    summary["failed"] = {k: sum(1 for r in rows if not r.get(k)) for k in keys}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    for met, d in summary.items():
        if isinstance(d, dict) and met != "failed":
            print(f"### {met}: " + " · ".join(f"{k} {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" for k, v in d.items()), flush=True)
    print(f"### n {len(rows)}, failed {summary['failed']}, distinct codes / syllables "
          f"{summary['distinct_codes_per_syl']:.2f}", flush=True)


if __name__ == "__main__":
    main()
