"""Turn listener labels into training records.

A label says, for one prompt, which of two candidates (``ka``/``kb``) was preferred: ``A``, ``B``, ``tie`` or ``bad``.
Only ``A``/``B`` become pairs. A repeated item (``lap_of``) is a consistency check: if its verdict disagrees with the
original, the pair is dropped; otherwise it is counted once.

Each record carries the conditioning prefix rebuilt exactly as at generation time (style, text, reference-voice tokens),
checked against the candidate's stored ``cond_hash``, so training scores the very context the tokens were sampled in.

    python -m omnivoice_dpo.data --model <ckpt> --cands run/cands --voices-dir voices/ --labels labels.jsonl --out records.pt
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np
import torch

from omnivoice_dpo.generate import cond_hash, load_voices


def resolve_labels(labels: list[dict]) -> tuple[list[dict], dict]:
    by_iid = {l["iid"]: l for l in labels}
    stats = collections.Counter()
    out = {}
    for l in labels:
        if l.get("lap_of"):
            continue
        stats[l["choice"]] += 1
        if l["choice"] not in ("A", "B"):
            continue
        win, lose = (l["ka"], l["kb"]) if l["choice"] == "A" else (l["kb"], l["ka"])
        rep = next((r for r in labels if r.get("lap_of") == l["iid"]), None)
        if rep is not None:
            rw = (rep["ka"], rep["kb"]) if rep["choice"] == "A" else (rep["kb"], rep["ka"]) if rep["choice"] == "B" else None
            stats["lap_" + ("same" if rw == (win, lose) else "diff")] += 1
            if rw != (win, lose):
                continue
        out[l["pair"]] = {"pair": l["pair"], "chosen": win, "rejected": lose, "iid": l["iid"]}
    stats["pairs"] = len(out)
    return list(out.values()), dict(stats)


@torch.inference_mode()
def build_records(model, cands_dir: str, voices: dict, pairs: list[dict], lang: str = "vi") -> list[dict]:
    meta = {(m["id"], m["k"]): m for m in map(json.loads, open(os.path.join(cands_dir, "meta.jsonl"), encoding="utf-8"))}
    recs = []
    for p in pairs:
        mc, mr = meta[(p["pair"], p["chosen"])], meta[(p["pair"], p["rejected"])]
        vcp = voices[mc["voice"]]
        assert mc["cond_hash"] == mr["cond_hash"] == cond_hash(mc["text"], mc["lang"], mc["voice"], vcp), p["pair"]
        T = mc["target_len"]
        inp = model._prepare_inference_inputs(mc["text"], T, vcp.ref_text, vcp.ref_audio_tokens, lang, None, True)
        ids, am = inp["input_ids"][0].cpu(), inp["audio_mask"][0].cpu()
        xc = torch.from_numpy(np.load(os.path.join(cands_dir, mc["tokens"])).astype(np.int64))
        xr = torch.from_numpy(np.load(os.path.join(cands_dir, mr["tokens"])).astype(np.int64))
        assert xc.shape == xr.shape == (ids.shape[0], T), p["pair"]
        recs.append({"pair": p["pair"], "prefix": ids[:, :-T].to(torch.int32), "audio_start": int(am.nonzero()[0, 0]),
                     "chosen": xc.to(torch.int16), "rejected": xr.to(torch.int16)})
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="the checkpoint that generated the candidates")
    ap.add_argument("--cands", required=True)
    ap.add_argument("--voices-dir", required=True)
    ap.add_argument("--labels", required=True, help="jsonl: iid, pair, ka, kb, choice, lap_of")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from omnivoice import OmniVoice

    labels = [json.loads(l) for l in open(a.labels, encoding="utf-8") if l.strip()]
    pairs, stats = resolve_labels(labels)
    print("labels:", stats)
    model = OmniVoice.from_pretrained(a.model, device_map="cuda", dtype=torch.float16)
    voices = load_voices(model, a.voices_dir, sorted({json.loads(l)["voice"] for l in
                                                       open(os.path.join(a.cands, "meta.jsonl"), encoding="utf-8")}))
    recs = build_records(model, a.cands, voices, pairs)
    torch.save({"records": recs, "stats": stats, "source": os.path.abspath(a.cands)}, a.out)
    print(f"{len(recs)} records -> {a.out}")


if __name__ == "__main__":
    main()
