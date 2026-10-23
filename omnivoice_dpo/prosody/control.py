"""Manual control: give one syllable a hand-set code (the other syllables get none) and measure what changes there.

The code dims of a trained encoder line up with simple prosodic quantities (see the README / analysis): which level means
"high pitch" etc. is passed as ``--edits``, a JSON object ``{name: [level per dim, with null for "neutral"]}``; neutral
levels are ``--neutral``. For every text (dev utterances of a data config, as in :mod:`.evaluate`), the middle syllable of
the sentence is edited; the output is decoded, re-aligned and measured against the uncoded generation (same voice, length,
seed): change of the target syllable's F0 (semitones), log duration, pause after (frames), and of its neighbours (to see
whether the change stays local), plus the CTC confidence of the whole sentence.

    python -m omnivoice_dpo.prosody.control --model <ckpt dir with audio_tokenizer/> --data-config data.json \\
        --ctc <CTC model> --voices-dir voices/ --out out/ --neutral '[4,0,1,2]' \\
        --edits '{"pitch_up": [0,null,null,null], "pitch_down": [7,null,null,null], "long": [null,null,null,4]}'
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch

from omnivoice_dpo.prosody.evaluate import boot, syl_stats
from omnivoice_dpo.prosody.text import has_unalignable, syllables


def main():
    from omnivoice import OmniVoiceGenerationConfig

    from omnivoice_dpo.generate import load_voices, pick_voice
    from omnivoice_dpo.prosody.infer import generate
    from omnivoice_dpo.prosody.model import OmniVoiceProsody
    from omnivoice_dpo.prosody.prep import Extractor

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--ctc", required=True)
    ap.add_argument("--voices-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", default="style_tts_metadata_dev,production_voice_dev")
    ap.add_argument("--n", type=int, default=30, help="texts per dataset")
    ap.add_argument("--neutral", required=True)
    ap.add_argument("--edits", required=True)
    ap.add_argument("--wav", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    neutral, edits = json.loads(a.neutral), json.loads(a.edits)
    rng = random.Random(a.seed)

    want = set(a.datasets.split(","))
    cfgj = json.load(open(a.data_config))
    per_ds = []
    for name in sorted(want):
        cands = []
        for d in cfgj.get("dev", []) + cfgj.get("train", []):
            for lst in d["manifest_path"]:
                if f"/dataset/{name}/" not in lst:
                    continue
                for line in open(lst):
                    for r in map(json.loads, open(line.split()[1], encoding="utf-8")):
                        t, T = r.get("text", ""), r.get("num_tokens", 0)
                        if "[" not in t and not has_unalignable(t) and 75 <= T <= 300 and len(syllables(t)) >= 7:
                            cands.append((name, r["id"], t, T))
        per_ds.append(rng.sample(cands, min(a.n, len(cands))))
    picked = [x for i in range(max(map(len, per_ds), default=0)) for d in per_ds if i < len(d) for x in [d[i]]]
    print(f"### {len(picked)} texts", flush=True)

    model = OmniVoiceProsody.from_pretrained(a.model, dtype=torch.float16).cuda().eval()
    enc = model.prosody_encoder
    voices = load_voices(model, a.voices_dir)
    names = sorted(voices)
    ex = Extractor(a.model, a.ctc)
    cfg = OmniVoiceGenerationConfig(postprocess_output=False)
    import soundfile as sf

    rows = []
    fo = open(os.path.join(a.out, "per_text.jsonl"), "w", encoding="utf-8")
    for j, (name, sid, text, T) in enumerate(picked):
        n = len(syllables(text))
        tgt = n // 2
        vname = pick_voice(sid, names)
        res = {"dataset": name, "id": sid, "text": text, "voice": vname, "target": tgt,
               "target_syl": text[slice(*syllables(text)[tgt])]}
        outs = {}
        for cond in ["null", *edits]:
            emb = None
            if cond != "null":
                lv = [neutral[i] if v is None else v for i, v in enumerate(edits[cond])]
                code = enc.fsq.from_indices(torch.tensor([lv], device=model.device))
                keep = torch.zeros(n, device=model.device)
                keep[tgt] = 1
                emb = enc.embed_codes(code.expand(n, -1), keep)
            tok = generate(model, text, voices[vname], T, cfg, syl_emb=emb, seed=a.seed)
            got = ex(tok.cpu().numpy(), text)
            if got[0] is None:
                res[cond] = None
                continue
            gf, gs, wav = got
            outs[cond] = (syl_stats(gf, gs), float(np.mean([s[5] for s in gs])))
            res[cond] = {"conf": outs[cond][1]}
            if j < a.wav:
                sf.write(os.path.join(a.out, f"{j:03d}_{cond}.wav"), wav.detach().float().cpu().numpy(), 24000)
        if "null" in outs:
            base = outs["null"][0]
            for cond, (st, _) in outs.items():
                if cond == "null":
                    continue
                nb = [i for i in (tgt - 1, tgt + 1) if 0 <= i < n]
                res[cond].update(d_f0=st[tgt, 0] - base[tgt, 0], d_dur=st[tgt, 1] - base[tgt, 1],
                                 d_pause=st[tgt, 2] - base[tgt, 2],
                                 d_f0_nb=float(np.nanmean(st[nb, 0] - base[nb, 0])),
                                 d_dur_nb=float(np.nanmean(st[nb, 1] - base[nb, 1])))
        fo.write(json.dumps(res, ensure_ascii=False, default=float) + "\n")
        fo.flush()
        rows.append(res)
        if (j + 1) % 10 == 0:
            print(f"### {j + 1}/{len(picked)}", flush=True)

    summary = {"n": len(rows), "neutral": neutral, "edits": edits}
    for cond in edits:
        summary[cond] = {m: boot([r[cond].get(m, np.nan) if r.get(cond) else np.nan for r in rows])
                         for m in ("d_f0", "d_dur", "d_pause", "d_f0_nb", "d_dur_nb")}
        summary[cond]["conf-null"] = boot([r[cond]["conf"] - r["null"]["conf"] if r.get(cond) and r.get("null") else np.nan
                                           for r in rows])
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    for cond in edits:
        print(f"### {cond}: " + " · ".join(f"{m} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"
                                          for m, v in summary[cond].items()), flush=True)


if __name__ == "__main__":
    main()
