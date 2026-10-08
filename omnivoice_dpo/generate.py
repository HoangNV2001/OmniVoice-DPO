"""Sample K candidates per prompt from an OmniVoice checkpoint and keep the exact token ids.

All K candidates of a prompt share the conditioning (text, language, voice prompt) and therefore the target length; only
the seed differs. Tokens are written as int16 ``[C, T]`` ``.npy`` files, next to a 24 kHz wav decoded from exactly those
tokens, plus one ``meta.jsonl`` row per candidate.

    python -m omnivoice_dpo.generate --model <ckpt> --prompts prompts.jsonl --voices-dir voices/ --k 4 --out run/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import soundfile as sf
import torch
from omnivoice import OmniVoice, OmniVoiceGenerationConfig


def load_voices(model: OmniVoice, voices_dir: str, names: list[str] | None = None) -> dict:
    """``<name>.wav`` + ``<name>.txt`` (transcript) → reusable voice-clone prompts."""
    out = {}
    for f in sorted(os.listdir(voices_dir)):
        name, ext = os.path.splitext(f)
        if ext != ".wav" or (names and name not in names):
            continue
        txt = os.path.join(voices_dir, name + ".txt")
        ref_text = open(txt, encoding="utf-8").read().strip() if os.path.exists(txt) else None
        out[name] = model.create_voice_clone_prompt(ref_audio=os.path.join(voices_dir, f), ref_text=ref_text)
    if not out:
        raise SystemExit(f"no voices found in {voices_dir}")
    return out


def pick_voice(prompt_id: str, names: list[str]) -> str:
    return names[int(hashlib.md5(prompt_id.encode()).hexdigest(), 16) % len(names)]


def cond_hash(text: str, lang: str | None, voice: str, vcp) -> str:
    h = hashlib.md5(f"{text}\x00{lang}\x00{voice}\x00{vcp.ref_text}".encode())
    h.update(vcp.ref_audio_tokens.cpu().numpy().astype(np.int16).tobytes())
    return h.hexdigest()


@torch.inference_mode()
def generate_candidates(model, rows, voices, k, seed, batch_size, cfg, out_dir, model_tag=""):
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.jsonl")
    done = set()
    if os.path.exists(meta_path):
        done = {(m["id"], m["k"]) for m in map(json.loads, open(meta_path, encoding="utf-8"))}
    names = sorted(voices)
    for r in rows:
        r.setdefault("voice", pick_voice(r["id"], names))
    # Batches are built once (sorted by text length) and reused for every k, so a candidate is reproducible from
    # (seed, k, batch index).
    rows = sorted(rows, key=lambda r: len(r["text"]))
    batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
    gen = {f: getattr(cfg, f) for f in ("num_step", "guidance_scale", "t_shift", "position_temperature",
                                         "class_temperature", "layer_penalty_factor", "postprocess_output")}
    with open(meta_path, "a", encoding="utf-8") as fmeta:
        for kk in range(k):
            for bi, batch in enumerate(batches):
                todo = [r for r in batch if (r["id"], kk) not in done]
                if not todo:
                    continue
                if len(todo) != len(batch):  # partial batch after an interruption: regenerate it whole
                    todo = batch
                s = seed + 100003 * kk + bi
                torch.manual_seed(s)
                vcps = [voices[r["voice"]] for r in batch]
                task = model._preprocess_all(text=[r["text"] for r in batch], language=[r.get("lang", "vi")] * len(batch),
                                             voice_clone_prompt=vcps, preprocess_prompt=cfg.preprocess_prompt)
                toks = model._generate_iterative(task, cfg)
                for i, (r, tok) in enumerate(zip(batch, toks)):
                    if (r["id"], kk) in done:
                        continue
                    stem = f"{r['id']}.{kk}"
                    np.save(os.path.join(out_dir, stem + ".npy"), tok.cpu().numpy().astype(np.int16))
                    wav = model._decode_and_post_process(tok, task.ref_rms[i], cfg)
                    sf.write(os.path.join(out_dir, stem + ".wav"), wav, model.sampling_rate)
                    fmeta.write(json.dumps(dict(
                        id=r["id"], k=kk, text=r["text"], lang=r.get("lang", "vi"), voice=r["voice"], seed=s, batch=bi,
                        target_len=int(task.target_lens[i]), cond_hash=cond_hash(r["text"], r.get("lang", "vi"),
                                                                                  r["voice"], vcps[i]),
                        tokens=stem + ".npy", wav=stem + ".wav", model=model_tag, gen=gen), ensure_ascii=False) + "\n")
                    done.add((r["id"], kk))
                fmeta.flush()
            print(f"candidate {kk + 1}/{k} done", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help="jsonl with id, text and optional lang / voice")
    ap.add_argument("--ids", help="optional file with one prompt id per line to restrict to")
    ap.add_argument("--voices-dir", required=True)
    ap.add_argument("--voices", help="comma-separated subset of voice names")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-step", type=int, default=32)
    ap.add_argument("--class-temperature", type=float, default=0.0)
    ap.add_argument("--postprocess", action="store_true", help="trim long silences in the wav (tokens are unaffected)")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.prompts, encoding="utf-8") if l.strip()]
    if a.ids:
        keep = {l.strip() for l in open(a.ids, encoding="utf-8") if l.strip()}
        rows = [r for r in rows if r["id"] in keep]
    model = OmniVoice.from_pretrained(a.model, device_map="cuda", dtype=getattr(torch, a.dtype))
    voices = load_voices(model, a.voices_dir, a.voices.split(",") if a.voices else None)
    cfg = OmniVoiceGenerationConfig(num_step=a.num_step, class_temperature=a.class_temperature,
                                    postprocess_output=a.postprocess)
    print(f"{len(rows)} prompts x {a.k} candidates, {len(voices)} voices", flush=True)
    generate_candidates(model, rows, voices, a.k, a.seed, a.batch_size, cfg, a.out, model_tag=os.path.abspath(a.model))


if __name__ == "__main__":
    main()
