"""Raw per-clip features for tag checking: ASR with word timestamps and AudioSet event probabilities over time.

Features are stored, not scores, so thresholds and rules can change without touching the GPU again:

- ``asr.jsonl``: ``{"id", "text", "words": [[word, start_s, end_s], ...]}`` from Whisper.
- ``ast.npz``: per clip an array ``[n_windows, 1 + n_classes]``; column 0 is the window centre (s), the rest are sigmoid
  probabilities of :data:`AST_KEEP`. AST sees the whole clip as "Speech", so it runs on 1.0 s windows with a 0.25 s hop.

    python -m omnivoice_dpo.rewards.features --wav-dir run/ --out run/feat
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

AST_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
ASR_ID = "openai/whisper-large-v3-turbo"
AST_KEEP = [
    "Speech", "Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle", "Crying, sobbing", "Whimper", "Sigh",
    "Groan", "Grunt", "Breathing", "Wheeze", "Snoring", "Gasp", "Pant", "Cough", "Throat clearing", "Sneeze", "Sniff",
    "Humming", "Cheering", "Shout", "Yell", "Screaming", "Whispering", "Clicking", "Hiccup", "Silence",
]


def run_asr(ids, paths, out_path, model_id=ASR_ID, language="vi", batch=16, max_new_tokens=400):
    import librosa
    import torch
    from transformers import pipeline

    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8") if l.strip()}
    todo = [(i, p) for i, p in zip(ids, paths) if i not in done]
    if not todo:
        return
    asr = pipeline("automatic-speech-recognition", model=model_id, dtype=torch.float16, device=0)
    gk = {"language": language, "task": "transcribe", "max_new_tokens": max_new_tokens}
    with open(out_path, "a", encoding="utf-8") as f:
        for k in range(0, len(todo), batch):
            items = todo[k:k + batch]
            audio = [{"raw": librosa.load(p, sr=16000)[0], "sampling_rate": 16000} for _, p in items]
            outs = asr(audio, batch_size=len(items), return_timestamps="word", generate_kwargs=gk)
            for (i, _), o in zip(items, outs):
                words = [[c["text"].strip(), c["timestamp"][0], c["timestamp"][1]] for c in (o.get("chunks") or [])]
                f.write(json.dumps({"id": i, "text": o["text"].strip(), "words": words}, ensure_ascii=False) + "\n")
            f.flush()
    del asr
    torch.cuda.empty_cache()


def run_ast(ids, paths, out_path, model_id=AST_ID, win_s=1.0, hop_s=0.25, chunk=32):
    import librosa
    import torch
    from transformers import ASTForAudioClassification, AutoFeatureExtractor

    if os.path.exists(out_path):
        return
    fe = AutoFeatureExtractor.from_pretrained(model_id)
    m = ASTForAudioClassification.from_pretrained(model_id).cuda().eval().half()
    lab2id = {v: int(k) for k, v in m.config.id2label.items()}
    keep = [lab2id[c] for c in AST_KEEP]
    W, H = int(win_s * 16000), int(hop_s * 16000)
    store = {}
    for i, p in zip(ids, paths):
        y, _ = librosa.load(p, sr=16000)
        if len(y) < W:
            y = np.pad(y, (0, W - len(y)))
        starts = list(range(0, len(y) - W + 1, H)) or [0]
        probs = []
        for b in range(0, len(starts), chunk):  # AST pads every window to 10.24 s, so keep batches small
            x = fe([y[s:s + W] for s in starts[b:b + chunk]], sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                logits = m(input_values=x["input_values"].cuda().half()).logits
            probs.append(torch.sigmoid(logits.float())[:, keep].cpu().numpy())
        centres = np.array(starts, np.float32)[:, None] / 16000 + win_s / 2
        store[i] = np.concatenate([centres, np.concatenate(probs, 0)], axis=1)
    np.savez_compressed(out_path, classes=np.array(AST_KEEP), **store)
    del m
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wav-dir", required=True, help="directory with meta.jsonl from omnivoice_dpo.generate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--asr-model", default=ASR_ID)
    ap.add_argument("--language", default="vi")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = [json.loads(l) for l in open(os.path.join(a.wav_dir, "meta.jsonl"), encoding="utf-8") if l.strip()]
    ids = [f"{m['id']}.{m['k']}" for m in meta]
    paths = [os.path.join(a.wav_dir, m["wav"]) for m in meta]
    run_asr(ids, paths, os.path.join(a.out, "asr.jsonl"), a.asr_model, a.language)
    run_ast(ids, paths, os.path.join(a.out, "ast.npz"))
    with open(os.path.join(a.out, "models.json"), "w") as f:
        json.dump({"asr": a.asr_model, "ast": AST_ID}, f)


if __name__ == "__main__":
    main()
