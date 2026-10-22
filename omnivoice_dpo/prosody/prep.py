"""Frame-level prosodic features and syllable alignment for training data (run once, on GPU).

For every sample of the selected datasets of a data config:

1. decode the stored audio tokens with the model's audio tokenizer (24 kHz), so the features describe exactly what the
   model learns to reproduce;
2. features per audio-token frame (25 Hz), :data:`FEATS`: F0 in semitones re the utterance's median voiced F0 (CREPE,
   0 when unvoiced), voiced fraction, energy in dB re the utterance maximum, 8 log-mel bands minus their utterance mean;
3. syllable spans by CTC forced alignment of the label's letters on the decoded audio, in token frames, with the mean CTC
   posterior of the syllable's characters as a confidence.

Output per dataset in ``<out>/<dataset>/``: ``feats.f16`` (rows of :data:`FEATS` float16, concatenated) and
``index.jsonl``, one line per sample::

    {"id", "md5" (of the label text), "off" (first row), "T", "syl": [[char_start, char_end, frame_start, frame_end, conf], ...]}

``syl`` is null (and nothing is decoded) when the text cannot be aligned: IPA spans, digits, a character outside the CTC
vocabulary, or a failed alignment. Shards listed in ``done.txt`` are skipped, so the run can resume.

    python -m omnivoice_dpo.prosody.prep --data-config data.json --model <ckpt with audio_tokenizer/> --ctc <CTC model> \\
        --out store/ [--datasets a,b] [--limit 200] [--check 5]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import time

import numpy as np
import torch

from omnivoice_dpo.prosody.text import has_unalignable, norm, syllables

FEATS = ["f0_st", "voiced", "energy_db", *[f"mel{i}" for i in range(8)]]
FPS, SR, HOP = 25, 24000, 960


def dataset_name(lst: str) -> str:
    return lst.split("/dataset/")[1].split("/")[0]


def manifests(cfg_path: str, wanted: set[str] | None) -> dict[str, list[str]]:
    """``{dataset: [lst, ...]}`` over train and dev; English datasets are left out unless named."""
    out = {}
    cfg = json.load(open(cfg_path))
    for phase in ("train", "dev"):
        for d in cfg.get(phase, []):
            for lst in d["manifest_path"]:
                name = dataset_name(lst)
                if (wanted and name in wanted) or (not wanted and d.get("language_id") == "vi"):
                    out.setdefault(name, []).append(lst)
    return out


def iter_shard(tar_path: str, jsonl_path: str):
    labels = {}
    for line in open(jsonl_path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            labels[r["id"]] = r
    with tarfile.open(tar_path) as tf:
        for m in tf:
            if not m.name.endswith(".npy"):
                continue
            key = m.name[:-4]
            if key in labels:
                yield key, np.load(io.BytesIO(tf.extractfile(m).read())), labels[key]


class Extractor:
    def __init__(self, model_dir: str, ctc_id: str, device: str = "cuda"):
        import torchaudio
        from omnivoice.models.omnivoice import HiggsAudioV2TokenizerModel
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        self.device = device
        self.codec = HiggsAudioV2TokenizerModel.from_pretrained(os.path.join(model_dir, "audio_tokenizer"),
                                                                device_map=device).eval()
        self.ctc = Wav2Vec2ForCTC.from_pretrained(ctc_id).to(device).eval()
        self.ctc_fe = Wav2Vec2FeatureExtractor.from_pretrained(ctc_id)
        vocab = json.load(open(os.path.join(_snapshot(ctc_id), "vocab.json")))
        self.vocab, self.blank, self.sep = vocab, self.ctc.config.pad_token_id, vocab["|"]
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=2048, win_length=1920, hop_length=HOP,
                                                        n_mels=8, f_min=50, f_max=8000).to(device)

    def targets(self, text: str, spans):
        ids, owner = [], []
        for i, (a, b) in enumerate(spans):
            if i:
                ids.append(self.sep)
                owner.append(-1)
            for ch in norm(text[a:b]):
                if ch not in self.vocab:
                    return None, None
                ids.append(self.vocab[ch])
                owner.append(i)
        return ids, owner

    @torch.inference_mode()
    def __call__(self, tokens: np.ndarray, text: str):
        """``(feats [T, F] float16, syl list, decoded wav)``, or ``(None, None)`` when the text cannot be aligned."""
        import torchaudio.functional as AF
        import torchcrepe

        spans = syllables(text)
        if not spans or has_unalignable(text):
            return None, None
        tgt, owner = self.targets(text, spans)
        if tgt is None:
            return None, None
        T = tokens.shape[1]
        wav = self.codec.decode(torch.from_numpy(tokens.astype(np.int64))[None].to(self.device)).audio_values
        wav = wav.reshape(-1).float()[: T * HOP]
        wav = torch.nn.functional.pad(wav, (0, T * HOP - wav.numel()))
        w16 = AF.resample(wav, SR, 16000)

        # CTC forced alignment, 20 ms frames
        x = w16
        if self.ctc_fe.do_normalize:
            x = (x - x.mean()) / torch.sqrt(x.var() + 1e-7)
        lp = self.ctc(x[None]).logits.float().log_softmax(-1)
        try:
            ali, sc = AF.forced_align(lp, torch.tensor([tgt], device=self.device), blank=self.blank)
        except RuntimeError:
            return None, None
        tok_spans = AF.merge_tokens(ali[0], sc[0].exp())
        if len(tok_spans) != len(tgt):
            return None, None
        per = {}
        for s, o in zip(tok_spans, owner):
            if o >= 0:
                per.setdefault(o, []).append(s)
        syl = []
        for i, (a, b) in enumerate(spans):
            ss = per[i]
            f0 = min(int(round(ss[0].start * 0.02 * FPS)), T - 1)
            f1 = min(max(f0 + 1, int(round(ss[-1].end * 0.02 * FPS))), T)
            syl.append([a, b, f0, f1, round(float(np.mean([s.score for s in ss])), 3)])

        # frame features
        e = 20 * torch.log10(wav.view(T, HOP).pow(2).mean(1).sqrt() + 1e-5)
        e = (e - e.max()).clamp(min=-60)
        mel = torch.log(self.mel(wav)[:, :T] + 1e-5).T
        mel = mel - mel.mean(0, keepdim=True)
        pitch, per_ = torchcrepe.predict(w16[None], 16000, hop_length=160, fmin=50, fmax=550, model="tiny",
                                         return_periodicity=True, batch_size=4096, device=self.device)
        n = T * 4
        pitch = torch.nn.functional.pad(pitch[0, :n], (0, max(0, n - pitch.shape[1])), value=1.0).view(T, 4)
        per_ = torch.nn.functional.pad(per_[0, :n], (0, max(0, n - per_.shape[1]))).view(T, 4)
        v = per_ > 0.35
        lf = torch.log2(pitch.clamp(min=1.0))
        vf = v.float().mean(1)
        st = torch.zeros(T, device=self.device)
        if v.any():
            med = lf[v].median()
            st = torch.where(v.any(1), 12 * ((lf * v).sum(1) / v.sum(1).clamp(min=1) - med), st)
        feats = torch.cat([st[:, None], vf[:, None], e[:, None], mel], 1)
        return feats.half().cpu().numpy(), syl, wav


def _snapshot(name: str) -> str:
    if os.path.isdir(name):
        return name
    from huggingface_hub import snapshot_download

    return snapshot_download(name)


def write_check(path: str, wav: torch.Tensor, syl: list, text: str):
    """Decoded audio with a short click at every syllable start, for listening to the alignment."""
    import soundfile as sf

    y = wav.cpu().numpy().copy()
    t = np.arange(int(0.012 * SR)) / SR
    click = 0.25 * np.sin(2 * np.pi * 2000 * t) * np.hanning(len(t))
    for s in syl:
        i = s[2] * HOP
        y[i:i + len(click)] += click[: len(y) - i]
    sf.write(path + ".wav", y, SR)
    json.dump({"text": text, "syl": [[text[s[0]:s[1]], s[2], s[3], s[4]] for s in syl]},
              open(path + ".json", "w", encoding="utf-8"), ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--model", required=True, help="OmniVoice checkpoint dir with audio_tokenizer/")
    ap.add_argument("--ctc", required=True, help="letter-level CTC model (HF id or dir) for the language")
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", help="comma-separated; default: every 'vi' dataset of the config")
    ap.add_argument("--limit", type=int, default=0, help="samples per dataset (smoke)")
    ap.add_argument("--check", type=int, default=0, help="write this many click-marked wavs per dataset")
    a = ap.parse_args()
    ex = Extractor(a.model, a.ctc)
    for name, lsts in manifests(a.data_config, set(a.datasets.split(",")) if a.datasets else None).items():
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        done_path = os.path.join(d, "done.txt")
        done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
        fe = open(os.path.join(d, "feats.f16"), "ab")
        fi = open(os.path.join(d, "index.jsonl"), "a", encoding="utf-8")
        rows = os.path.getsize(os.path.join(d, "feats.f16")) // (2 * len(FEATS))
        n = n_ok = n_check = 0
        t0 = time.time()
        for lst in lsts:
            for line in open(lst):
                p = line.split()
                if len(p) < 4 or p[0] in done:
                    continue
                for key, tok, lab in iter_shard(p[0], p[1]):
                    text = lab.get("text", "")
                    out = ex(tok, text)
                    rec = {"id": key, "md5": hashlib.md5(text.encode()).hexdigest()[:12], "off": -1,
                           "T": int(tok.shape[1]), "syl": None}
                    if out[0] is not None:
                        feats, syl, wav = out
                        fe.write(feats.tobytes())
                        rec.update(off=rows, syl=syl)
                        rows += feats.shape[0]
                        n_ok += 1
                        if n_check < a.check:
                            os.makedirs(os.path.join(d, "check"), exist_ok=True)
                            write_check(os.path.join(d, "check", key), wav, syl, text)
                            n_check += 1
                    fi.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
                    if a.limit and n >= a.limit:
                        break
                if a.limit and n >= a.limit:
                    break
                fe.flush()
                fi.flush()
                with open(done_path, "a") as f:
                    f.write(p[0] + "\n")
            if a.limit and n >= a.limit:
                break
        fe.close()
        fi.close()
        dt = time.time() - t0
        print(f"### {name}: {n} samples, {n_ok} aligned, {dt:.0f} s ({n / max(dt, 1e-6):.1f} samples/s)", flush=True)
    json.dump({"feats": FEATS, "fps": FPS, "ctc": a.ctc}, open(os.path.join(a.out, "meta.json"), "w"))


if __name__ == "__main__":
    main()
