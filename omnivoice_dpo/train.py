"""Preference training on ELBO scores (DPO / IPO / RPO / chosen-only SFT), one GPU.

The policy starts from ``--init`` (LoRA on the LLM projections with full audio embeddings/heads, as in upstream LoRA
fine-tuning, or full fine-tuning); the reference is an independent frozen copy of ``--ref`` (the previous round). Both
score each pair on the same corruptions (:mod:`omnivoice_dpo.elbo`).

    python -m omnivoice_dpo.train --records records.pt --init <ckpt> --ref <ckpt> --out out/ [--measure]

``--measure`` only scores the records with the reference and prints the margin distribution, to pick ``--beta`` so
that ``beta * |margin|`` is of order 1.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time

import numpy as np
import torch

from omnivoice_dpo import elbo
from omnivoice_dpo.losses import preference_loss

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load(path, trainable: bool, lora: bool, r: int, alpha: int, dtype):
    from omnivoice import OmniVoice

    m = OmniVoice.from_pretrained(path, train=True, attn_implementation="sdpa", dtype=dtype).cuda()
    if not trainable:
        return m.eval().requires_grad_(False)
    if lora:
        from peft import LoraConfig, get_peft_model

        m = get_peft_model(m, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05, target_modules=LORA_TARGETS,
                                         modules_to_save=["audio_embeddings", "audio_heads"]))
    return m.train()


def corruptions_for(recs, idxs, n_t, t_min, seed):
    out = []
    for i in idxs:
        g = torch.Generator().manual_seed(seed + 7919 * i)
        C, T = recs[i]["chosen"].shape
        out.append(elbo.sample_corruption(C, T, n_t, t_min, g))
    return out


def scores(model, recs, idxs, cors, a, cfg):
    """Scores of chosen and rejected for the pairs ``idxs``; each pair's two targets share one corruption."""
    prefixes, starts, targets, cs = [], [], [], []
    for i, c in zip(idxs, cors):
        for side in ("chosen", "rejected"):
            prefixes.append(recs[i]["prefix"].long())
            starts.append(recs[i]["audio_start"])
            targets.append(recs[i][side].long())
            cs.append(c)
    batch, meta = elbo.build_batch(prefixes, starts, targets, cs, cfg.audio_mask_id, cfg.pad_id)
    batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        s, per_cb = elbo.score(model, batch, meta, cs, len(targets), a.weighting, a.norm)
    return s[0::2], s[1::2], per_cb


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--kind", default="dpo", choices=["dpo", "ipo", "rank", "sft"])
    ap.add_argument("--beta", type=float, default=20.0)
    ap.add_argument("--lam-nll", type=float, default=0.2)
    ap.add_argument("--full", action="store_true", help="full fine-tuning instead of LoRA")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-pairs", type=int, default=4)
    ap.add_argument("--n-t", type=int, default=4)
    ap.add_argument("--t-min", type=float, default=0.1)
    ap.add_argument("--weighting", default="elbo", choices=["elbo", "mean"])
    ap.add_argument("--norm", default="tokens", choices=["tokens", "none"])
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)

    data = torch.load(a.records)
    recs = data["records"]
    order = list(range(len(recs)))
    rng.shuffle(order)
    n_val = int(round(a.val_frac * len(recs))) if not a.measure else 0
    val, tr = order[:n_val], order[n_val:]

    ref = load(a.ref, False, False, 0, 0, torch.bfloat16)
    cfg = ref.config
    cfg.pad_id = cfg.pad_token_id if cfg.pad_token_id is not None else 0
    log = open(os.path.join(a.out, "log.jsonl"), "a")

    @torch.no_grad()
    def ref_margins(idxs, seed):
        out = []
        for b in range(0, len(idxs), a.batch_pairs):
            ii = idxs[b:b + a.batch_pairs]
            c, r, _ = scores(ref, recs, ii, corruptions_for(recs, ii, a.n_t, a.t_min, seed), a, cfg)
            out += (c - r).float().cpu().tolist()
        return np.array(out)

    if a.measure:
        m = ref_margins(order, a.seed)
        q = np.percentile(np.abs(m), [10, 50, 90])
        res = {"n": len(m), "acc_ref": float((m > 0).mean()), "abs_margin_p10_p50_p90": q.tolist(),
               "beta_for_unit_p50": float(1 / max(q[1], 1e-9))}
        print(json.dumps(res))
        json.dump(res, open(os.path.join(a.out, "measure.json"), "w"), indent=1)
        return

    policy = load(a.init, True, not a.full, a.lora_r, a.lora_alpha, torch.float32)
    params = [p for p in policy.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.1f} M", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(tr) / a.batch_pairs)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))

    def evaluate(step):
        if not val:
            return
        policy.eval()
        zs = []
        with torch.no_grad():
            for b in range(0, len(val), a.batch_pairs):
                ii = val[b:b + a.batch_pairs]
                cors = corruptions_for(recs, ii, a.n_t, a.t_min, 10**6)
                pc, pr, _ = scores(policy, recs, ii, cors, a, cfg)
                rc, rr, _ = scores(ref, recs, ii, cors, a, cfg)
                zs += (a.beta * ((pc - rc) - (pr - rr))).float().cpu().tolist()
        policy.train()
        rec = {"step": step, "val_acc": float(np.mean(np.array(zs) > 0)), "val_z": float(np.mean(zs))}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")

    step, t0 = 0, time.time()
    evaluate(0)
    for ep in range(a.epochs):
        rng.shuffle(tr)
        for b in range(0, len(tr), a.batch_pairs):
            ii = tr[b:b + a.batch_pairs]
            cors = corruptions_for(recs, ii, a.n_t, a.t_min, a.seed + 10**5 * (ep + 1))
            pc, pr, per_cb = scores(policy, recs, ii, cors, a, cfg)
            with torch.no_grad():
                rc, rr, _ = scores(ref, recs, ii, cors, a, cfg)
            loss, mt = preference_loss(pc.float(), pr.float(), rc.float(), rr.float(), a.beta, a.kind, a.lam_nll)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            opt.step()
            sched.step()
            step += 1
            rec = {"step": step, "epoch": ep, "loss": loss.item(), "grad_norm": gn, "lr": sched.get_last_lr()[0],
                   "ce_by_codebook": [round(x, 4) for x in per_cb.mean(0).tolist()], **mt}
            log.write(json.dumps(rec) + "\n")
            if step % 10 == 0 or step == 1:
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()
                                  if k != "ce_by_codebook"}), flush=True)
        evaluate(step)
    log.flush()
    print(f"trained {step} steps in {time.time() - t0:.0f} s", flush=True)

    # Save: adapter (if LoRA) and a merged checkpoint loadable by OmniVoice.from_pretrained for the next round.
    merged = os.path.join(a.out, "merged")
    if not a.full:
        policy.save_pretrained(os.path.join(a.out, "adapter"))
        policy = policy.merge_and_unload()
    policy.save_pretrained(merged)
    for f in os.listdir(a.init):
        src = os.path.join(a.init, f)
        if f in ("model.safetensors", "config.json", "MD5", "README.md", "train_config.json") or \
                os.path.exists(os.path.join(merged, f)):
            continue
        (shutil.copytree if os.path.isdir(src) else shutil.copy)(src, os.path.join(merged, f))
    json.dump(vars(a), open(os.path.join(a.out, "train_args.json"), "w"), indent=1)
    print(f"saved -> {merged}")


if __name__ == "__main__":
    main()
