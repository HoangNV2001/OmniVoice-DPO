"""Fine-tune OmniVoice with syllable prosody embeddings, on the upstream trainer (accelerate, packing).

Same configs as upstream training (``--train_config``, ``--data_config``); the model class is swapped for
:class:`OmniVoiceProsody` and the processor / collators are wrapped to add prosody from ``--store`` (output of
:mod:`omnivoice_dpo.prosody.prep`). The encoder gets its own learning rate. When ``--init`` has no prosody weights the
encoder starts with a zero output layer, so step 0 reproduces the initial model.

    accelerate launch -m omnivoice_dpo.prosody.train --train_config train.json --data_config data.json \\
        --output_dir out/ --store store/ [--init <ckpt>] [--enc_lr 3e-4] [--p_utt 0.3] [--p_syl 0.1]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os

import torch

import omnivoice.training.builder as B
from omnivoice.data.collator import PackingDataCollator, PaddingDataCollator
from omnivoice.data.processor import OmniVoiceSampleProcessor
from omnivoice.training.config import TrainingConfig
from omnivoice.training.trainer import OmniTrainer
from transformers import get_cosine_schedule_with_warmup

from omnivoice_dpo.prosody.data import ProsodyCollator, ProsodySampleProcessor, ProsodyStore
from omnivoice_dpo.prosody.model import OmniVoiceProsody

logger = logging.getLogger(__name__)


def has_prosody_weights(path: str) -> bool:
    f = os.path.join(path, "model.safetensors")
    if not os.path.isfile(f):
        return False
    from safetensors import safe_open

    with safe_open(f, "pt") as st:
        return any(k.startswith("prosody_encoder.") for k in st.keys())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train_config", required=True)
    ap.add_argument("--data_config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--init", help="overrides init_from_checkpoint")
    ap.add_argument("--enc_lr", type=float, default=3e-4)
    ap.add_argument("--p_utt", type=float, default=0.3)
    ap.add_argument("--p_syl", type=float, default=0.1)
    ap.add_argument("--min_conf", type=float, default=0.3)
    a = ap.parse_args()

    config = TrainingConfig.from_json(a.train_config)
    config.output_dir, config.data_config = a.output_dir, a.data_config
    if a.init:
        config.init_from_checkpoint = a.init
    store = ProsodyStore(a.store)

    B.OmniVoice = OmniVoiceProsody
    B.OmniVoiceSampleProcessor = lambda **kw: ProsodySampleProcessor(OmniVoiceSampleProcessor(**kw), store, a.p_utt,
                                                                     a.p_syl, a.min_conf)
    B.PackingDataCollator = lambda proc, bt: ProsodyCollator(PackingDataCollator(proc, bt))
    B.PaddingDataCollator = lambda proc, bt: ProsodyCollator(PaddingDataCollator(proc, bt))

    model, tokenizer = B.build_model_and_tokenizer(config)
    fresh = not (config.init_from_checkpoint and has_prosody_weights(config.init_from_checkpoint))
    if fresh:
        model.prosody_encoder.reset_parameters()
    train_loader, eval_loader = B.build_dataloaders(config, tokenizer)

    enc = [p for n, p in model.named_parameters() if n.startswith("prosody_encoder.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("prosody_encoder.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": rest, "lr": config.learning_rate}, {"params": enc, "lr": a.enc_lr}],
                            weight_decay=config.weight_decay)
    warm = math.ceil(config.steps * config.warmup_ratio) if config.warmup_type == "ratio" else config.warmup_steps
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warm, num_training_steps=config.steps)

    trainer = OmniTrainer(model=model, config=config, train_dataloader=train_loader, eval_dataloader=eval_loader,
                          tokenizer=tokenizer, optimizer=opt, lr_scheduler=sched)
    if trainer.accelerator.is_main_process:
        os.makedirs(a.output_dir, exist_ok=True)
        json.dump({**vars(a), "store_samples": len(store), "encoder_fresh": fresh,
                   "encoder_params": sum(p.numel() for p in enc)},
                  open(os.path.join(a.output_dir, "prosody_args.json"), "w"), indent=1)
        logger.info(f"prosody store: {len(store)} samples; encoder {'fresh' if fresh else 'loaded'}, "
                    f"{sum(p.numel() for p in enc) / 1e6:.2f} M params")
    trainer.train()


if __name__ == "__main__":
    main()
