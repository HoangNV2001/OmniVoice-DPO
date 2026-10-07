# OmniVoice-DPO

Preference optimization for [OmniVoice](https://github.com/k2-fsa/OmniVoice), a masked discrete-diffusion TTS model.
The first target is expressive control: inline non-verbal tags (`[laughs]`, `[coughs]`), fillers and
emotion/style tags.

> **Status: work in progress.** The layout below is the design; most modules are not written yet.

## Why not plain DPO

OmniVoice is not autoregressive, so there is no sequence log-probability to plug into DPO. This package scores a
candidate with the masked-diffusion ELBO and keeps the estimator's variance low, as in VRPO
([LLaDA 1.5](https://arxiv.org/abs/2505.19223)):
- stratified timesteps, one mask per timestep;
- the same timesteps and masks for chosen, rejected, policy and reference.

Candidates for the same prompt share the target length, so pairs always share masks.

## Pipeline

```
prompts.jsonl → ovdpo-generate → candidates (exact token ids + wav) → ovdpo-score → ovdpo-pairs → ovdpo-train → ovdpo-eval
```

Each round samples K candidates per prompt from the current policy. Then:
1. Candidates are ranked inside their group:
   - hard constraints first: content CER, no tag read out as text, speaker similarity;
   - then tag realization;
   - then naturalness.
2. The policy is trained on (best, worst) pairs against a frozen copy of the previous round.

## Install

```bash
pip install torch==2.8.0 torchaudio==2.8.0
pip install -e ".[rewards,dev]"
```

OmniVoice is pinned to commit `08be0b4`, because the sampler and scorer reuse its input-building code.

## Layout

| module | role |
|---|---|
| `omnivoice_dpo/tags.py` | tag inventory (YAML), parsing, tag-free text for ASR |
| `frontend.py` | tag-aware duration estimate; stable tokenization of tags |
| `sampling.py` | iterative unmasking with per-item seeds, token sampling, optional trajectory recording |
| `elbo.py`, `losses.py` | per-sample ELBO scores; DPO / IPO / RPO / chosen-only SFT losses |
| `rewards/` | ASR (CER, leaked tags, fillers), AudioSet event detection, emotion, speaker similarity, LLM-audio judge, ranking |
| `pairs.py`, `trainer.py`, `evaluate.py`, `cli/` | pair building, training loop (LoRA or full), paired-bootstrap evaluation |

## License

Apache-2.0, same as OmniVoice.
