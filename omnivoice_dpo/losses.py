"""Preference losses on per-sample scores (higher score = candidate more likely under the model).

``pi_c, pi_r``: policy scores of chosen / rejected; ``ref_c, ref_r``: frozen-reference scores on the same corruptions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def preference_loss(pi_c, pi_r, ref_c, ref_r, beta: float, kind: str = "dpo", lam_nll: float = 0.0):
    """Returns (loss, metrics). ``kind``:

    - ``dpo``: ``-log sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))``; with ``lam_nll > 0`` it is RPO (adds
      ``-lam_nll * pi_c``, an anchor that keeps the chosen candidate likely);
    - ``ipo``: ``((z / beta) - 1 / (2 * beta)) ** 2`` on the same margin;
    - ``rank``: DPO without a reference (``ref_*`` ignored);
    - ``sft``: only ``-pi_c`` (chosen-only fine-tuning baseline).
    """
    if kind == "sft":
        loss = -pi_c.mean()
        z = torch.zeros_like(pi_c)
    else:
        margin = (pi_c - pi_r) if kind == "rank" else (pi_c - ref_c) - (pi_r - ref_r)
        z = beta * margin
        if kind in ("dpo", "rank"):
            loss = -F.logsigmoid(z).mean()
        elif kind == "ipo":
            loss = ((margin - 1 / (2 * beta)) ** 2).mean()
        else:
            raise ValueError(kind)
        if lam_nll:
            loss = loss - lam_nll * pi_c.mean()
    with torch.no_grad():
        metrics = {
            "z_mean": z.mean().item(), "acc": (z > 0).float().mean().item(),
            "pi_chosen": pi_c.mean().item(), "pi_rejected": pi_r.mean().item(),
            "ref_chosen": ref_c.mean().item(), "ref_rejected": ref_r.mean().item(),
        }
    return loss, metrics
