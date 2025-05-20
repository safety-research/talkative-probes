"""Parameter group helpers for fused AdamW."""

from typing import List

from torch import nn

__all__ = ["param_groups"]


def param_groups(model: nn.Module, lr: float, proj_lr_mult: float = 10.0) -> List[dict]:  # noqa: D401
    """Two-way param grouping.

    * All parameters receive base ``lr``.
    * Parameters whose name contains ``.proj`` get ``lr * proj_lr_mult``.
    * Weight-decay set to 0.01 for matrix weights, 0.0 for biases/LayerNorm.
    """

    groups: List[dict] = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        decay = 0.0 if p.ndim == 1 else 0.01
        lr_scale = proj_lr_mult if ".proj" in n else 1.0
        groups.append({"params": [p], "weight_decay": decay, "lr": lr * lr_scale})
    return groups
