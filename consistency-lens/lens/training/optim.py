"""Parameter group helpers for fused AdamW."""

from typing import List

from torch import nn

__all__ = ["param_groups"]


def param_groups(model: nn.Module) -> List[dict]:  # noqa: D401
    """Return flat param groups with proj layers having higher lr placeholder."""
    decays, no_decays = [], []
    for n, p in model.named_parameters():
        (no_decays if p.ndim == 1 else decays).append(p)
    return [
        {"params": decays, "weight_decay": 0.01},
        {"params": no_decays, "weight_decay": 0.0},
    ]
