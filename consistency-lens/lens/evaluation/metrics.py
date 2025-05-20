"""KL, cosine similarity, and PPL helpers."""

import torch
from torch import nn

__all__ = ["kl", "cosine"]


kl_loss = nn.KLDivLoss(reduction="batchmean")
cos = nn.CosineSimilarity(dim=-1)


def kl(p_log: torch.Tensor, q: torch.Tensor) -> torch.Tensor:  # noqa: D401
    return kl_loss(p_log, q)


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # noqa: D401
    return cos(a, b).mean()
