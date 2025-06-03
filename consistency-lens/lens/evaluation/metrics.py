"""KL, cosine similarity, and PPL helpers."""

import torch
from torch import nn

__all__ = ["kl", "cosine"]


kl_loss = nn.KLDivLoss(reduction="batchmean")
cos = nn.CosineSimilarity(dim=-1)


def kl(q_log: torch.Tensor, p: torch.Tensor) -> torch.Tensor:  # noqa: D401
    # this is KL( p || q ), i.e. target = p, input = q
    return kl_loss(q_log, p)


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # noqa: D401
    return cos(a, b).mean()
