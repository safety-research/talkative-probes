"""Pads and transfers batch to GPU."""

from typing import List

import torch

__all__ = ["collate"]


def collate(batch: List[dict]) -> dict:  # noqa: D401
    """Placeholder collate that returns dict of stacked tensors."""
    out = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch])
    return out
