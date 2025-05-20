"""Torch Dataset that streams activation triples from mmap shards."""

from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset

__all__ = ["ActivationDataset"]


class ActivationDataset(Dataset):
    """Placeholder dataset."""

    def __init__(self, root: str):
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.pt"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # TODO: memory-map load instead of eager load
        return torch.load(self.files[idx])
