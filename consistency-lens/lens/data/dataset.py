"""Torch Dataset that streams activation triples.

Supports two on-disk layouts:
1.  Legacy – one ``*.pt`` file per sample.
2.  Sharded – one ``*.pt`` file contains a list of sample dicts (written by
   the updated activation-dumper).
"""

from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Optional
import tqdm

import torch
from torch.utils.data import Dataset

__all__ = ["ActivationDataset"]


class ActivationDataset(Dataset):
    def __init__(self, root: str, max_samples: Optional[int] = None, desc: str = "Loading activations"):
        self.root = Path(root)

        # Collect all .pt files (either shards or single-sample files)
        self.shards: List[Path] = sorted(self.root.glob("*.pt")) if self.root.is_dir() else [self.root]

        self.lengths: List[int] = []
        self.offsets: List[int] = []

        total = 0
        for fp in tqdm.tqdm(self.shards, desc=desc):
            obj = torch.load(fp, map_location="cpu")
            shard_len = len(obj) if isinstance(obj, list) else 1
            self.lengths.append(shard_len)
            self.offsets.append(total)
            total += shard_len
            
            # Stop loading if we've reached max_samples
            if max_samples is not None and total >= max_samples:
                # Adjust the last shard's length if needed
                if total > max_samples:
                    excess = total - max_samples
                    self.lengths[-1] -= excess
                    total = max_samples
                break

        self.total = total
        self._cache: dict[Path, List[Dict[str, torch.Tensor]]] = {}  # only used for list-shards

    def __len__(self) -> int:
        return self.total

    def _load_shard(self, fp: Path):
        if fp not in self._cache:
            self._cache[fp] = torch.load(fp, map_location="cpu")
        return self._cache[fp]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Locate the shard containing idx
        shard_idx = bisect_right(self.offsets, idx) - 1
        local_idx = idx - self.offsets[shard_idx]
        fp = self.shards[shard_idx]

        obj = self._load_shard(fp)

        # obj is either a single sample dict or a list of them
        sample = obj[local_idx] if isinstance(obj, list) else obj

        # Ensure all fields are tensors
        for k, v in list(sample.items()):
            if not torch.is_tensor(v):
                sample[k] = torch.tensor(v)
        return sample