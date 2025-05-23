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
import json

import torch
from torch.utils.data import Dataset

__all__ = ["ActivationDataset"]


class ActivationDataset(Dataset):
    def __init__(self, root: str, max_samples: Optional[int] = None, desc: str = "Loading activations"):
        self.root = Path(root)
        self.shards: List[Path] = []
        self.lengths: List[int] = []
        self.offsets: List[int] = []
        self.total = 0
        self._cache: dict[Path, List[Dict[str, torch.Tensor]]] = {}

        metadata_path = self.root / "metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            current_total = 0
            for shard_info in tqdm.tqdm(metadata["shards"], desc=f"{desc} (from metadata)"):
                if max_samples is not None and current_total >= max_samples:
                    break

                # Handle rank subdirectories if present
                if "rank" in shard_info:
                    shard_path = self.root / f"rank_{shard_info['rank']}" / shard_info["name"]
                else:
                    shard_path = self.root / shard_info["name"]
                self.shards.append(shard_path)
                
                shard_len = shard_info["num_samples"]
                if max_samples is not None and current_total + shard_len > max_samples:
                    shard_len = max_samples - current_total
                
                self.lengths.append(shard_len)
                self.offsets.append(current_total)
                current_total += shard_len
            
            self.total = current_total
        else:
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
            # Fallback to original loading method if metadata.json doesn't exist
            potential_shards = sorted(self.root.glob("*.pt")) if self.root.is_dir() else [self.root]
            
            current_total = 0
            for fp in tqdm.tqdm(potential_shards, desc=f"{desc} (legacy scan)"):
                if max_samples is not None and current_total >= max_samples:
                    break

                # This is the slow part we are trying to avoid with metadata
                obj = torch.load(fp, map_location="cpu")
                shard_len = len(obj) if isinstance(obj, list) else 1
                
                actual_shard_len = shard_len
                if max_samples is not None and current_total + shard_len > max_samples:
                    actual_shard_len = max_samples - current_total
                
                if actual_shard_len <= 0 : # No more samples needed or shard is effectively empty for our needs
                    continue

                self.shards.append(fp)
                self.lengths.append(actual_shard_len)
                self.offsets.append(current_total)
                current_total += actual_shard_len
            
            self.total = current_total

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