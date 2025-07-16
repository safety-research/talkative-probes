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
import psutil

import torch
from torch.utils.data import Dataset

__all__ = ["ActivationDataset"]


class ActivationDataset(Dataset):
    def __init__(
        self,
        root: str,
        max_samples: Optional[int] = None,
        desc: str = "Loading activations",
        preload_to_shared_ram: bool = False,
        use_mmap: bool = False,
    ):
        if preload_to_shared_ram and use_mmap:
            raise ValueError(
                "preload_to_shared_ram and use_mmap are mutually exclusive options."
            )

        self.root = Path(root)
        self.shards: List[Path] = []
        self.lengths: List[int] = []
        self.offsets: List[int] = []
        self.total = 0
        self.data: Optional[List[Dict[str, torch.Tensor]]] = None
        self.use_mmap = use_mmap

        # This will be a per-worker cache if not preloading
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

        if preload_to_shared_ram:
            self._preload_and_share(desc)

    def _preload_and_share(self, desc: str):
        """Loads the entire dataset into a list and moves tensors to shared memory."""
        all_samples = []
        for shard_path in tqdm.tqdm(self.shards, desc=f"{desc} (preloading all shards)"):
            shard_obj = torch.load(shard_path, map_location="cpu")
            if isinstance(shard_obj, list):
                all_samples.extend(shard_obj)
            else:
                all_samples.append(shard_obj)

        # Truncate to max_samples if needed
        if len(all_samples) > self.total:
            all_samples = all_samples[:self.total]

        # Process each sample to ensure correct format and shared memory
        processed_samples = []
        for sample in tqdm.tqdm(all_samples, desc="Processing and moving to shared memory"):
            processed_sample = {}
            for k, v in sample.items():
                if not torch.is_tensor(v):
                    # Convert scalars to tensors
                    if k in ["token_pos", "token_pos_A", "token_pos_A_prime"]:
                        v = torch.tensor([v])
                    else:
                        v = torch.tensor(v)
                
                # Clone the tensor to ensure we own it, then move to shared memory
                v_shared = v.clone()
                v_shared.share_memory_()
                processed_sample[k] = v_shared
            
            processed_samples.append(processed_sample)
        
        self.data = processed_samples

        

    def __len__(self) -> int:
        return self.total

    def _load_shard_private(self, fp: Path):
        """Load shard into a private, per-worker cache."""
        if fp not in self._cache:
            # Use mmap if enabled
            self._cache[fp] = torch.load(fp, map_location="cpu", mmap=self.use_mmap)
        return self._cache[fp]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Fast path: data is preloaded to shared RAM
        if self.data is not None:
            return self.data[idx]

        # Slow path: load on-demand with per-worker cache
        shard_idx = bisect_right(self.offsets, idx) - 1
        local_idx = idx - self.offsets[shard_idx]
        fp = self.shards[shard_idx]

        obj = self._load_shard_private(fp)

        # obj is either a single sample dict or a list of them
        sample = obj[local_idx] if isinstance(obj, list) else obj

        # Ensure all fields are tensors
        for k, v in list(sample.items()):
            if not torch.is_tensor(v):
                # Convert scalars to tensors
                # For token_pos fields, ensure they are at least 1D
                if k in ["token_pos", "token_pos_A", "token_pos_A_prime"]:
                    sample[k] = torch.tensor([v])
                else:
                    sample[k] = torch.tensor(v)
        return sample