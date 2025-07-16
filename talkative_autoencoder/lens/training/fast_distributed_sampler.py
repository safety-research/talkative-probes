"""Fast DistributedSampler that avoids expensive .tolist() conversion for large datasets."""

import math
from typing import Iterator, Optional, TypeVar

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler

T_co = TypeVar("T_co", covariant=True)


class FastDistributedSampler(Sampler[T_co]):
    """A faster DistributedSampler that avoids .tolist() conversion for better performance.

    This sampler keeps indices as tensors for as long as possible to avoid the expensive
    tensor -> list conversion that causes slowdowns at epoch boundaries.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        # Calculate samples per replica
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(len(self.dataset))

        if not self.drop_last:
            # Add extra samples to make evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                if padding_size <= len(indices):
                    indices = torch.cat([indices, indices[:padding_size]])
                else:
                    repeat_times = math.ceil(padding_size / len(indices))
                    indices = torch.cat([indices] + [indices] * repeat_times)[: self.total_size]
        else:
            # Remove tail to make evenly divisible
            indices = indices[: self.total_size]

        # Subsample for this rank
        indices = indices[self.rank : self.total_size : self.num_replicas]

        # Only convert to list at the very end when yielding
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler."""
        self.epoch = epoch
