"""Utilities for distributed training and seed setup."""

import os
import random
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

__all__ = [
    "set_seed", 
    "is_main", 
    "get_rank", 
    "get_world_size", 
    "get_local_rank",
    "init_distributed",
    "cleanup_distributed",
    "setup_for_distributed",
    "reduce_dict",
    "all_gather_tensors"
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_main() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get total number of processes."""
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_local_rank() -> int:
    """Get local rank within the node."""
    return int(os.environ.get("LOCAL_RANK", 0))


def init_distributed(backend: str = "nccl") -> Tuple[int, int, int]:
    """Initialize distributed training.
    
    Returns:
        Tuple of (rank, world_size, local_rank)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        # Single process training
        rank = 0
        world_size = 1
        local_rank = 0
        
    if world_size > 1:
        # Initialize the process group
        dist.init_process_group(backend=backend, init_method="env://")
        torch.cuda.set_device(local_rank)
        
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def setup_for_distributed(is_master: bool):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def reduce_dict(input_dict: Dict[str, torch.Tensor], average: bool = True) -> Dict[str, torch.Tensor]:
    """
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
        
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def all_gather_tensors(tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs all_gather operation on the provided tensor.
    """
    world_size = get_world_size()
    if world_size < 2:
        return tensor
        
    tensors_gather = [
        torch.zeros_like(tensor) for _ in range(world_size)
    ]
    dist.all_gather(tensors_gather, tensor)
    output = torch.cat(tensors_gather, dim=0)
    return output
