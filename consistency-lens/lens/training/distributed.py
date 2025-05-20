"""Utilities for distributed training and seed setup."""

import os
import random

import numpy as np
import torch

__all__ = ["set_seed", "is_main"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_main() -> bool:
    return int(os.environ.get("RANK", 0)) == 0
