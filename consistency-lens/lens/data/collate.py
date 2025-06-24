"""Pads and transfers batch to GPU."""

from typing import List, Dict, Any, Union
from collections.abc import Mapping

import torch

__all__ = ["collate"]


def _pad_1d(t: torch.Tensor, length: int, pad_val: int) -> torch.Tensor:
    """Right-pad a 1-D tensor to *length* with *pad_val*."""

    if t.size(0) >= length:
        return t[:length]
    pad = t.new_full((length - t.size(0),), pad_val)
    return torch.cat([t, pad], dim=0)


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate a list of samples into a batch.
    Handles tensors and scalar Python numbers (int, float) by converting them to tensors.
    Other types are returned as lists.
    """
    if not batch:
        return {}

    # Use the first element to determine keys and types
    elem = batch[0]
    batch_keys = elem.keys()
    collated_batch: Dict[str, Any] = {key: [] for key in batch_keys}

    # Group values for each key
    for sample in batch:
        for key in batch_keys:
            collated_batch[key].append(sample[key])

    # Process each key based on the type of its first element
    for key in batch_keys:
        example_value = elem[key]
        values_list = collated_batch[key]

        if isinstance(example_value, torch.Tensor):
            # Stack if elements are tensors
            try:
                collated_batch[key] = torch.stack(values_list)
            except Exception as e:
                # Handle cases like list of tensors with different shapes if necessary
                # For now, assume tensors are stackable or raise
                # print(f"Warning: Could not stack tensors for key '{key}'. Returning as list. Error: {e}")
                # collated_batch[key] = values_list # Keep as list of tensors if stacking fails
                raise RuntimeError(f"Failed to stack tensors for key '{key}'. Check tensor shapes. Values: {[v.shape for v in values_list if isinstance(v, torch.Tensor)]}") from e
        elif isinstance(example_value, (int, float)):
            # Convert list of Python numbers (int, float) to a tensor
            if isinstance(example_value, int):
                collated_batch[key] = torch.tensor(values_list, dtype=torch.long)
            else: # float
                collated_batch[key] = torch.tensor(values_list, dtype=torch.float)
        else:
            # For any other types (e.g., list of strings, or complex objects not handled above),
            # they remain as a list of those objects.
            # If `values_list` was already prepared (e.g. list of strings), this is fine.
            pass # `collated_batch[key]` is already the list of values

    return collated_batch
