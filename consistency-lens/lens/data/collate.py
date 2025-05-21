"""Pads and transfers batch to GPU."""

from typing import List

import torch

__all__ = ["collate"]


def _pad_1d(t: torch.Tensor, length: int, pad_val: int) -> torch.Tensor:
    """Right-pad a 1-D tensor to *length* with *pad_val*."""

    if t.size(0) >= length:
        return t[:length]
    pad = t.new_full((length - t.size(0),), pad_val)
    return torch.cat([t, pad], dim=0)


def collate(batch: List[dict]) -> dict:  # noqa: D401
    """Pad variable-length sequences and stack the batch.

    Keys ending with ``_ids`` are assumed to be 1-D token sequences that need
    padding.  Everything else is stacked verbatim.
    """

    out: dict[str, torch.Tensor] = {}
    keys = batch[0].keys()

    for k in keys:
        values = [b[k] for b in batch]

        # if k.endswith("_ids") or k.endswith("token_pos"):
        #     pad_id = int(values[0].new_tensor(0).item())  # assume 0 is pad
        #     max_len = max(v.size(0) for v in values)
        #     padded = [_pad_1d(v, max_len, pad_id) for v in values]
        #     out[k] = torch.stack(padded)
        # else:
        out[k] = torch.stack(values)

    return out
