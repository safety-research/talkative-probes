"""Single training step composition of loss components."""

from typing import Dict

import torch
from torch import nn

__all__ = ["train_step"]


def train_step(  # noqa: D401
    batch: Dict[str, torch.Tensor],
    models: Dict[str, nn.Module],
    _loss_fns: Dict[str, nn.Module] | None = None,
) -> torch.Tensor:
    """Compute a simple reconstruction loss.

    The MVP only trains Decoder (``dec``) and Encoder (``enc``) to project an
    activation vector ``A`` ↦ text-embedding ↦ reconstruction ``Â``.
    """

    dec: nn.Module = models["dec"]
    enc: nn.Module = models["enc"]

    A = batch["A"].float()
    gen = dec.generate_soft(A, max_length=1, gumbel_tau=1.0)
    A_hat = enc(gen.generated_text_embeddings)

    return torch.nn.functional.mse_loss(A_hat, A)
