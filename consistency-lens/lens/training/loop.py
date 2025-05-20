"""Single training step composition of loss components."""

from typing import Dict

import torch
from torch import nn

from lens.evaluation.metrics import kl as kl_fn

__all__ = ["train_step"]


def train_step(  # noqa: D401
    batch: Dict[str, torch.Tensor],
    models: Dict[str, nn.Module],
    _loss_fns: Dict[str, nn.Module] | None = None,
) -> torch.Tensor:
    """Composite loss (MSE + CE + KL) as per README formula.

    The function still *greatly* simplifies the original plan (single token,
    fixed layer / position) but gives us:

        • Reconstruction MSE  (A ↦ Â)
        • Language-model CE  (soft token distribution ↦ hard id)
        • KL                  (LLM_orig logits(original) ↦ logits(Â + Δ))
    """

    dec: nn.Module = models["dec"]
    enc: nn.Module = models["enc"]
    orig = models.get("orig")  # may be None in unit tests

    # ----------------------- auto-encoding path ------------------------------
    A = batch["A"].float()
    gen = dec.generate_soft(A, max_length=1, gumbel_tau=1.0)
    A_hat = enc(gen.generated_text_embeddings)

    loss_mse = torch.nn.functional.mse_loss(A_hat, A)

    # ----------------------- language-model CE -------------------------------
    logits = gen.raw_lm_logits.squeeze(1)  # (B, V)
    targets = gen.hard_token_ids.squeeze(1)  # (B,)
    loss_ce = torch.nn.functional.cross_entropy(logits, targets)

    # ----------------------- KL (optional) -----------------------------------
    if orig is not None and all(k in batch for k in ("A_prime", "input_ids_A")):
        # Reconstruct A′ as well to get Δ
        Ap = batch["A_prime"].float()
        gen_ap = dec.generate_soft(Ap, max_length=1, gumbel_tau=1.0)
        Ap_hat = enc(gen_ap.generated_text_embeddings)
        delta = Ap - Ap_hat  # (B, d_model)

        A_target = A_hat + delta

        input_ids = batch["input_ids_A"]
        layer_idx = 0
        token_pos = 0

        with torch.no_grad():
            logits_orig = orig.forward_with_replacement(input_ids, A, layer_idx, token_pos).logits[:, token_pos]
        logits_target = orig.forward_with_replacement(input_ids, A_target, layer_idx, token_pos).logits[:, token_pos]

        loss_kl = kl_fn(torch.log_softmax(logits_target, dim=-1), torch.softmax(logits_orig, dim=-1))
    else:
        loss_kl = torch.tensor(0.0, device=A.device)

    return loss_mse + loss_ce + loss_kl
