"""Single training step composition of loss components."""

from typing import Dict

import torch
from torch import nn

from lens.evaluation.metrics import kl as kl_fn

__all__ = ["train_step"]


def train_step(  # noqa: D401
    batch: Dict[str, torch.Tensor],
    models: Dict[str, nn.Module],
    _loss_fns: Dict[str, nn.Module] | None = None,  # (e.g. {"T_text": 8, "tau": 1.0, "alpha": 0.1})
) -> Dict[str, torch.Tensor]:
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
    tau = _loss_fns.get("tau", 1.0) if _loss_fns else 1.0
    T_text = _loss_fns.get("T_text", 8) if _loss_fns else 8

    gen = dec.generate_soft(A, max_length=T_text, gumbel_tau=tau)
    A_hat = enc(gen.generated_text_embeddings)

    loss_mse = torch.nn.functional.mse_loss(A_hat, A)

    # ----------------------- language-model CE -------------------------------
    logits = gen.raw_lm_logits  # (B, T, V)
    targets = gen.hard_token_ids  # (B, T)
    B, T, V = logits.shape
    loss_ce = torch.nn.functional.cross_entropy(logits.view(B * T, V), targets.view(B * T))

    # ----------------------- KL (optional) -----------------------------------
    if orig is not None and all(k in batch for k in ("A_prime", "input_ids_A")):
        # Reconstruct A′ as well to get Δ
        Ap = batch["A_prime"].float()
        tau = _loss_fns.get("tau", 1.0) if _loss_fns else 1.0
        gen_ap = dec.generate_soft(Ap, max_length=T_text, gumbel_tau=tau)
        Ap_hat = enc(gen_ap.generated_text_embeddings)
        delta = (Ap - Ap_hat).detach()  # stop-grad on A′ branch

        A_target = A_hat + delta

        input_ids_batch = batch["input_ids_A"]
        layer_idx_batch = batch.get("layer_idx")  # (B,)
        token_pos_batch = batch.get("token_pos")  # (B,)

        logits_orig_chunks = []
        logits_target_chunks = []

        unique_pairs = {}
        for i in range(B):
            l = int(layer_idx_batch[i].item()) if layer_idx_batch is not None else 0
            p = int(token_pos_batch[i].item()) if token_pos_batch is not None else 0
            unique_pairs.setdefault((l, p), []).append(i)

        for (l_idx, t_pos), idx_list in unique_pairs.items():
            ids_subset = input_ids_batch[idx_list]
            A_subset = A[idx_list]
            A_target_subset = A_target[idx_list]

            with torch.no_grad():
                lo = orig.forward_with_replacement(
                    input_ids=ids_subset,
                    new_activation=A_subset,
                    layer_idx=l_idx,
                    token_pos=t_pos,
                ).logits[:, t_pos]

            lt = orig.forward_with_replacement(
                input_ids=ids_subset,
                new_activation=A_target_subset,
                layer_idx=l_idx,
                token_pos=t_pos,
            ).logits[:, t_pos]

            logits_orig_chunks.append(lo)
            logits_target_chunks.append(lt)

        logits_orig = torch.cat(logits_orig_chunks, dim=0)  # (B, V)
        logits_target = torch.cat(logits_target_chunks, dim=0)

        loss_kl = kl_fn(torch.log_softmax(logits_target, dim=-1), torch.softmax(logits_orig, dim=-1))
    else:
        loss_kl = torch.tensor(0.0, device=A.device)

    alpha = _loss_fns["alpha"] if _loss_fns and "alpha" in _loss_fns else 0.1

    # README primary formula: loss_lm + current_alpha * loss_kl
    # loss_lm is equivalent to our loss_ce here.
    total_loss = loss_ce + alpha * loss_kl

    return {
        "total": total_loss,
        "mse": loss_mse,  # For logging
        "ce": loss_ce,
        "kl": loss_kl,
    }
