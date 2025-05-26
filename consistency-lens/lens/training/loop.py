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
    lm_loss_natural_prefix: str | None = None,  # Natural language prefix for base model in LM loss
    tokenizer = None,  # Tokenizer for natural prefix
    cached_prefix_ids: torch.Tensor | None = None  # Pre-tokenized prefix to avoid re-tokenization
) -> Dict[str, torch.Tensor]:
    """Composite loss (MSE + CE + KL) as per README formula.

    The function still *greatly* simplifies the original plan (single token,
    fixed layer / position) but gives us:

        • Reconstruction MSE  (A ↦ Â) #just for monitoring! We do not train on this, as we need to allow for non-semantic shifts
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

    # Extract decoded token IDs for logging popularity
    # gen.hard_token_ids is (Batch, T_text)
    # We flatten it to count all tokens produced in the batch
    decoded_token_ids_batch = gen.hard_token_ids.detach().cpu().view(-1)

    # loss_mse - just for monitoring! We do not train on this, as we need to allow the resample ablation to work
    loss_mse = torch.nn.functional.mse_loss(A_hat, A)

    lm_w = _loss_fns.get("lm_weight", 0.0) if _loss_fns else 0.0
    # ----------------------- language-model KL divergence (loss_lm) ------------------------------
    # This loss regularizes the decoder's (D_model) linguistic knowledge with the base model's (LLM_orig_model)
    # linguistic knowledge, aiming to keep explanations on-manifold.
    # It computes KL(P_D || P_Orig) where P_D are predictions from the decoder on its own generated sequence,
    # and P_Orig are predictions from the original LLM on natural language tokens only.
    if lm_w > 0 and orig is not None:
        if T_text > 1:  # need a next-token target for KL divergence
            # Logits from D_model for its own generation (predicting token t+1 given prefix 0..t from D)
            # gen.raw_lm_logits are (B, T_text, V) - these correspond to generated natural language tokens only
            d_model_pred_logits = gen.raw_lm_logits[:, :-1, :] # Shape: (B, T_text-1, V)

            # Create natural language conditioning for base model
            if cached_prefix_ids is not None or (lm_loss_natural_prefix and tokenizer):
                if cached_prefix_ids is not None:
                    # Use cached prefix tokens
                    B = gen.hard_token_ids.shape[0]
                    natural_prefix_expanded = cached_prefix_ids.expand(B, -1).to(A.device)  # Shape: (B, prefix_len)
                else:
                    # Tokenize the natural language prefix (fallback if not cached)
                    natural_prefix_ids = tokenizer(lm_loss_natural_prefix, add_special_tokens=False, return_tensors="pt").input_ids
                    natural_prefix_ids = natural_prefix_ids.to(A.device)
                    B = gen.hard_token_ids.shape[0]
                    natural_prefix_expanded = natural_prefix_ids.expand(B, -1)  # Shape: (B, prefix_len)
                
                # Concatenate prefix with generated tokens
                base_model_input = torch.cat([natural_prefix_expanded, gen.hard_token_ids], dim=1)  # Shape: (B, prefix_len + T_text)
                
                # Logits from LLM_orig_model conditioned on natural language prefix + generated tokens
                # (predicting token t+1 given natural prefix + generated tokens 0..t)
                with torch.no_grad(): # Ensure orig model isn't updated by this loss component
                    orig_model_pred_logits_all_pos = orig.model(
                        input_ids=base_model_input
                    ).logits # Shape: (B, prefix_len + T_text, V)
                
                # Extract logits corresponding to predictions for the generated portion
                # We want predictions for positions [prefix_len : prefix_len + T_text - 1]
                prefix_len = natural_prefix_expanded.shape[1]
                start_idx = prefix_len
                end_idx = prefix_len + T_text - 1
                orig_model_pred_logits = orig_model_pred_logits_all_pos[:, start_idx:end_idx, :] # Shape: (B, T_text-1, V)
            else:
                # Fallback to old behavior if no natural prefix provided
                # Extract only the natural language tokens (excluding prompts and activation embedding)
                # gen.hard_token_ids contains only the generated natural language portion
                natural_lang_tokens = gen.hard_token_ids # Shape: (B, T_text)
                
                # Logits from LLM_orig_model conditioned on natural language tokens only
                # (predicting token t+1 given prefix 0..t of natural language)
                with torch.no_grad(): # Ensure orig model isn't updated by this loss component
                    orig_model_pred_logits_all_pos = orig.model(
                        input_ids=natural_lang_tokens
                    ).logits # Shape: (B, T_text, V)
                # We need the logits that predict the same tokens as d_model_pred_logits
                orig_model_pred_logits = orig_model_pred_logits_all_pos[:, :-1, :] # Shape: (B, T_text-1, V)

            # log_P_D: Log-distribution from D_model (this is the distribution we are training, q).
            # This will be the `input` to F.kl_div.
            # These are log-probabilities for tokens 1...T_text-1.
            log_P_D_log_probs = torch.nn.functional.log_softmax(d_model_pred_logits, dim=-1)

            # P_Orig: Distribution from LLM_orig_model (this is the reference distribution, p).
            # This will be the `target` for F.kl_div.
            # These are probabilities for tokens 1...T_text-1 (since log_target=False for kl_div).
            P_Orig_probs = torch.nn.functional.softmax(orig_model_pred_logits, dim=-1)
            
            # Reshape for kl_div to (N, C) where N = B * (T_text-1), C = V.
            # This ensures 'batchmean' reduction averages over token positions.
            V_lm = d_model_pred_logits.size(-1) # Vocabulary size for language model
            log_P_D_log_probs_flat = log_P_D_log_probs.reshape(-1, V_lm)
            P_Orig_probs_flat = P_Orig_probs.reshape(-1, V_lm)

            # loss_lm = KL(P_Orig || P_D)
            # F.kl_div(input, target) with log_target=False computes sum(target * (log(target) - input)).
            # Here, input is log_P_D_log_probs_flat (log q: log-probabilities from Decoder), 
            # and target is P_Orig_probs_flat (p: probabilities from Original LLM).
            # So it computes sum(P_Orig * (log P_Orig - log_P_D)), which is KL(P_Orig || P_D).
            # This regularizes the Decoder's distribution (P_D) to match the Original LLM's distribution (P_Orig).
            loss_lm = torch.nn.functional.kl_div(
                input=log_P_D_log_probs_flat,    # log q: log-probabilities of P_D (Decoder model output)
                target=P_Orig_probs_flat,        # p: probabilities of P_Orig (Original LLM target distribution)
                reduction='batchmean',           # average KL divergence per token position
                log_target=False                 # P_Orig_probs_flat contains probabilities, not log-probabilities
            )
        else:
            loss_lm = torch.tensor(0.0, device=A.device)
    else:
        loss_lm = torch.tensor(0.0, device=A.device)

    # ------------------ entropy (optional regulariser) ---------------------
    logits = gen.raw_lm_logits  # (B, T, V) from Decoder – still useful for entropy
    probs = torch.softmax(logits, dim=-1)
    entropy = (-probs * torch.log(probs + 1e-9)).sum(-1).mean()

    # ----------------------- KL (optional) -----------------------------------
    if orig is not None and all(k in batch for k in ("A_prime", "input_ids_A")):
        # Reconstruct A′ as well to get Δ
        Ap = batch["A_prime"].float()
        tau = _loss_fns.get("tau", 1.0) if _loss_fns else 1.0
        gen_ap = dec.generate_soft(Ap, max_length=T_text, gumbel_tau=tau)
        Ap_hat = enc(gen_ap.generated_text_embeddings)
        if enc.config.stop_grad_aprime:
            delta = (Ap - Ap_hat).detach()  # stop-grad on A′ branch
        else:
            delta = (Ap - Ap_hat)  # no stop-grad on A′ branch

        A_target = A_hat + delta

        input_ids_batch = batch["input_ids_A"]
        layer_idx_batch = batch.get("layer_idx")  # (B,)
        token_pos_batch = batch.get("token_pos")  # (B,)
        B = A.shape[0]

        # Check if we can use vectorized approach (same layer across batch)
        layer_idx = int(layer_idx_batch[0].item()) if layer_idx_batch is not None else 0
        use_vectorized = True
        
        if layer_idx_batch is not None and not torch.all(layer_idx_batch == layer_idx):
            # Mixed layers in batch - fall back to original grouped approach
            use_vectorized = False
            
        if use_vectorized:
            # Vectorized approach: single forward pass with multi-position hook
            # Get logits with original activations
            with torch.no_grad():
                logits_orig = orig.forward_with_replacement_vectorized(
                    input_ids=input_ids_batch,
                    new_activations=A,
                    layer_idx=layer_idx,
                    token_positions=token_pos_batch,
                    no_grad=True,
                ).logits
                # Extract logits at target positions for each sample
                batch_indices = torch.arange(B, device=A.device)
                logits_orig = logits_orig[batch_indices, token_pos_batch]  # (B, V)

            # Get logits with reconstructed activations  
            logits_target_full = orig.forward_with_replacement_vectorized(
                input_ids=input_ids_batch,
                new_activations=A_target,
                layer_idx=layer_idx,
                token_positions=token_pos_batch,
                no_grad=False,
            ).logits
            # Extract logits at target positions for each sample
            logits_target = logits_target_full[batch_indices, token_pos_batch]  # (B, V)
        else:
            # Fallback: Original grouped approach for mixed-layer batches
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
                        no_grad=True,
                    ).logits[:, t_pos]

                lt = orig.forward_with_replacement(
                    input_ids=ids_subset,
                    new_activation=A_target_subset,
                    layer_idx=l_idx,
                    token_pos=t_pos,
                    no_grad=False,
                ).logits[:, t_pos]

                logits_orig_chunks.append(lo)
                logits_target_chunks.append(lt)

            logits_orig = torch.cat(logits_orig_chunks, dim=0)  # (B, V)
            logits_target = torch.cat(logits_target_chunks, dim=0)

        # For D_KL(P || Q):
        # P = softmax(logits_orig)
        # Q = softmax(logits_target)
        # F.kl_div expects input=log_Q, target=P (if log_target=False)
        # Original: kl_fn(softmax(logits_orig), log_softmax(logits_target))
        #   This leads to log(log_Q) which is log(negative) = NaN
        # Corrected: kl_fn(log_softmax(logits_target), softmax(logits_orig))
        loss_kl = kl_fn(#kl_fn expect log y_pred, y_true - this is what we ahve here
            torch.log_softmax(logits_target, dim=-1),
            torch.softmax(logits_orig, dim=-1)
        ) # KL(P||Q) penalises Q for assigning mass where P has none; encourages Q to cover P.
    else:
        raise ValueError("No KL loss")

    alpha = _loss_fns.get("alpha", 0.1) if _loss_fns else 0.1

    kl_base = _loss_fns.get("kl_base_weight", 1.0) if _loss_fns else 1.0
    ent_w = _loss_fns.get("entropy_weight", 0.0) if _loss_fns else 0.0

    # Total loss composition:
    # - KL loss (fundamental objective): fixed weight, measures functional preservation
    # - LM loss (linguistic regularizer): ramped up via alpha schedule for fluency
    # - Alpha schedule gradually introduces linguistic constraints during training
    total_loss = (lm_w * alpha) * loss_lm + kl_base * loss_kl - ent_w * entropy #+ loss_mse

    return {
        "total": total_loss,
        "mse": loss_mse,  # For logging
        "lm": loss_lm,
        "kl": loss_kl,
        "entropy": entropy,
        "decoded_tokens_batch": decoded_token_ids_batch,  # Add this for epoch-level aggregation
    }
