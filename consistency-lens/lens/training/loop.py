"""Single training step composition of loss components."""

from typing import Dict

import torch
from torch import nn

from lens.evaluation.metrics import kl as kl_fn
import logging as _logging
import contextlib

__all__ = ["train_step", "compute_kl_divergence_robust"]
log = _logging.getLogger(__name__)


def compute_kl_divergence_robust(logits_approx: torch.Tensor, logits_orig: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence with numerical stability.
    
    Computes KL(P||Q) where:
    - P = softmax(logits_orig) is the reference distribution
    - Q = softmax(logits_approx) is the distribution being optimized
    
    This measures how much Q diverges from P.
    
    Args:
        logits_approx: Target logits (for Q distribution)
        logits_orig: Original/reference logits (for P distribution)
        
    Returns:
        KL divergence value
    """
    # Compute in float32 for numerical stability
    with torch.amp.autocast('cuda', enabled=False):
        logits_orig_f32 = logits_orig.float()
        logits_approx_f32 = logits_approx.float()
        
        # Numerical stability: subtract max before softmax (detached to avoid gradient issues)
        logits_orig_f32 = logits_orig_f32 - logits_orig_f32.max(dim=-1, keepdim=True)[0].detach() # 0 to just get values
        logits_approx_f32 = logits_approx_f32 - logits_approx_f32.max(dim=-1, keepdim=True)[0].detach()
        
        # Compute KL divergence
        kl_value = kl_fn(  # kl_fn expects log y_pred, y_true
            torch.log_softmax(logits_approx_f32, dim=-1),
            torch.softmax(logits_orig_f32, dim=-1)
        )
    
    return kl_value.to(logits_orig.dtype)

def train_step(  # noqa: D401
    batch: Dict[str, torch.Tensor],
    models: Dict[str, nn.Module],
    _loss_fns: Dict[str, nn.Module] | None = None,  # (e.g. {"T_text": 8, "tau": 1.0, "alpha": 0.1})
    lm_loss_natural_prefix: str | None = None,  # Natural language prefix for base model in LM loss
    tokenizer = None,  # Tokenizer for natural prefix
    cached_prefix_ids: torch.Tensor | None = None,  # Pre-tokenized prefix to avoid re-tokenization
    resample_ablation: bool = True,
) -> Dict[str, torch.Tensor]:
    """Composite loss (MSE + LM + KL + entropy) with flexible weighting.

    All loss components are always computed for monitoring purposes.
    When a component's weight is 0, it's computed with torch.no_grad() to prevent gradient flow.

    Loss components:
        • Reconstruction MSE  (A ↦ Â) - Direct activation reconstruction
        • Language-model KL   (Decoder distribution ↦ Original LLM distribution) 
        • Functional KL       (LLM logits(original) ↦ logits(Â + Δ))
        • Entropy bonus       (Encourage/discourage token diversity)
    """

    dec: nn.Module = models["dec"]
    enc: nn.Module = models["enc"]
    orig = models.get("orig")  # may be None in unit tests

    # ----------------------- auto-encoding path ------------------------------
    A = batch["A"]  # Keep original dtype - autocast will handle conversions
    tau = _loss_fns.get("tau", 1.0) if _loss_fns else 1.0
    T_text = _loss_fns.get("T_text", 8) if _loss_fns else 8

    # Use differentiable generation if configured
    if dec.config.use_flash_attention:
        # Use Flash Attention with KV cache for optimized O(n) generation
        gen = dec.generate_soft_kv_flash(A, max_length=T_text, gumbel_tau=tau)
    elif dec.config.use_kv_cache:
        # Use KV-cached generation for O(n) attention computation
        gen = dec.generate_soft_kv_cached(A, max_length=T_text, gumbel_tau=tau)
    elif dec.config.use_checkpointing:
        gen = dec.generate_soft_chkpt(A, max_length=T_text, gumbel_tau=tau, checkpoint_every_n_tokens=dec.config.checkpoint_every_n_tokens)
    else:
        gen = dec.generate_soft(A, max_length=T_text, gumbel_tau=tau)
    A_hat = enc(gen.generated_text_embeddings)

    # Extract decoded token IDs for logging popularity
    # gen.hard_token_ids is (Batch, T_text)
    # We flatten it to count all tokens produced in the batch
    decoded_token_ids_batch = gen.hard_token_ids.detach().cpu().view(-1)


    alpha = _loss_fns.get("alpha", 0.1) if _loss_fns else 0.1

    kl_base = _loss_fns.get("kl_base_weight", 1.0) if _loss_fns else 1.0
    ent_w = _loss_fns.get("entropy_weight", 0.0) if _loss_fns else 0.0
    mse_w = _loss_fns.get("mse_weight", 0.0) if _loss_fns else 0.0

    
    lm_w = _loss_fns.get("lm_base_weight", 0.0) if _loss_fns else 0.0
    # ----------------------- language-model KL divergence (loss_lm) ------------------------------
    # This loss regularizes the decoder's (D_model) linguistic knowledge with the base model's (LLM_orig_model)
    # linguistic knowledge, aiming to keep explanations on-manifold.
    # It computes KL(P_D || P_Orig) where P_D are predictions from the decoder on its own generated sequence,
    # and P_Orig are predictions from the original LLM on natural language tokens only.
    # Always compute for monitoring, but detach gradients when weight is 0
    grad_context_lm = torch.no_grad() if lm_w == 0 else contextlib.nullcontext()
    if orig is not None:
        with grad_context_lm:
            # Compute LM loss, using no_grad if weight is 0
            # Logits from D_model for its own generation (predicting token t+1 given prefix 0..t from D)
            # gen.raw_lm_logits are (B, T_text, V) - these correspond to generated natural language tokens only
            d_model_pred_logits = gen.raw_lm_logits # Shape is (B, T_text, V) as per decoder.py

            # Create natural language conditioning for base model
            if cached_prefix_ids is not None or (lm_loss_natural_prefix is not None and tokenizer is not None):
                if cached_prefix_ids is not None:
                    # Use cached prefix tokens
                    B = A.shape[0] # Batch size from input A
                    natural_prefix_expanded = cached_prefix_ids.expand(B, -1).to(A.device)  # Shape: (B, prefix_len)
                # else:
                #     # Tokenize the natural language prefix (fallback if not cached)
                #     if tokenizer is None or lm_loss_natural_prefix is None:
                #         # This case should ideally be caught by an earlier config validation
                #         raise ValueError(
                #             "Tokenizer or lm_loss_natural_prefix not available for tokenizing natural prefix, "
                #             "but cached_prefix_ids is also None."
                #         )
                #     natural_prefix_ids = tokenizer(lm_loss_natural_prefix, add_special_tokens=False, return_tensors="pt").input_ids
                #     natural_prefix_ids = natural_prefix_ids.to(A.device)
                #     B = A.shape[0] # Batch size from input A
                #     natural_prefix_expanded = natural_prefix_ids.expand(B, -1)  # Shape: (B, prefix_len)

                # Determine if Gumbel-Softmax outputs from decoder should feed the original LM
                # cfg is available in train_step's scope
                use_gumbel_for_LMorig = dec.config.use_gumbel_for_LMorig

                if use_gumbel_for_LMorig:
                    # Embed the natural prefix using the original model's embeddings
                    prefix_embeds = orig.model.get_input_embeddings()(natural_prefix_expanded) # (B, prefix_len, D_model)
                    
                    # Get the decoder's generated embeddings (output of Gumbel-Softmax STE)
                    # gen.generated_text_embeddings has shape (B, T_text, D_model)
                    decoder_generated_embeds = gen.generated_text_embeddings 

                    # Concatenate embedded prefix with decoder's generated embeddings
                    # Total length of embedded sequence: prefix_len + T_text
                    base_model_input_embeds = torch.cat([prefix_embeds, decoder_generated_embeds], dim=1)

                    # Logits from LLM_orig_model conditioned on embedded prefix + decoder's generated embeddings.
                    # Gradients ARE allowed to flow through decoder_generated_embeds back to the decoder.
                    orig_model_pred_logits_all_pos = orig.model(
                        inputs_embeds=base_model_input_embeds
                    ).logits # Shape: (B, prefix_len + T_text, V)
                    
                    prefix_len_for_slicing = prefix_embeds.shape[1]

                else: # Original behavior: use hard token IDs and no_grad for orig model
                    # Concatenate prefix token IDs with generated hard token IDs
                    # gen.hard_token_ids has shape (B, T_text)
                    base_model_input_ids = torch.cat([natural_prefix_expanded, gen.hard_token_ids], dim=1)  # Shape: (B, prefix_len + T_text)

                    # Logits from LLM_orig_model conditioned on natural language prefix + generated tokens
                    # (predicting token t+1 given natural prefix + generated tokens 0..t)
                    with torch.no_grad(): # Ensure orig model isn't updated by this loss component
                        orig_model_pred_logits_all_pos = orig.model(
                            input_ids=base_model_input_ids
                        ).logits # Shape: (B, prefix_len + T_text, V)
                    
                    prefix_len_for_slicing = natural_prefix_expanded.shape[1]

                start_idx = prefix_len_for_slicing-1 # this predicts the first token of the generated sequence
                
                # T_text here refers to the number of tokens generated by the decoder,
                # which is d_model_pred_logits.shape[1].
                num_generated_tokens = d_model_pred_logits.shape[1]
                end_idx = prefix_len_for_slicing + num_generated_tokens -1 # ending here ensures we predict the last token of the generated sequence
                
                orig_model_pred_logits = orig_model_pred_logits_all_pos[:, start_idx:end_idx, :] # Shape: (B, T_text, V)
            else:
                # This path is taken if no prefix information is available at all.
                raise ValueError(
                    "No natural prefix provided for LM loss calculation: "
                    "cached_prefix_ids is None and either lm_loss_natural_prefix or tokenizer is not set/available."
                )
            # else:
            #     # Fallback to old behavior if no natural prefix provided
            #     # Extract only the natural language tokens (excluding prompts and activation embedding)
            #     # gen.hard_token_ids contains only the generated natural language portion
            #     natural_lang_tokens = gen.hard_token_ids # Shape: (B, T_text)

            #     # Logits from LLM_orig_model conditioned on natural language tokens only
            #     # (predicting token t+1 given prefix 0..t of natural language)
            #     with torch.no_grad(): # Ensure orig model isn't updated by this loss component
            #         orig_model_pred_logits_all_pos = orig.model(
            #             input_ids=natural_lang_tokens
            #         ).logits # Shape: (B, T_text, V)
            #     # We need the logits that predict the same tokens as d_model_pred_logits
            #     orig_model_pred_logits = orig_model_pred_logits_all_pos[:, :-1, :] # Shape: (B, T_text-1, V)

            # log_P_D: Log-distribution from D_model (this is the distribution we are training, q).
            # This will be the `input` to F.kl_div.
            # These are log-probabilities for tokens 1...T_text-1.
            # Compute in float32 with numerical stability
            with torch.amp.autocast('cuda',enabled=False):
                d_logits_f32 = d_model_pred_logits.float()
                d_logits_f32 = d_logits_f32 - d_logits_f32.max(dim=-1, keepdim=True)[0].detach()
                log_P_D_log_probs = torch.nn.functional.log_softmax(d_logits_f32, dim=-1)
                log_P_D_log_probs = log_P_D_log_probs.to(d_model_pred_logits.dtype)

            # P_Orig: Distribution from LLM_orig_model (this is the reference distribution, p).
            # This will be the `target` for F.kl_div.
            # These are probabilities for tokens 1...T_text-1 (since log_target=False for kl_div).
            with torch.amp.autocast('cuda',enabled=False):
                orig_logits_f32 = orig_model_pred_logits.float()
                orig_logits_f32 = orig_logits_f32 - orig_logits_f32.max(dim=-1, keepdim=True)[0].detach()
                P_Orig_probs = torch.nn.functional.softmax(orig_logits_f32, dim=-1)
                P_Orig_probs = P_Orig_probs.to(orig_model_pred_logits.dtype)
        
            # Reshape for kl_div to (N, C) where N = B * (T_text-1), C = V.
            # This ensures 'batchmean' reduction averages over token positions.
            V_lm = d_model_pred_logits.size(-1) # Vocabulary size for language model
            log_P_D_log_probs_flat = log_P_D_log_probs.reshape(-1, V_lm)
            P_Orig_probs_flat = P_Orig_probs.reshape(-1, V_lm)
            
            # Add numerical stability checks
            if torch.isnan(log_P_D_log_probs_flat).any():
                log.warning("NaN detected in decoder log probs before KL computation")
            if torch.isnan(P_Orig_probs_flat).any():
                log.warning("NaN detected in original model probs before KL computation")

            # loss_lm = KL(P_Orig || P_D)
            # F.kl_div(input, target) with log_target=False computes sum(target * (log(target) - input)).
            # Here, input is log_P_D_log_probs_flat (log q: log-probabilities from Decoder), 
            # and target is P_Orig_probs_flat (p: probabilities from Original LLM).
            # So it computes sum(P_Orig * (log P_Orig - log_P_D)), which is KL(P_Orig || P_D).
            # This regularizes the Decoder's distribution (P_D) to match the Original LLM's distribution (P_Orig).
            if lm_w > 0:
                loss_lm = torch.nn.functional.kl_div(
                    input=log_P_D_log_probs_flat,    # log q: log-probabilities of P_D (Decoder model output)
                    target=P_Orig_probs_flat,        # p: probabilities of P_Orig (Original LLM target distribution)
                    reduction='batchmean',           # average KL divergence per token position
                    log_target=False                 # P_Orig_probs_flat contains probabilities, not log-probabilities
                )
            else:
                # Compute without gradients for monitoring
                with torch.no_grad():
                    loss_lm = torch.nn.functional.kl_div(
                        input=log_P_D_log_probs_flat,
                        target=P_Orig_probs_flat,
                        reduction='batchmean',
                        log_target=False
                    ).detach()
    else:
        loss_lm = torch.tensor(0.0, device=A.device)

    # ------------------ entropy (optional regulariser) ---------------------
    # Always compute for monitoring, but detach gradients when weight is 0
    logits = gen.raw_lm_logits  # (B, T, V) from Decoder – still useful for entropy
    if ent_w != 0:  # Can be positive (reward entropy) or negative (penalize entropy)
        # Compute entropy in float32 for numerical stability with bf16
        orig_dtype = logits.dtype
        with torch.amp.autocast('cuda',enabled=False):
            logits_f32 = logits.float()
            # Numerical stability: subtract max before softmax (detached)
            logits_f32 = logits_f32 - logits_f32.max(dim=-1, keepdim=True)[0].detach()
            probs = torch.softmax(logits_f32, dim=-1)
            # Clamp to avoid log(0) - use smaller threshold for numerical stability
            probs = probs.clamp(min=1e-10)
            entropy = (-probs * torch.log(probs)).sum(-1).mean()
        entropy = entropy.to(orig_dtype)
    else:
        # Compute without gradients for monitoring
        with torch.no_grad():
            # Compute entropy in float32 for numerical stability with bf16
            orig_dtype = logits.dtype
            with torch.amp.autocast('cuda',enabled=False):
                logits_f32 = logits.float()
                # Numerical stability: subtract max before softmax (detached)
                logits_f32 = logits_f32 - logits_f32.max(dim=-1, keepdim=True)[0].detach()
                probs = torch.softmax(logits_f32, dim=-1)
                # Clamp to avoid log(0) - use smaller threshold for numerical stability
                probs = probs.clamp(min=1e-10)
                entropy = (-probs * torch.log(probs)).sum(-1).mean()
            entropy = entropy.to(orig_dtype)

    # ----------------------- KL divergence between original and reconstructed model outputs -----------------------------------
    # Always compute for monitoring, but detach gradients when weight is 0
    if orig is not None and all(k in batch for k in ("A_prime", "input_ids_A")):
        # Determine context for gradient computation based on kl_base.
        # If kl_base is 0, ops in this block are for monitoring and shouldn't track gradients.
        # (Requires import contextlib; kl_base is a function argument)
        grad_context = torch.no_grad() if kl_base == 0 else contextlib.nullcontext()

        with grad_context: # This context applies to all subsequent ops in this block
            # Reconstruct A′ as well to get Δ
            Ap = batch["A_prime"]  # Keep original dtype - autocast will handle conversions
            tau = _loss_fns.get("tau", 1.0) if _loss_fns else 1.0
            # Use differentiable generation if configured
            if hasattr(dec.config, 'use_kv_cache') and dec.config.use_kv_cache:
                gen_ap = dec.generate_soft_kv_cached(Ap, max_length=T_text, gumbel_tau=tau)
            elif hasattr(dec.config, 'use_checkpointing') and dec.config.use_checkpointing:
                gen_ap = dec.generate_soft_chkpt(Ap, max_length=T_text, gumbel_tau=tau)
            else:
                gen_ap = dec.generate_soft(Ap, max_length=T_text, gumbel_tau=tau)
            Ap_hat = enc(gen_ap.generated_text_embeddings) # Detached if kl_base == 0 via grad_context

            # Original delta logic is correct under grad_context:
            # If kl_base == 0, Ap_hat is detached. (Ap - Ap_hat).detach() is same as (Ap - Ap_hat).
            # If kl_base > 0, .detach() acts as intended for stop_grad_aprime.
            if enc.config.stop_grad_aprime:
                delta = (Ap - Ap_hat).detach()  # stop-grad on A′ branch
            else:
                delta = (Ap - Ap_hat)  # no stop-grad on A′ branch
            
            if resample_ablation:
                A_train = A_hat + delta # Detached if kl_base == 0 via grad_context
            else:
                A_train = A_hat

            # loss_mse - just for monitoring! We do not train on this, as we need to allow the resample ablation to work
            if mse_w > 0:
                loss_mse = torch.nn.functional.mse_loss(A_train, A)
            else:
                with torch.no_grad():
                    loss_mse = torch.nn.functional.mse_loss(A_train, A)


            input_ids_batch = batch["input_ids_A"]
            layer_idx_batch = batch.get("layer_idx").squeeze()  # (B,)
            token_pos_batch = batch.get("token_pos_A").squeeze()  # (B,)
            B = A.shape[0]

            # Check if we can use vectorized approach (same layer across batch)
            layer_idx = int(layer_idx_batch[0].item()) if layer_idx_batch is not None else 0
            use_vectorized = True
            
            if layer_idx_batch is not None and not torch.all(layer_idx_batch == layer_idx):
                # Mixed layers in batch - fall back to original grouped approach
                use_vectorized = False
                
            if use_vectorized:
                # Vectorized approach: single forward pass with multi-position hook
                # Get logits with original activations (always no_grad for this part)
                with torch.no_grad(): # This inner no_grad is for logits_orig specifically
                    logits_orig = orig.forward_with_replacement_vectorized(
                        input_ids=input_ids_batch,
                        new_activations=A,
                        layer_idx=layer_idx,
                        token_positions=token_pos_batch,
                        no_grad=True,
                    ).logits.detach()
                    # Extract logits at target positions for each sample
                    batch_indices = torch.arange(B, device=A.device)
                    logits_orig = logits_orig[batch_indices, token_pos_batch]  # (B, V)

                # Get logits with reconstructed activations  
                # A_train is detached if kl_base == 0 due to the outer grad_context.
                # no_grad=(kl_base == 0) ensures the forward pass itself respects this.
                logits_approx_full = orig.forward_with_replacement_vectorized(
                    input_ids=input_ids_batch,
                    new_activations=A_train,
                    layer_idx=layer_idx,
                    token_positions=token_pos_batch,
                    no_grad=False,
                ).logits
                # Extract logits at target positions for each sample
                logits_approx = logits_approx_full[batch_indices, token_pos_batch]  # (B, V)
            else:
                raise ValueError("Mixed layers in batch")
                # Fallback: Original grouped approach for mixed-layer batches
                logits_orig_chunks = []
                logits_approx_chunks = []

                unique_pairs = {}
                for i in range(B):
                    l = int(layer_idx_batch[i].item()) if layer_idx_batch is not None else 0
                    p = int(token_pos_batch[i].item()) if token_pos_batch is not None else 0
                    unique_pairs.setdefault((l, p), []).append(i)

                for (l_idx, t_pos), idx_list in unique_pairs.items():
                    ids_subset = input_ids_batch[idx_list]
                    A_subset = A[idx_list]
                    A_train_subset = A_train[idx_list]

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
                        new_activation=A_train_subset,
                        layer_idx=l_idx,
                        token_pos=t_pos,
                        no_grad=False,
                    ).logits[:, t_pos]

                    logits_orig_chunks.append(lo)
                    logits_approx_chunks.append(lt)

                logits_orig = torch.cat(logits_orig_chunks, dim=0)  # (B, V)
                logits_approx = torch.cat(logits_approx_chunks, dim=0)

        # For D_KL(P || Q):
        # P = softmax(logits_orig)
        # Q = softmax(logits_approx)
        # F.kl_div expects input=log_Q, target=P (if log_target=False)
        # Original: kl_fn(softmax(logits_orig), log_softmax(logits_approx))
        #   This leads to log(log_Q) which is log(negative)= NaN
        # For D_KL(P || Q):
        # P = softmax(logits_orig)
        # Q = softmax(logits_approx) 
        # Compute KL loss, using no_grad if weight is 0
        if kl_base > 0:
            # Compute in float32 for numerical stability
            with torch.amp.autocast('cuda',enabled=False):
                logits_orig_f32 = logits_orig.float()
                logits_approx_f32 = logits_approx.float()
                
                # Numerical stability: subtract max before softmax (detached to avoid gradient issues)
                logits_orig_f32 = logits_orig_f32 - logits_orig_f32.max(dim=-1, keepdim=True)[0].detach() # 0 to just get values    
                logits_approx_f32 = logits_approx_f32 - logits_approx_f32.max(dim=-1, keepdim=True)[0].detach()
                
                loss_kl = kl_fn(  # kl_fn expects log y_pred, y_true
                    torch.log_softmax(logits_approx_f32, dim=-1),
                    torch.softmax(logits_orig_f32, dim=-1)
                ) # KL(P||Q) penalises Q for assigning mass where P has none; encourages Q to cover P.
            loss_kl = loss_kl.to(logits_orig.dtype)
        else:
            # Compute without gradients for monitoring
            with torch.no_grad():
                loss_kl = kl_fn(
                    torch.log_softmax(logits_approx, dim=-1),
                    torch.softmax(logits_orig, dim=-1)
                ).detach()
    else:
        # If we can't compute KL (missing data), set to 0 for monitoring
        loss_kl = torch.tensor(0.0, device=A.device)


    # Total loss composition:
    # - KL loss (fundamental objective): fixed weight, measures functional preservation
    # - LM loss (linguistic regularizer): ramped up via alpha schedule for fluency
    # - MSE loss (direct reconstruction): alternative/additional to KL for direct activation matching
    # - Alpha schedule gradually introduces linguistic constraints during training
    total_loss = (lm_w * alpha) * loss_lm + kl_base * loss_kl - ent_w * entropy + mse_w * loss_mse

    return {
        "total": total_loss,
        "mse": loss_mse,  # For logging
        "lm": loss_lm,
        "kl": loss_kl,
        "entropy": entropy,
        "decoded_tokens_batch": decoded_token_ids_batch,  # Add this for epoch-level aggregation
    }
