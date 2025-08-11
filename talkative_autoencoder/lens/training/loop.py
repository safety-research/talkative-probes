"""Single training step composition of loss components."""

from typing import Dict, Optional

import torch
from torch import nn

from lens.evaluation.metrics import kl as kl_fn
import logging as _logging
import contextlib

__all__ = ["train_step", "compute_kl_divergence_robust"]
log = _logging.getLogger(__name__)


def compute_entropy_robust(logits):
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
    return entropy

def compute_advantages(A,A_train, group_n, no_advantage=False):
    # minus sign, crucial
    mse_reward = -(A-A_train).pow(2).mean(dim=-1)#only mean over features, still per-batch
    if no_advantage:
        mse_reward_std = torch.zeros_like(mse_reward)
        return mse_reward_std, mse_reward.mean()
    reshaped_mse_reward = mse_reward.reshape(-1,group_n )
    means = reshaped_mse_reward.mean(dim=-1,keepdim=True)
    stds = reshaped_mse_reward.std(dim=-1,keepdim=True)
    
    # Debug: Check for zero or near-zero std
    if (stds < 1e-8).any():
        log.warning(f"Very small std detected in advantages: min std = {stds.min().item():.2e}, max std = {stds.max().item():.2e}, avg std = {stds.mean().item():.2e}")
        # Add small epsilon to prevent division by zero
        stds = stds.clamp(min=1e-8)
    
    advantages = ((reshaped_mse_reward-means)/stds).reshape(-1)
    mean_reward = means.mean()
    mean_reward_std = stds.mean()
    return advantages, mean_reward_std, mean_reward

def KL_schulman_estimator(probs_of_interest,orig_model_logprobs_of_interest):
    # Debug: Check inputs
    if (probs_of_interest < 1e-10).any():
        log.warning(f"Very small probs_of_interest detected: min = {probs_of_interest.min().item():.2e}")
    
    ratio_r =  torch.exp(orig_model_logprobs_of_interest)/probs_of_interest
    KL_reverse = ratio_r - (orig_model_logprobs_of_interest-torch.log(probs_of_interest))-1

    return KL_reverse.mean()

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
    _loss_fns: Dict[str, nn.Module] | None = None,  # (e.g. {"t_text": 8, "tau": 1.0, "alpha": 0.1})
    lm_loss_natural_prefix: str | None = None,  # Natural language prefix for base model in LM loss
    tokenizer = None,  # Tokenizer for natural prefix
    cached_prefix_ids: torch.Tensor | None = None,  # Pre-tokenized prefix to avoid re-tokenization
    resample_ablation: bool = True,
    should_run_interventions: bool = False,  # Whether we're in evaluation mode
    verbose_eval: bool = False,  # Whether to collect verbose evaluation data
    do_kl_computation: bool = True,
    do_lm_computation: bool = True,
    GRPO_validate_mode: bool = False,
    debug_mode: bool = False,
    return_reconstruction: bool = False,
    mean_n_sequences = False
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


    alpha = _loss_fns.get("alpha", 0.1) if _loss_fns else 0.1
    entropy_clamp = _loss_fns.get("entropy_clamp", 1000) if _loss_fns else 1000

    kl_base = _loss_fns.get("kl_base_weight", 1.0) if _loss_fns else 1.0
    ent_w = _loss_fns.get("entropy_weight", 0.0) if _loss_fns else 0.0
    mse_w = _loss_fns.get("mse_weight", 0.0) if _loss_fns else 0.0
    lm_w = _loss_fns.get("lm_base_weight", 0.0) if _loss_fns else 0.0
    GRPO_w = _loss_fns.get("GRPO_weight", 0.0) if _loss_fns else 0.0
    skip_tokens_KL_GRPO = _loss_fns.get("skip_tokens_KL_GRPO", 0) if _loss_fns else 0
    GRPO_beta = _loss_fns.get("GRPO_beta", 0.0) if _loss_fns else 0.0
    GRPO_training = GRPO_w !=0
    if GRPO_training:
        group_n = _loss_fns['group_n']

    do_lm_computation = do_lm_computation and not (mean_n_sequences and mean_n_sequences>1)
    do_kl_computation = do_kl_computation and not (mean_n_sequences and mean_n_sequences>1)


    dec: nn.Module = models["dec"]
    enc: nn.Module = models["enc"]
    orig = models.get("orig")  # may be None in unit tests

    # ----------------------- auto-encoding path ------------------------------
    A = batch["A"]  # Keep original dtype - autocast will handle conversions
    # Get original token position for A, to be used if subtracting/adding positional embeddings
    original_token_pos_A = batch.get("token_pos_A")

    tau = _loss_fns.get("tau", 1.0) if _loss_fns else 1.0
    t_text = _loss_fns.get("t_text", 8) if _loss_fns else 8
    
    if debug_mode:
        # Debug: Check for NaN in input activation
        if torch.isnan(A).any():
            log.error(f"NaN detected in activation A before generation. Shape: {A.shape}")
    
        # Debug: Check decoder parameters for NaN
        if hasattr(dec, 'check_for_nans') and dec.check_for_nans():
            log.error("NaN detected in decoder parameters before generation")

    # Ensure original_token_pos_A is 2D for gather (B, 1)
    if original_token_pos_A.dim() == 1:
        gather_index = original_token_pos_A.unsqueeze(1)
    else:
        gather_index = original_token_pos_A
    current_token_ids = batch["input_ids_A"].gather(1, gather_index).squeeze(1)

    if mean_n_sequences and mean_n_sequences>1:
        A_orig = A.clone()
        A_repeat = A.repeat_interleave(mean_n_sequences, dim=0)
        A = A_repeat
        original_token_pos_A = original_token_pos_A.repeat_interleave(mean_n_sequences, dim=0)
        current_token_ids = current_token_ids.repeat_interleave(mean_n_sequences, dim=0)
    
    # Use differentiable generation if configured
    if GRPO_w>0: ##kl_base==0 and ent_w==0 and mse_w==0 and lm_w==0
        context = torch.no_grad()
    else:
        context = contextlib.nullcontext()
    with context:
        if dec.config.use_flash_attention:
            # Use Flash Attention with KV cache for optimized O(n) generation
            gen = dec.generate_soft_kv_flash(A, max_length=t_text, gumbel_tau=tau, original_token_pos=original_token_pos_A)
        elif dec.config.use_kv_cache:
            # Use KV-cached generation for O(n) attention computation
            if GRPO_training:# and not GRPO_validate_mode:
                gen = dec.generate_soft_kv_cached_nondiff(A, max_length=t_text, gumbel_tau=tau, original_token_pos=original_token_pos_A, return_logits=GRPO_validate_mode)
            else:
                gen = dec.generate_soft_kv_cached(A, max_length=t_text, gumbel_tau=tau, original_token_pos=original_token_pos_A)
        elif dec.config.use_checkpointing:
            gen = dec.generate_soft_chkpt(A, max_length=t_text, gumbel_tau=tau, checkpoint_every_n_tokens=dec.config.checkpoint_every_n_tokens, original_token_pos=original_token_pos_A)
        else:
            gen = dec.generate_soft(A, max_length=t_text, gumbel_tau=tau, original_token_pos=original_token_pos_A)


    A_hat = enc(gen.generated_text_embeddings, original_token_pos=original_token_pos_A, current_token_ids=current_token_ids if enc.config.add_current_token else None)

    if mean_n_sequences and mean_n_sequences>1:
        A_hat = A_hat.reshape(-1, mean_n_sequences, A_hat.shape[-1]).mean(dim=1)
        A = A_orig
    

    # Extract decoded token IDs for logging popularity
    # gen.hard_token_ids is (Batch, t_text)
    # We flatten it to count all tokens produced in the batch
    decoded_token_ids_batch = gen.hard_token_ids.detach().cpu().view(-1)
    
    # If in eval mode, run interventions (but continue with normal loss computation)
    intervention_results = {}
    if should_run_interventions:
        from lens.training.eval_interventions import run_eval_interventions
        
        # Run the intervention analysis
        with torch.no_grad():
            intervention_results = run_eval_interventions(
                generated_embeddings=gen.generated_text_embeddings,
                enc=enc,
                orig_A=A,
                A_hat_decoder=A_hat,  # Pass the already computed decoder reconstruction
                batch=batch,
                orig_model=orig,
                verbose=verbose_eval,
                number_of_examples=8,
                decoder_prompt=dec.prompt_text,
                dec=dec,
                tok=tokenizer,
                sch_args=_loss_fns,
            )
        
            # Store verbose data separately if present
            if "verbose_data" in intervention_results:
                verbose_data = intervention_results.pop("verbose_data")
                intervention_results["verbose_data"] = verbose_data

    

    # ----------------------- language-model KL divergence (loss_lm) ------------------------------
    # This loss regularizes the decoder's (D_model) linguistic knowledge with the base model's (LLM_orig_model)
    # linguistic knowledge, aiming to keep explanations on-manifold.
    # It computes KL(P_D || P_Orig) where P_D are predictions from the decoder on its own generated sequence,
    # and P_Orig are predictions from the original LLM on natural language tokens only.
    # Always compute for monitoring, but detach gradients when weight is 0
    grad_context_lm = torch.no_grad() if lm_w == 0 and (not GRPO_training) else contextlib.nullcontext()
    orig_model_pred_logits_all_pos=None
    if do_lm_computation or lm_w > 0:
        with grad_context_lm:
            # Compute LM loss, using no_grad if weight is 0
            d_model_pred_logits = gen.raw_lm_logits # Shape is (B, t_text, V) as per decoder.py
            if d_model_pred_logits is None:
                log.error("d_model_pred_logits is None")
                raise ValueError(f"d_model_pred_logits is None: {GRPO_validate_mode}, {GRPO_training}, {lm_w}, {do_lm_computation}, {resample_ablation}, {kl_base}, {mse_w}, {ent_w}, {GRPO_w}, {GRPO_training}, {group_n}, {t_text}, {tau}, {alpha}")
            B = A.shape[0] # Batch size from input A
            natural_prefix_expanded = cached_prefix_ids.expand(B, -1).to(A.device)  # Shape: (B, prefix_len)

            # Determine if Gumbel-Softmax outputs from decoder should feed the original LM
            # cfg is available in train_step's scope
            use_gumbel_for_LMorig = dec.config.use_gumbel_for_LMorig

            if use_gumbel_for_LMorig:
                # Embed the natural prefix using the original model's embeddings
                prefix_embeds = orig.model.get_input_embeddings()(natural_prefix_expanded) # (B, prefix_len, D_model)
                # Get the decoder's generated embeddings (output of Gumbel-Softmax STE)
                decoder_generated_embeds = gen.generated_text_embeddings #(B, t_text, D_model)

                # Concatenate embedded prefix with decoder's generated embeddings
                base_model_input_embeds = torch.cat([prefix_embeds, decoder_generated_embeds], dim=1)# length of embedded sequence: prefix_len + t_text
                # Logits from LLM_orig_model conditioned on embedded prefix + decoder's generated embeddings.
                # Gradients ARE allowed to flow through decoder_generated_embeds back to the decoder.
                orig_model_pred_logits_all_pos = orig.model(
                    inputs_embeds=base_model_input_embeds
                ).logits # Shape: (B, prefix_len + t_text, V)
                
                prefix_len_for_slicing = prefix_embeds.shape[1]

            else: # Original behavior: use hard token IDs and no_grad for orig model
                base_model_input_ids = torch.cat([natural_prefix_expanded, gen.hard_token_ids], dim=1)  # Shape: (B, prefix_len + t_text)

                # Logits from LLM_orig_model conditioned on natural language prefix + generated tokens
                # (predicting token t+1 given natural prefix + generated tokens 0..t)
                with torch.no_grad(): # Ensure orig model isn't updated by this loss component
                    orig_model_pred_logits_all_pos = orig.model(
                        input_ids=base_model_input_ids
                    ).logits # Shape: (B, prefix_len + t_text, V)
                
                prefix_len_for_slicing = natural_prefix_expanded.shape[1]

            start_idx = prefix_len_for_slicing-1 # this predicts the first token of the generated sequence
            
            num_generated_tokens = d_model_pred_logits.shape[1]
            end_idx = prefix_len_for_slicing + num_generated_tokens -1 # ending here ensures we predict the last token of the generated sequence
            
            orig_model_pred_logits = orig_model_pred_logits_all_pos[:, start_idx:end_idx, :].clone() # Shape: (B, t_text, V)
            del orig_model_pred_logits_all_pos
         

            # Compute LM loss KL(P_Orig || P_D).
            # P_Orig is the reference distribution from the original LLM (p).
            # P_D is the distribution from the decoder we are training (q).
            
            # Reshape logits to (N, C) for KL divergence, where N = B * t_text,
            # to average KL over all token positions.
            V_lm = d_model_pred_logits.size(-1)
            d_model_pred_logits_flat = d_model_pred_logits.reshape(-1, V_lm)
            orig_model_pred_logits_flat = orig_model_pred_logits.reshape(-1, V_lm)

            # Add numerical stability checks on the raw logits
            if torch.isnan(d_model_pred_logits_flat).any():
                log.warning("NaN detected in decoder logits before LM KL computation")
            if torch.isnan(orig_model_pred_logits_flat).any():
                log.warning("NaN detected in original model logits before LM KL computation")

            if lm_w > 0:
                loss_lm = compute_kl_divergence_robust(
                    logits_approx=d_model_pred_logits_flat,  # q
                    logits_orig=orig_model_pred_logits_flat    # p
                )
            else:
                # Compute without gradients for monitoring
                with torch.no_grad():
                    loss_lm = compute_kl_divergence_robust(
                        logits_approx=d_model_pred_logits_flat,
                        logits_orig=orig_model_pred_logits_flat
                    ).detach()
    else:
        loss_lm = torch.tensor(0.0, device=A.device, dtype=A.dtype)
        
    if GRPO_beta!=0:
        B=A.shape[0]
        natural_prefix_expanded = cached_prefix_ids.expand(B, -1).to(A.device)  # Shape: (B, prefix_len)
        prefix_len_for_slicing = natural_prefix_expanded.shape[-1]
        base_model_input_ids = torch.cat([natural_prefix_expanded, gen.hard_token_ids], dim=1)  # Shape: (B, prefix_len + t_text)
        slice_index = prefix_len_for_slicing-1
        # Logits from LLM_orig_model conditioned on natural language prefix + generated tokens
        # (predicting token t+1 given natural prefix + generated tokens 0..t)
        with torch.no_grad(): # Ensure orig model isn't updated by this loss component
            num_generated_tokens = gen.hard_token_ids.shape[-1]
            orig_model_pred_logits_all_pos = orig.model(
                input_ids=base_model_input_ids,
            ).logits[:, slice_index:slice_index+num_generated_tokens, :] # Shape: (B, t_text, V)
            orig_model_pred_logprobs_all_pos = torch.log_softmax(orig_model_pred_logits_all_pos, dim=-1)
            trimmed_orig_model_pred_logprobs_of_interest = torch.gather(
                orig_model_pred_logprobs_all_pos,
                dim=-1,
                index=gen.hard_token_ids.unsqueeze(-1)
            ).squeeze(-1)
                



    # ----------------------- KL divergence between original and reconstructed model outputs -----------------------------------
    # Always compute for monitoring, but detach gradients when weight is 0
    #if all(k in batch for k in ("A_prime", "input_ids_A")):
    # Determine context for gradient computation based on kl_base.
    # If kl_base is 0, ops in this block are for monitoring and shouldn't track gradients.
    # (Requires import contextlib; kl_base is a function argument)
    grad_context = torch.no_grad() if (kl_base == 0 and mse_w == 0) else contextlib.nullcontext()
    with grad_context: # This context applies to all subsequent ops in this block
        # Reconstruct A′ as well to get Δ
        if resample_ablation:
            Ap = batch["A_prime"]  # Keep original dtype - autocast will handle conversions
            # Assuming A_prime corresponds to the same original token position context as A.
            # If A_prime could have a different original_token_pos, that would need to be passed from the batch.
            # For now, use original_token_pos_A for Ap as well.
            original_token_pos_Ap = original_token_pos_A # Placeholder if specific Ap pos is needed
            tau = _loss_fns.get("tau", 1.0) if _loss_fns else 1.0
            with torch.no_grad() if enc.config.stop_grad_aprime else contextlib.nullcontext():
                # Use differentiable generation if configured
                if hasattr(dec.config, 'use_kv_cache') and dec.config.use_kv_cache:
                    gen_ap = dec.generate_soft_kv_cached(Ap, max_length=t_text, gumbel_tau=tau, original_token_pos=original_token_pos_Ap)
                elif hasattr(dec.config, 'use_checkpointing') and dec.config.use_checkpointing:
                    # Note: generate_soft_chkpt was not updated in this round of changes for original_token_pos
                    gen_ap = dec.generate_soft_chkpt(Ap, max_length=t_text, gumbel_tau=tau, original_token_pos=original_token_pos_Ap)
                else:
                    gen_ap = dec.generate_soft(Ap, max_length=t_text, gumbel_tau=tau, original_token_pos=original_token_pos_Ap)
                current_token_ids_ap = batch["input_ids_A"].gather(1, original_token_pos_Ap.unsqueeze(1)).squeeze(1)
                Ap_hat = enc(gen_ap.generated_text_embeddings, original_token_pos=original_token_pos_Ap, current_token_ids=current_token_ids_ap if enc.config.add_current_token else None) # Detached if kl_base == 0 via grad_context
                del gen_ap

                # Original delta logic is correct under grad_context:
                # If kl_base == 0, Ap_hat is detached. (Ap - Ap_hat).detach() is same as (Ap - Ap_hat).
                # If kl_base > 0, .detach() acts as intended for stop_grad_aprime.
                if enc.config.stop_grad_aprime:
                    delta = (Ap - Ap_hat).detach()  # stop-grad on A′ branch
                else:
                    delta = (Ap - Ap_hat)  # no stop-grad on A′ branch - this is the default
        
        if resample_ablation:
            A_train = A_hat + delta # Detached if kl_base == 0 via grad_context
        else:
            A_train = A_hat # this is the default

        # loss_mse – normalized by ||A||^2/D (per-sample), mean over batch
        if mse_w > 0:
            per_sample_mse = (A_train - A).pow(2).mean(dim=-1)
            denom = A.pow(2).mean(dim=-1).clamp_min(1e-12)
            loss_mse = (per_sample_mse / denom).mean()
        else:
            with torch.no_grad():
                per_sample_mse = (A_train - A).pow(2).mean(dim=-1)
                denom = A.pow(2).mean(dim=-1).clamp_min(1e-12)
                loss_mse = (per_sample_mse / denom).mean()
        if A.dim() == 2 and A.shape[0]!=1:
            A_variance = torch.var(A, dim=0).mean() # var over batch, mean over features
            residual_variance = torch.var(A_train.detach()-A, dim=0).mean()
        else:
            log.warning("A is 2D and batch size is 1, so we cannot compute variance")
            A_variance = torch.tensor(1.0)
            residual_variance = torch.tensor(0)
        with torch.no_grad():
            fraction_variance_explained = 1 - (residual_variance / A_variance)
            mean_normalised_rmse = (A_train.detach()-A).pow(2).sum(dim=0).mean()/A_variance
        

        input_ids_batch = batch["input_ids_A"]
        layer_idx_batch = batch.get("layer_idx")  # Don't squeeze yet
        token_pos_batch = batch.get("token_pos_A")  # Don't squeeze yet. This is the same as original_token_pos_A before squeezing.
                                                  # It's used here for indexing logits.
        B = A.shape[0]

        # Check if we can use vectorized approach (same layer across batch)
        if layer_idx_batch is not None:
            if layer_idx_batch.dim() == 0:
                layer_idx = int(layer_idx_batch.item())
            else:
                layer_idx = int(layer_idx_batch[0].item())
        else:
            layer_idx = 0
        
        # Now squeeze for later use
        if layer_idx_batch is not None:
            layer_idx_batch = layer_idx_batch.squeeze()
        if token_pos_batch is not None:
            token_pos_batch = token_pos_batch.squeeze()
        use_vectorized = True
        
        if layer_idx_batch is not None and not torch.all(layer_idx_batch == layer_idx):
            # Mixed layers in batch - fall back to original grouped approach
            use_vectorized = False
        if do_kl_computation:
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
                with torch.no_grad() if kl_base==0 else contextlib.nullcontext():
                    logits_approx_full = orig.forward_with_replacement_vectorized(
                        input_ids=input_ids_batch,
                        new_activations=A_train,
                        layer_idx=layer_idx,
                        token_positions=token_pos_batch,
                        no_grad=False,
                    ).logits
                # Extract logits at target positions for each sample
                logits_approx = logits_approx_full[batch_indices, token_pos_batch].clone()  # (B, V)
                del logits_approx_full
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
        # KL(P||Q) penalises Q for assigning mass where P has none; encourages Q to cover P.
        loss_kl = compute_kl_divergence_robust(
            logits_approx=logits_approx,
            logits_orig=logits_orig
        )
    elif do_kl_computation:
        # Compute without gradients for monitoring
        with torch.no_grad():
            loss_kl = compute_kl_divergence_robust(
                logits_approx=logits_approx,
                logits_orig=logits_orig
            ).detach()
    else:
        loss_kl = torch.tensor(0.0, device=A.device, dtype=A.dtype)

    hard_token_ids = gen.hard_token_ids
    

    if GRPO_w > 0:
        if not verbose_eval:
            del gen
        probs_of_interest, entropy = dec.fwd_tokens(
            activation_input=A_repeat if mean_n_sequences and mean_n_sequences>1 else A,
            input_tokens=hard_token_ids,
            original_token_pos=original_token_pos_A,
            detach_entropy=ent_w==0,
            calculate_entropy=ent_w!=0 or GRPO_validate_mode,
            return_logits=False
        )#probs shape (B,t_text
            
        if ent_w==0:
            if entropy is not None and ent_w==0:
                entropy = entropy.detach()
                entropy=entropy.mean()
            else:
                entropy = torch.tensor(0.0, device=A.device, dtype=A.dtype)
        elif ent_w!=0:
            if entropy_clamp and not GRPO_validate_mode:
                entropy = torch.clamp(entropy.mean(dim=-1), min=0.0, max=entropy_clamp)# only clip if the mean of sequence is above the threshold
            entropy = entropy.mean()
        if GRPO_beta!=0:
            # print("probs",probs_of_interest[:10])
            # print("orig",trimmed_orig_model_pred_logprobs_of_interest[:10])
            # print("ratio",probs_of_interest/trimmed_orig_model_pred_logprobs_of_interest)
            KL_GRPO = KL_schulman_estimator(probs_of_interest[:,skip_tokens_KL_GRPO:],trimmed_orig_model_pred_logprobs_of_interest[:,skip_tokens_KL_GRPO:] )
            #raise ValueError("KL_GRPO is not implemented")
        else:   
            KL_GRPO = torch.tensor(0.0, device=A.device, dtype=A.dtype)
        with torch.no_grad():
            #A and A-train are both not-expanded
            if not GRPO_validate_mode:
                advantage, mean_reward_std, mean_reward = compute_advantages(A,A_train.detach(),group_n)
                
                # Track proportion of zero advantages
                zero_advantage_threshold = 1e-6  # Consider advantages below this threshold as zero
                zero_advantages_mask = torch.abs(advantage) < zero_advantage_threshold
                zero_advantages_proportion = zero_advantages_mask.float().mean()
                
                advantage = advantage.unsqueeze(-1) # so we can broadcast over sequence length??
                if mean_n_sequences and mean_n_sequences>1:
                    # repeat so each element of average is rewarded
                    advantage = advantage.repeat_interleave(mean_n_sequences, dim=0)
            else:
                mean_reward_std, mean_reward = compute_advantages(A,A_train.detach(),group_n, no_advantage=True)
                advantage = torch.tensor(0.0, device=A.device, dtype=A.dtype).unsqueeze(-1)
                zero_advantages_proportion = torch.tensor(0.0, device=A.device, dtype=A.dtype)

        #fine to do mean because our rollouts are fixed length
        loss_GRPO = -torch.mean(probs_of_interest/probs_of_interest.detach() * advantage - GRPO_beta*KL_GRPO,axis=-1)
        loss_GRPO = loss_GRPO.mean() # align with canonical GRPO normalisation.
    else:
        advantage = torch.tensor(0.0, device=A.device, dtype=A.dtype) 
        # ------------------ entropy (optional regulariser) ---------------------
        # Always compute for monitoring, but detach gradients when weight is 0
        logits = gen.raw_lm_logits  # (B, T, V) from Decoder – still useful for entropy
        if ent_w != 0:  # Can be positive (reward entropy) or negative (penalize entropy)
            entropy = compute_entropy_robust(logits)
        else:
            with torch.no_grad():
                entropy = compute_entropy_robust(logits).detach()
        loss_GRPO = torch.tensor(0.0, device=A.device, dtype=A.dtype)

    # Clean up large tensors if not in verbose evaluation mode
    # if not verbose_eval:
    #     if hasattr(gen, 'raw_lm_logits'):
    #         # gen.raw_lm_logits has been used for loss_lm and potentially non-GRPO entropy.
    #         # It's no longer needed if not in verbose_eval mode.
    #         del gen.raw_lm_logits
        
    #     # Delete intermediate logits from KL computation if they were created
    #     if 'logits_orig' in locals():
    #         del logits_orig
    #     if 'logits_approx' in locals():
    #         del logits_approx

    # Total loss composition:
    # - KL loss (fundamental objective): fixed weight, measures functional preservation
    # - LM loss (linguistic regularizer): ramped up via alpha schedule for fluency
    # - MSE loss (direct reconstruction): alternative/additional to KL for direct activation matching
    # - Alpha schedule gradually introduces linguistic constraints during training
    #total_loss = (lm_w * alpha) * loss_lm + kl_base * loss_kl - ent_w * entropy + mse_w * loss_mse + GRPO_w * loss_GRPO
    loss1 = (lm_w * alpha) * loss_lm + kl_base * loss_kl - ent_w * entropy + GRPO_w * loss_GRPO
    loss2 = mse_w * loss_mse 
    total_loss = loss1 + loss2
    if debug_mode:
        # Debug: Check individual loss components   
        if torch.isnan(loss_lm):
            log.error(f"NaN in loss_lm. Weight: lm_w={lm_w}, alpha={alpha}")
        if torch.isnan(loss_kl):
            log.error(f"NaN in loss_kl. Weight: kl_base={kl_base}")
        if torch.isnan(entropy):
            log.error(f"NaN in entropy. Weight: ent_w={ent_w}")
        if torch.isnan(loss_mse):
            log.error(f"NaN in loss_mse. Weight: mse_w={mse_w}")
        if torch.isnan(loss_GRPO):
            log.error(f"NaN in loss_GRPO. Weight: GRPO_w={GRPO_w}")
        if torch.isnan(total_loss):
            log.error("NaN in total_loss after combining components")
    
        # Debug: Check for inf values
        if torch.isinf(total_loss):
            log.error(f"Inf in total_loss. Components: lm={loss_lm.item():.2e}, kl={loss_kl.item():.2e}, ent={entropy.item():.2e}, mse={loss_mse.item():.2e}, GRPO={loss_GRPO.item():.2e}")

    # Build return dictionary
    result_dict = {
        "total": total_loss,
        "total_loss_1": loss1,
        "total_loss_2": loss2,
        "mse": loss_mse,  # For logging
        "lm": loss_lm,
        "kl": loss_kl,
        "entropy": entropy,
        "decoded_tokens_batch": decoded_token_ids_batch,  # Add this for epoch-level aggregation
        "fraction_variance_explained": fraction_variance_explained,
        "mean_normalised_rmse": mean_normalised_rmse,
        "advantages_mean": advantage.mean() if GRPO_training and not GRPO_validate_mode else torch.tensor(0.0),
        "advantages_std": advantage.std() if GRPO_training and not GRPO_validate_mode else torch.tensor(0.0),
        "mean_reward": mean_reward if GRPO_training and not GRPO_validate_mode else torch.tensor(0.0),
        "mean_reward_std": mean_reward_std if GRPO_training and not GRPO_validate_mode else torch.tensor(0.0),
        "KL_GRPO": KL_GRPO if GRPO_training else torch.tensor(0.0),
        "loss_GRPO": loss_GRPO,
        "zero_advantages_proportion": zero_advantages_proportion if GRPO_training and not GRPO_validate_mode else torch.tensor(0.0),
    }
    
    if return_reconstruction:
        result_dict["reconstruction"] = A_hat.detach().cpu()
    
    # Add intervention results if we're in eval mode
    if intervention_results:
        # Add all intervention metrics with prefix
        for k, v in intervention_results.items():
            if k != "verbose_data":
                result_dict[f"intervention_{k}"] = v
            else:
                # Add verbose data without prefix
                result_dict["intervention_verbose_data"] = v
    
    # If in verbose mode, include intermediate tensors needed for analysis
    if verbose_eval:
        result_dict["verbose_intermediate"] = {
            "gen": gen,  # Contains generated embeddings and hard token IDs
            "A_hat": A_hat,  # Decoder reconstruction
            "A_train": A_train if 'A_train' in locals() else None,  # KL training activation
            "logits_orig": logits_orig if 'logits_orig' in locals() else None,  # Original logits at position
            "logits_approx": logits_approx if 'logits_approx' in locals() else None,  # Approx logits at position
        }
    
    return result_dict
