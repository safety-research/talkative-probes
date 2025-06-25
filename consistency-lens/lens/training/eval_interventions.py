"""Evaluation interventions for decoder outputs."""

from typing import Dict, Optional, Any
import torch
from torch import nn
import random

__all__ = ["run_eval_interventions"]


def run_eval_interventions(
    generated_embeddings: torch.Tensor,  # Decoder output
    enc: nn.Module,
    orig_A: torch.Tensor,
    A_hat_decoder: torch.Tensor,  # Already computed decoder reconstruction
    batch: Dict[str, torch.Tensor],
    orig_model: nn.Module,
    verbose: bool = False,
    number_of_examples: int = 8,
    decoder_prompt: Optional[str] = None,  # New argument for hard prompt
    dec: Optional[nn.Module] = None,  # New argument for decoder model
    tok: Optional[Any] = None,  # New argument for tokenizer
    sch_args: Optional[Dict[str, Any]] = None,  # New argument for schedule args
) -> Dict[str, torch.Tensor]:
    """Run interventions comparing decoder output to baseline token embeddings.
    
    This function compares the decoder's generated explanation against a baseline
    where we just take the original tokens from the sequence that produced A.
    
    Args:
        generated_embeddings: Decoder output embeddings (B, t_text, D)
        enc: Encoder model
        orig_A: Original activations (B, D) from position p in layer L
        A_hat_decoder: Already computed decoder reconstruction (B, D)
        batch: Batch data containing input_ids_A, token_pos_A
        orig_model: Original model to get token embeddings
        verbose: Whether to include per-sample verbose data
        decoder_prompt: Decoder prompt text for hard prompt intervention
        dec: Decoder model for hard prompt generation
        tok: Tokenizer for hard prompt generation
        sch_args: Schedule arguments including 'tau' for generation
        
    Returns:
        Dictionary with intervention results
    """
    B, t_text, D = generated_embeddings.shape
    results = {}
    
    # Get the position p where activation A was taken from
    token_pos_batch = batch.get("token_pos_A").squeeze()  # (B,)
    input_ids = batch["input_ids_A"]  # (B, seq_len)
    if B == 1:
        token_pos_batch=token_pos_batch.repeat(2)
        input_ids=input_ids.repeat(2,1)
    
    # Restrict to at most 10 examples for efficiency
    if B > number_of_examples:
        B = number_of_examples
        generated_embeddings = generated_embeddings[:B]
        token_pos_batch = token_pos_batch[:B]
        input_ids = input_ids[:B]
        orig_A = orig_A[:B]
        A_hat_decoder = A_hat_decoder[:B]
    
    # Baseline: Take t_text tokens from the original sequence up to and including position p
    baseline_embeddings_list = []
    
    for b in range(B):
        p = int(token_pos_batch[b].item())

        # Get tokens ending at position p (inclusive)
        # We want t_text tokens, so we take from (p - t_text + 1) to p (inclusive)
        start_pos = max(0, p - t_text + 1)
        end_pos = p + 1  # +1 because we want inclusive

        # Extract those token IDs
        baseline_token_ids = input_ids[b, start_pos:end_pos]  # Shape: (actual_len,)

        # Get embeddings for these tokens
        token_embeddings = orig_model.model.get_input_embeddings()(baseline_token_ids)  # (actual_len, D)

        # If we got fewer than t_text tokens (because p is near the start), pad with EOS token
        if token_embeddings.shape[0] < t_text:
            eos_token_id = orig_model.model.config.eos_token_id
            if eos_token_id is None:
                eos_token_id = orig_model.model.config.pad_token_id

            if eos_token_id is None:
                raise ValueError("Model has no EOS or PAD token configured for padding")

            eos_embedding = orig_model.model.get_input_embeddings()(
                torch.tensor([eos_token_id], device=token_embeddings.device)
            ).squeeze(0)  # (D,)

            num_pad = t_text - token_embeddings.shape[0]
            padding = eos_embedding.unsqueeze(0).expand(num_pad, -1)  # (num_pad, D)
            token_embeddings = torch.cat([padding, token_embeddings], dim=0)

        baseline_embeddings_list.append(token_embeddings)

    # Stack to get (B, t_text, D)
    baseline_embeddings = torch.stack(baseline_embeddings_list, dim=0)

    # Gather current_token_ids for each sample using token_pos_batch
    # input_ids: (B, seq_len), token_pos_batch: (B,)
    current_token_ids = torch.gather(
        input_ids, 1, token_pos_batch.unsqueeze(1)
    ).squeeze(1)  # (B,)

    # Now we can run interventions:

    # 1. Baseline reconstruction: encode the original tokens directly
    if enc.config.add_current_token:
        A_hat_baseline = enc(baseline_embeddings, current_token_ids=current_token_ids)
    else:
        A_hat_baseline = enc(baseline_embeddings, current_token_ids=None)
    mse_baseline = torch.nn.functional.mse_loss(A_hat_baseline, orig_A)
    results["mse_baseline"] = mse_baseline.item()

    # 2. Decoder reconstruction: use the already computed reconstruction
    mse_decoder = torch.nn.functional.mse_loss(A_hat_decoder, orig_A)
    results["mse_decoder"] = mse_decoder.item()

    # 3. Shuffle intervention: shuffle first (n-3) tokens of decoder output
    n_tokens = t_text
    shuffled_embeds = generated_embeddings.clone()
    shuffle_count = max(0, n_tokens - 3)

    if shuffle_count > 0:
        for b in range(B):
            indices = torch.randperm(shuffle_count, device=generated_embeddings.device)
            shuffled_embeds[b, :shuffle_count] = generated_embeddings[b, indices]

    if enc.config.add_current_token:
        A_hat_shuffled = enc(shuffled_embeds, current_token_ids=current_token_ids)
    else:
        A_hat_shuffled = enc(shuffled_embeds, current_token_ids=None)
    mse_shuffled = torch.nn.functional.mse_loss(A_hat_shuffled, orig_A)
    results["mse_shuffle"] = mse_shuffled.item()

    # 4. Full shuffle intervention: shuffle ALL tokens of decoder output
    full_shuffled_embeds = generated_embeddings.clone()
    for b in range(B):
        indices = torch.randperm(t_text, device=generated_embeddings.device)
        full_shuffled_embeds[b] = generated_embeddings[b, indices]

    if enc.config.add_current_token:
        A_hat_full_shuffled = enc(full_shuffled_embeds, current_token_ids=current_token_ids)
    else:
        A_hat_full_shuffled = enc(full_shuffled_embeds, current_token_ids=None)
    mse_full_shuffled = torch.nn.functional.mse_loss(A_hat_full_shuffled, orig_A)
    results["mse_shuffle_all"] = mse_full_shuffled.item()
    
    # Compute KL divergences if we have the necessary components
    if all(k in batch for k in ["layer_idx"]):
        layer_idx = int(batch.get("layer_idx").squeeze()[0].item())
        
        with torch.no_grad():
            # Original logits
            logits_orig = orig_model.forward_with_replacement_vectorized(
                input_ids=input_ids,
                new_activations=orig_A,
                layer_idx=layer_idx,
                token_positions=token_pos_batch,
                no_grad=True,
            ).logits
            batch_indices = torch.arange(B, device=orig_A.device)
            logits_orig = logits_orig[batch_indices, token_pos_batch]  # (B, V)
            
            # Compute KL for each reconstruction
            from lens.evaluation.metrics import kl as kl_fn
            
            # Baseline KL
            logits_baseline = orig_model.forward_with_replacement_vectorized(
                input_ids=input_ids,
                new_activations=A_hat_baseline,
                layer_idx=layer_idx,
                token_positions=token_pos_batch,
                no_grad=True,
            ).logits[batch_indices, token_pos_batch]
            
            kl_baseline = kl_fn(
                torch.log_softmax(logits_baseline.float(), dim=-1),
                torch.softmax(logits_orig.float(), dim=-1)
            )
            results["kl_baseline"] = kl_baseline.item()
            
            # Decoder KL
            logits_decoder = orig_model.forward_with_replacement_vectorized(
                input_ids=input_ids,
                new_activations=A_hat_decoder,
                layer_idx=layer_idx,
                token_positions=token_pos_batch,
                no_grad=True,
            ).logits[batch_indices, token_pos_batch]
            
            kl_decoder = kl_fn(
                torch.log_softmax(logits_decoder.float(), dim=-1),
                torch.softmax(logits_orig.float(), dim=-1)
            )
            results["kl_decoder"] = kl_decoder.item()
            
            # Shuffle KL
            logits_shuffled = orig_model.forward_with_replacement_vectorized(
                input_ids=input_ids,
                new_activations=A_hat_shuffled,
                layer_idx=layer_idx,
                token_positions=token_pos_batch,
                no_grad=True,
            ).logits[batch_indices, token_pos_batch]
            
            kl_shuffled = kl_fn(
                torch.log_softmax(logits_shuffled.float(), dim=-1),
                torch.softmax(logits_orig.float(), dim=-1)
            )
            results["kl_shuffle"] = kl_shuffled.item()
            
            # Full shuffle KL
            logits_full_shuffled = orig_model.forward_with_replacement_vectorized(
                input_ids=input_ids,
                new_activations=A_hat_full_shuffled,
                layer_idx=layer_idx,
                token_positions=token_pos_batch,
                no_grad=True,
            ).logits[batch_indices, token_pos_batch]
            
            kl_full_shuffled = kl_fn(
                torch.log_softmax(logits_full_shuffled.float(), dim=-1),
                torch.softmax(logits_orig.float(), dim=-1)
            )
            results["kl_shuffle_all"] = kl_full_shuffled.item()
            
            # Hard prompt KL
            if decoder_prompt is not None and dec is not None and tok is not None and sch_args is not None:
                # Tokenize the hard prompt
                left_ids, right_ids, _, _ = dec.tokenize_and_embed_prompt(decoder_prompt, tok)
                
                # Generate using hard prompts with the same activation input
                # Get t_text from config if available, otherwise use shape
                t_text_config = sch_args.get("t_text", t_text)
                tau = sch_args.get("tau", 1.0)
                
                # Generate with hard prompts for the whole batch at once, no fallback
                gen_hard = dec.generate_soft_kv_cached_nondiff(
                    orig_A,
                    max_length=t_text_config,
                    gumbel_tau=tau,
                    override_model_base_and_out=orig_model,
                    hard_left_emb=left_ids,
                    hard_right_emb=right_ids,
                    use_projection=True,
                    return_logits=True
                ).detach()
                hard_prompt_embeddings = gen_hard.generated_text_embeddings
                # Pass through encoder
                if enc.config.add_current_token:
                    A_hat_hard_prompt = enc(hard_prompt_embeddings, current_token_ids=current_token_ids)
                else:
                    A_hat_hard_prompt = enc(hard_prompt_embeddings, current_token_ids=None)
                
                mse_hard_prompt = torch.nn.functional.mse_loss(A_hat_hard_prompt, orig_A)
                results["mse_hard_prompt"] = mse_hard_prompt.item()
                
                # Hard prompt KL
                logits_hard_prompt = orig_model.forward_with_replacement_vectorized(
                    input_ids=input_ids,
                    new_activations=A_hat_hard_prompt,
                    layer_idx=layer_idx,
                    token_positions=token_pos_batch,
                    no_grad=True,
                ).logits
                
                batch_indices = torch.arange(B, device=orig_A.device)
                logits_hard_prompt = logits_hard_prompt[batch_indices, token_pos_batch]
                
                kl_hard_prompt = kl_fn(
                    torch.log_softmax(logits_hard_prompt.float(), dim=-1),
                    torch.softmax(logits_orig.float(), dim=-1)
                )
                results["kl_hard_prompt"] = kl_hard_prompt.item()
    
    # Add verbose data if requested
    if verbose:
        verbose_data = []
        for b in range(B):
            p = int(token_pos_batch[b].item())
            start_pos = max(0, p - t_text + 1)
            end_pos = p + 1
            
            sample_data = {
                "sample_idx": b,
                "position_p": p,
                "baseline_token_ids": input_ids[b, start_pos:end_pos].cpu().tolist(),
                "mse_baseline": mse_baseline.item(),
                "mse_decoder": mse_decoder.item(),
                "mse_shuffle": mse_shuffled.item(),
                "mse_shuffle_all": mse_full_shuffled.item(),
            }
            
            if "kl_baseline" in results:
                sample_data["kl_baseline"] = kl_baseline.item()
                sample_data["kl_decoder"] = kl_decoder.item()
                sample_data["kl_shuffle"] = kl_shuffled.item()
                sample_data["kl_shuffle_all"] = kl_full_shuffled.item()
            
            if "mse_hard_prompt" in results:
                sample_data["mse_hard_prompt"] = mse_hard_prompt.item()
                if "kl_hard_prompt" in results:
                    sample_data["kl_hard_prompt"] = kl_hard_prompt.item()
                
            verbose_data.append(sample_data)
        
        results["verbose_data"] = verbose_data
    
    return results