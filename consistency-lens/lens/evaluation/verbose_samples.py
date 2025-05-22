"""Shared verbose sample analysis functionality for evaluation and training."""

from __future__ import annotations

import torch
from typing import Any, Dict, List, Optional
from transformers import PreTrainedTokenizerBase

from lens.models.decoder import Decoder
from lens.models.encoder import Encoder
from lens.models.orig import OrigWrapper


def escape_newlines(text: str) -> str:
    """Helper to escape newlines for display."""
    return text.replace("\n", "\\n")


def generate_autoregressive_continuation(
    orig_model: OrigWrapper,
    input_ids: torch.Tensor,
    start_position: int,
    num_tokens: int,
    tok: PreTrainedTokenizerBase,
    device: torch.device,
    temperature: float = 0.8,
    top_p: float = 0.9
) -> str:
    """Generate autoregressive continuation from a specific position.
    
    Args:
        orig_model: The original model wrapper
        input_ids: Input token IDs [batch_size, seq_len]
        start_position: Position to start generation from (0-indexed)
        num_tokens: Number of tokens to generate
        tok: Tokenizer
        device: Device to run on
        temperature: Sampling temperature
        top_p: Top-p sampling threshold
        
    Returns:
        Generated text string
    """
    # Get the prefix up to and including start_position
    if start_position >= input_ids.size(1):
        prefix_ids = input_ids
    else:
        prefix_ids = input_ids[:, :start_position + 1]
    
    generated_ids = prefix_ids.clone()
    
    with torch.no_grad():
        for _ in range(num_tokens):
            # Get model predictions
            outputs = orig_model.model(input_ids=generated_ids)
            next_token_logits = outputs.logits[:, -1, :] / temperature
            
            # Apply top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            # Set logits to -inf for removed indices
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Stop if we hit end of sequence token
            if tok.eos_token_id is not None and next_token.item() == tok.eos_token_id:
                break
    
    # Decode only the generated portion
    generated_portion = generated_ids[0, prefix_ids.size(1):].tolist()
    return tok.decode(generated_portion)


def get_top_n_tokens(logits_tensor_slice: torch.Tensor, tok: PreTrainedTokenizerBase, top_n: int) -> List[str]:
    """Decodes the top N tokens from a logits slice."""
    # Ensure logits_tensor_slice is 1D
    if logits_tensor_slice.dim() > 1:
        logits_tensor_slice = logits_tensor_slice.squeeze()
    if logits_tensor_slice.dim() == 0: # Should not happen if vocab_size > 0
        return ["ERR_LOGITS_DIM"] * top_n
    
    # Ensure top_n is not greater than vocab size
    effective_top_n = min(top_n, logits_tensor_slice.size(-1))
    if effective_top_n == 0:
        return []

    top_k_values, top_k_indices = torch.topk(logits_tensor_slice, k=effective_top_n, dim=-1)
    
    top_k_ids_list = top_k_indices.tolist()
    # topk might return a single int if k=1 and squeeze was used, ensure it's a list
    if not isinstance(top_k_ids_list, list):
        top_k_ids_list = [top_k_ids_list]
        
    return [escape_newlines(tok.decode([_id])) for _id in top_k_ids_list]


def print_formatted_table(labels: List[str], data_rows: List[List[str]]) -> None:
    """Prints a formatted table with row labels and data."""
    if not labels:
        return
    if len(labels) != len(data_rows):
        print("  (Error: Mismatch between number of labels and data rows for table printing)")
        return

    num_cols = 0
    for row_data in data_rows:
        if row_data:
            num_cols = len(row_data)
            break
    
    if num_cols == 0: # All data rows are empty. Print labels only.
        max_label_width_only = max(len(s) for s in labels) + 2 if labels else 0
        for label in labels:
            print(f"{label:<{max_label_width_only}}")
        return

    col_widths = [0] * num_cols
    for row_data in data_rows:
        if row_data and len(row_data) != num_cols:
            # This case should ideally be handled by the caller ensuring consistent row lengths.
            # For robustness, we proceed but it might lead to misalignment for this specific row.
            pass 
        for c_idx, cell in enumerate(row_data):
            if c_idx < num_cols:
                col_widths[c_idx] = max(col_widths[c_idx], len(str(cell)))

    max_label_width = max(len(s) for s in labels) + 2
    for r_idx, label in enumerate(labels):
        row_str = f"{label:<{max_label_width}}"
        current_data_row = data_rows[r_idx]
        for c_idx in range(num_cols):
            if c_idx < len(current_data_row):
                cell_content = str(current_data_row[c_idx])
                row_str += f" {cell_content:<{col_widths[c_idx]}} |"
            else: # Pad if current_data_row is shorter than num_cols
                row_str += f" {'':<{col_widths[c_idx]}} |"
        print(row_str.rstrip(" |"))


def print_verbose_sample_details(
    l: int,
    p: int,
    original_token_at_p_str: str,
    context_display_range: str,
    context_labels: List[str],
    context_data_rows: List[List[str]],
    analysis_predictions: Dict[str, List[str]],
    gen_tokens: List[str],
    topk_per_pos: List[str],
    top_n_analysis_val: int,
    original_string_cropped: str, # Accepts the wider context string
    autoregressive_continuation: Optional[str] = None,
) -> None:
    """Prints detailed information for a single verbose sample."""
    print("--- Verbose sample ---")
    # Print the wider text context, assumed to be pre-cropped appropriately
    print(f"Original Text (cropped): \"{escape_newlines(original_string_cropped)}\"")
    
    # Print autoregressive continuation if provided
    if autoregressive_continuation:
        print(f"\nModel's continuation from position {p}: \"{escape_newlines(autoregressive_continuation)}\"")
    
    print(f"\nLayer {l}, Position {p} (0-indexed for activation A_i from original token '{original_token_at_p_str}')\n")

    print(f"Original Input Context (window around P={p}, showing positions {context_display_range}):")
    print_formatted_table(context_labels, context_data_rows)

    print(f"\nAnalysis for token following Position P={p} (i.e., predicting for seq position {p+1}):")
    for pred_type, top_n_tokens_list in analysis_predictions.items():
        tokens_str = ", ".join([f"'{t}'" for t in top_n_tokens_list])
        print(f"  - {pred_type} (Top {top_n_analysis_val}): {tokens_str}")

    print("\nGenerated Explanation (from Decoder using A_i):")
    if gen_tokens:
        expl_labels = ["Token:", f"Decoder Top-{top_n_analysis_val} Preds:"]
        expl_data_rows = [gen_tokens, topk_per_pos]
        print_formatted_table(expl_labels, expl_data_rows)
    else:
        print("  (No explanation generated or empty)")
    
    print("-" * 60)


def process_and_print_verbose_batch_samples(
    batch: Dict[str, Any],
    cfg: Dict[str, Any],
    models: Dict[str, torch.nn.Module],
    orig: OrigWrapper,
    tok: PreTrainedTokenizerBase,
    sch_args: Dict[str, Any],
    device: torch.device,
    num_samples: int = 3,
    top_n_analysis: int = 3,
    printed_count_so_far: int = 0,
    generate_continuation: bool = True,
    continuation_tokens: int = 30,
) -> int:
    """Processes and prints verbose samples from a batch.
    
    Args:
        batch: Batch of activation data
        cfg: Configuration dictionary
        models: Dictionary with 'dec' and 'enc' models
        orig: Original model wrapper
        tok: Tokenizer
        sch_args: Schedule arguments including 'tau'
        device: Device to run on
        num_samples: Number of samples to print
        top_n_analysis: Number of top predictions to show
        printed_count_so_far: Number already printed (for limiting)
        generate_continuation: Whether to generate autoregressive continuation
        continuation_tokens: Number of tokens to generate for continuation
        
    Returns:
        Number of samples printed from this batch
    """
    dec = models["dec"]
    enc = models["enc"]
    num_printed_this_batch = 0

    for i in range(min(num_samples - printed_count_so_far, batch["A"].size(0))):
        l = int(batch["layer_idx"][i].item())
        p = int(batch["token_pos_A"][i].item())

        input_ids_seq = batch["input_ids_A"][i].unsqueeze(0).to(device)
        A_i = batch["A"][i : i + 1].to(device) # Shape: [1, hidden_size]

        # Logit Lens prediction from A_i
        A_i_cast = A_i.to(orig.model.lm_head.weight.dtype)
        logit_lens_logits_from_A_i = orig.model.lm_head(A_i_cast) # Shape [1, vocab_size]
        top_n_logit_lens_tokens = get_top_n_tokens(logit_lens_logits_from_A_i.squeeze(0), tok, top_n_analysis)

        # Forward passes for other logits (interventions)
        # Base Model (orig A) - prediction using A_i at (l,p)
        logits_orig_all_pos = orig.forward_with_replacement(input_ids_seq, A_i, l, p).logits # Shape [1, seq_len, vocab_size]
        logits_orig_at_p = logits_orig_all_pos[:, p].squeeze(0) # Shape [vocab_size]
        top_n_orig_A_tokens = get_top_n_tokens(logits_orig_at_p, tok, top_n_analysis)
        
        # Lens Recon (A_hat) - prediction using A_hat_single at (l,p)
        gen_single = dec.generate_soft(A_i, max_length=cfg["t_text"], gumbel_tau=sch_args["tau"])
        A_hat_single = enc(gen_single.generated_text_embeddings)
        logits_target_all_pos = orig.forward_with_replacement(input_ids_seq, A_hat_single, l, p).logits
        logits_target_at_p = logits_target_all_pos[:, p].squeeze(0)
        top_n_lens_recon_tokens = get_top_n_tokens(logits_target_at_p, tok, top_n_analysis)

        # Resample Ablation (A_hat+Δ) - prediction using A_target_i at (l,p)
        alt_idx = (i + 1) % batch["A_prime"].size(0)
        A_prime_i = batch["A_prime"][alt_idx : alt_idx + 1].to(device)
        gen_ap = dec.generate_soft(A_prime_i, max_length=cfg["t_text"], gumbel_tau=sch_args["tau"])
        A_prime_hat = enc(gen_ap.generated_text_embeddings)
        delta_res = (A_prime_i - A_prime_hat).detach()
        A_target_i = A_hat_single + delta_res
        logits_resample_all_pos = orig.forward_with_replacement(input_ids_seq, A_target_i, l, p).logits
        logits_resample_at_p = logits_resample_all_pos[:, p].squeeze(0)
        top_n_resample_ablation_tokens = get_top_n_tokens(logits_resample_at_p, tok, top_n_analysis)
        
        # Original input sequence processing & Base Model's natural prediction
        raw_prefix_ids = input_ids_seq[0].tolist()
        base_model_full_logits = orig.model(input_ids=input_ids_seq).logits # Shape [1, seq_len, vocab_size]
        
        top_n_natural_base_preds_for_p_plus_1: List[str]
        if p < base_model_full_logits.size(1): # Ensure p is a valid index for logits
            natural_base_logits_at_p = base_model_full_logits[0, p] # Logits for predicting token at p+1
            top_n_natural_base_preds_for_p_plus_1 = get_top_n_tokens(natural_base_logits_at_p, tok, top_n_analysis)
        else:
            # This case means p is at or beyond the last token, so no "next token" prediction from base model.
            top_n_natural_base_preds_for_p_plus_1 = ["N/A"] * top_n_analysis


        display_start_idx = max(0, p - 10)
        display_end_idx = min(len(raw_prefix_ids), p + 3 + 1)
        displayed_raw_ids = raw_prefix_ids[display_start_idx:display_end_idx]
        displayed_prefix_tokens = [escape_newlines(tok.decode([tid])) for tid in displayed_raw_ids]
        displayed_positions = [str(k) for k in range(display_start_idx, display_end_idx)]

        #for printing the original string cropped
        crop_start_idx = max(0, p - 100)
        crop_end_idx = min(len(raw_prefix_ids), p + 30 + 1)
        
        # Decode the parts separately to insert stars around the analyzed token
        before_p = tok.decode(raw_prefix_ids[crop_start_idx:p]) if p > crop_start_idx else ""
        token_at_p = tok.decode([raw_prefix_ids[p]]) if p < len(raw_prefix_ids) else ""
        after_p = tok.decode(raw_prefix_ids[p+1:crop_end_idx]) if p+1 < crop_end_idx else ""
        
        # Build the cropped string with the analyzed token highlighted
        original_string_cropped = escape_newlines(before_p + "*" + token_at_p + "*" + after_p)
        
        # Add ellipsis if cropped
        if crop_start_idx > 0:
            original_string_cropped = "..." + original_string_cropped
        if crop_end_idx < len(raw_prefix_ids):
            original_string_cropped = original_string_cropped + "..."
        
        original_token_at_p_str = escape_newlines(tok.decode([raw_prefix_ids[p]])) if p < len(raw_prefix_ids) else "N/A"

        # Generated explanation processing
        gen_token_ids_full = gen_single.hard_token_ids[0].tolist()
        gen_tokens = [escape_newlines(tok.decode([tid])) for tid in gen_token_ids_full]
        
        # Use top_n_analysis for decoder's own predictions during explanation generation.
        # Ensure k for topk is not larger than vocab size and not zero if top_n_analysis is zero.
        k_for_decoder_preds = min(top_n_analysis, gen_single.raw_lm_logits.size(-1))
        if top_n_analysis == 0: # if top_n_analysis is 0, effectively no tokens.
            k_for_decoder_preds = 0

        topk_per_pos = []
        if k_for_decoder_preds > 0:
            topk_per_pos = [
                ", ".join(f'"{escape_newlines(tok.decode([x_id]).strip())}"' 
                          for x_id in torch.topk(logit_slice, k=k_for_decoder_preds).indices.tolist())
                for logit_slice in gen_single.raw_lm_logits[0][: len(gen_tokens)]
            ]
        else: # Handle case where k_for_decoder_preds is 0
             topk_per_pos = [""] * len(gen_tokens)


        # Prepare context table data
        context_display_range = f"{display_start_idx}-{display_end_idx-1}" if displayed_positions else "empty"
        context_labels = ["Position:", "Token:", "BaseLM (shift):"]
        
        # For context table, we show single top predictions from base model
        preds_prefix_full_single_top = [
            escape_newlines(tok.decode([base_model_full_logits[0, t_idx].argmax().item()]))
            if t_idx < base_model_full_logits.size(1) else "N/A"
            for t_idx in range(len(raw_prefix_ids))
        ]
        shifted_preds_for_display = [
            preds_prefix_full_single_top[display_start_idx + k_rel - 1] 
            if display_start_idx + k_rel > 0 and display_start_idx + k_rel -1 < len(preds_prefix_full_single_top) 
            else ("" if display_start_idx + k_rel == 0 else "ERR_IDX")
            for k_rel in range(len(displayed_prefix_tokens))
        ]
        
        context_data_rows = [list(displayed_positions), list(displayed_prefix_tokens), list(shifted_preds_for_display)]
        relative_p = p - display_start_idx
        if 0 <= relative_p < len(displayed_prefix_tokens):
            context_data_rows[0][relative_p] = f"[{context_data_rows[0][relative_p]}]P"
            context_data_rows[1][relative_p] = f"*{context_data_rows[1][relative_p]}*"

        # Prepare analysis predictions dictionary
        analysis_preds_dict = {
            "Base Model's natural prediction": top_n_natural_base_preds_for_p_plus_1,
            "Logit Lens (from A_i)": top_n_logit_lens_tokens,
            "Base Model (orig A)": top_n_orig_A_tokens,
            "Log w/o Ablation (A_hat)": top_n_lens_recon_tokens,
            "Log w/Resample Ablation (A_hat+Δ)": top_n_resample_ablation_tokens,
        }

        # Generate autoregressive continuation from position p
        autoregressive_continuation = None
        if generate_continuation:
            autoregressive_continuation = generate_autoregressive_continuation(
                orig, input_ids_seq, p, num_tokens=continuation_tokens, tok=tok, device=device
            )

        print_verbose_sample_details(
            l, p, original_token_at_p_str,
            context_display_range, context_labels, context_data_rows,
            analysis_preds_dict,
            gen_tokens, topk_per_pos,
            top_n_analysis, original_string_cropped,
            autoregressive_continuation
        )
        num_printed_this_batch += 1
        if (printed_count_so_far + num_printed_this_batch) >= num_samples:
            break
    return num_printed_this_batch
