"""Shared verbose sample analysis functionality for evaluation and training."""

from __future__ import annotations

import torch
from typing import Any, Dict, List, Optional, NamedTuple, Union, Tuple
from transformers import PreTrainedTokenizerBase
from torch.nn import functional as F  # Added for gumbel_softmax

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
    decoder_tokens: List[str],
    decoder_preds_by_rank: List[List[str]],
    base_tokens: List[str],
    base_tokens_hard: List[str],
    base_tokens_hard_no_map: List[str],
    base_preds_by_rank: List[List[str]],
    base_preds_by_rank_hard: List[List[str]],
    base_preds_by_rank_hard_no_map: List[List[str]],
    top_n_analysis_val: int,
    original_string_cropped: str,
    autoregressive_continuation: Optional[str] = None,
    a_prime_string_cropped: Optional[str] = None,
    cfg: Dict[str, Any] = None,
) -> None:
    """Prints detailed information for a single verbose sample."""
    print("--- Verbose sample ---")
    # Print the wider text context, assumed to be pre-cropped appropriately
    print(f"Original Text (cropped): \"{escape_newlines(original_string_cropped)}\"")
    
    # Print A' context if provided
    if a_prime_string_cropped:
        print(f"A' Text (cropped): \"{escape_newlines(a_prime_string_cropped)}\"")
    
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
    if decoder_tokens:
        dec_labels = ["Token:"] + [f"Dec Top {i+1}:" for i in range(len(decoder_preds_by_rank))]
        dec_data_rows = [decoder_tokens] + decoder_preds_by_rank
        print_formatted_table(dec_labels, dec_data_rows)
    else:
        print("  (No explanation generated or empty)")
    
    # ------------------------------------------------------------------
    # Base model generation comparison
    # ------------------------------------------------------------------

    print("\nGenerated Explanation (from BASE model using same context + still with soft tokens??? Perhaps should be with hard tokens... not sure.):")
    if base_tokens:
        base_labels = ["Token:"] + [f"Base Top {i+1}:" for i in range(len(base_preds_by_rank))]
        base_data_rows = [base_tokens] + base_preds_by_rank
        print_formatted_table(base_labels, base_data_rows)
    else:
        print("  (No explanation generated or empty)")

    print(f"\nGenerated Explanation (from BASE model using hard context = tuned talkative probe): {cfg['decoder_prompt']}")
    if base_tokens_hard:
        base_labels = ["Token:"] + [f"Base Top {i+1}:" for i in range(len(base_preds_by_rank_hard))]
        base_data_rows = [base_tokens_hard] + base_preds_by_rank_hard
        print_formatted_table(base_labels, base_data_rows)
    else:
        print("  (No explanation generated or empty)")

    print(f"\nGenerated Explanation (from BASE model using hard context = bare talkative probe): {cfg['decoder_prompt']}")
    if base_tokens_hard_no_map:
        base_labels = ["Token:"] + [f"Base Top {i+1}:" for i in range(len(base_preds_by_rank_hard_no_map))]
        base_data_rows = [base_tokens_hard_no_map] + base_preds_by_rank_hard_no_map
        print_formatted_table(base_labels, base_data_rows)
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
    return_structured_data: bool = False,
    capture_output: bool = False,
) -> Union[int, Tuple[int, List[Dict[str, Any]]], Tuple[int, str]]:
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
        return_structured_data: If True, return structured data for logging
        capture_output: If True, capture console output and return it
        
    Returns:
        If return_structured_data is False and capture_output is False: Number of samples printed from this batch
        If return_structured_data is True: Tuple of (num_printed, structured_samples_list)
        If capture_output is True: Tuple of (num_printed, captured_output_string)
    """
    dec = models["dec"]
    enc = models["enc"]
    num_printed_this_batch = 0
    structured_samples = [] if return_structured_data else None
    captured_output = [] if capture_output else None

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
        
        # Get A_prime_i early for the intervention test
        alt_idx = (i + 1) % batch["A_prime"].size(0)
        A_prime_i = batch["A_prime"][alt_idx : alt_idx + 1].to(device)
        
        # Base Model (A') - prediction using A_prime_i at (l,p)
        logits_aprime_all_pos = orig.forward_with_replacement(input_ids_seq, A_prime_i, l, p).logits # Shape [1, seq_len, vocab_size]
        logits_aprime_at_p = logits_aprime_all_pos[:, p].squeeze(0) # Shape [vocab_size]
        top_n_aprime_tokens = get_top_n_tokens(logits_aprime_at_p, tok, top_n_analysis)
        
        # Lens Recon (A_hat) - prediction using A_hat_single at (l,p)
        gen_single = dec.generate_soft(A_i, max_length=cfg["t_text"], gumbel_tau=sch_args["tau"])
        A_hat_single = enc(gen_single.generated_text_embeddings)
        logits_target_all_pos = orig.forward_with_replacement(input_ids_seq, A_hat_single, l, p).logits
        logits_target_at_p = logits_target_all_pos[:, p].squeeze(0)
        top_n_lens_recon_tokens = get_top_n_tokens(logits_target_at_p, tok, top_n_analysis)

        # Resample Ablation (A_hat+Δ) - prediction using A_target_i at (l,p)
        # Note: A_prime_i already defined above
        
        # Get A_prime input sequence and position if available
        a_prime_string_cropped = None
        if "input_ids_A_prime" in batch and "token_pos_A_prime" in batch:
            input_ids_A_prime_seq = batch["input_ids_A_prime"][alt_idx].to(device)
            p_prime = int(batch["token_pos_A_prime"][alt_idx].item())
            
            # Create cropped string for A_prime
            raw_prime_ids = input_ids_A_prime_seq.tolist()
            crop_start_idx_prime = max(0, p_prime - 100)
            crop_end_idx_prime = min(len(raw_prime_ids), p_prime + 30 + 1)
            
            # Decode the parts separately to insert stars around the analyzed token
            before_p_prime = tok.decode(raw_prime_ids[crop_start_idx_prime:p_prime]) if p_prime > crop_start_idx_prime else ""
            token_at_p_prime = tok.decode([raw_prime_ids[p_prime]]) if p_prime < len(raw_prime_ids) else ""
            after_p_prime = tok.decode(raw_prime_ids[p_prime+1:crop_end_idx_prime]) if p_prime+1 < crop_end_idx_prime else ""
            
            # Build the cropped string with the analyzed token highlighted
            a_prime_string_cropped = escape_newlines(before_p_prime + "*" + token_at_p_prime + "*" + after_p_prime)
            
            # Add ellipsis if cropped
            if crop_start_idx_prime > 0:
                a_prime_string_cropped = "..." + a_prime_string_cropped
            if crop_end_idx_prime < len(raw_prime_ids):
                a_prime_string_cropped = a_prime_string_cropped + "..."
        
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

        def build_topk_preds_by_rank(raw_lm_logits, num_tokens, k, tok):
            """Builds a list of rows: one row per rank (top-1, top-2, ...)."""
            if k <= 0:
                return []
            preds_by_rank = [[] for _ in range(k)]
            for logit_slice in raw_lm_logits[0][:num_tokens]:
                topk_ids = torch.topk(logit_slice, k=k).indices.tolist()
                for rank_idx, tok_id in enumerate(topk_ids):
                    preds_by_rank[rank_idx].append(
                        escape_newlines(tok.decode([tok_id]).strip())
                    )
            return preds_by_rank

        # ------------------------------------------------------------------
        # (1) Decoder explanation & stacked top-k predictions (per-rank rows)
        # ------------------------------------------------------------------

        gen_token_ids_full = gen_single.hard_token_ids[0].tolist()
        gen_tokens = [escape_newlines(tok.decode([tid])) for tid in gen_token_ids_full]

        k_for_decoder_preds = min(top_n_analysis, gen_single.raw_lm_logits.size(-1))

        decoder_preds_by_rank = build_topk_preds_by_rank(
            gen_single.raw_lm_logits, len(gen_tokens), k_for_decoder_preds, tok
        )

        # ------------------------------------------------------------------
        # (2) Base model generation with identical (learned) context + incl linear map                
        # ------------------------------------------------------------------
        base_gen_single = dec.generate_soft(A_i, max_length=cfg["t_text"], gumbel_tau=sch_args["tau"], override_model_base_and_out=orig, use_projection=True)
        # base_gen_single = generate_soft_with_base(
        #     orig,
        #     A_i,
        #     proj_layer=dec.proj,
        #     prompt_left_emb=dec.prompt_left_emb,
        #     prompt_right_emb=dec.prompt_right_emb,
        #     prompt_len=dec.prompt_len,
        #     max_length=cfg["t_text"],
        #     gumbel_tau=sch_args["tau"],
        #     device=device,
        # )

        base_token_ids_full = base_gen_single.hard_token_ids[0].tolist()
        base_gen_tokens = [escape_newlines(tok.decode([tid])) for tid in base_token_ids_full]

        base_preds_by_rank = build_topk_preds_by_rank(
            base_gen_single.raw_lm_logits, len(base_gen_tokens), k_for_decoder_preds, tok
        )

        # ------------------------------------------------------------------
        # (3 ) Base model generation with hard context but using linear map 
        # ------------------------------------------------------------------
        left_ids, right_ids, left_emb, right_emb = dec.tokenize_and_embed_prompt(cfg["decoder_prompt"], tok)
        base_gen_single_hard = dec.generate_soft(A_i, 
                                                 max_length=cfg["t_text"], 
                                                 gumbel_tau=sch_args["tau"], 
                                                 override_model_base_and_out=orig, 
                                                 hard_left_emb=left_ids, 
                                                 hard_right_emb=right_ids,
                                                 use_projection=True)

        base_token_ids_full_hard = base_gen_single_hard.hard_token_ids[0].tolist()
        base_gen_tokens_hard = [escape_newlines(tok.decode([tid])) for tid in base_token_ids_full_hard]

        base_preds_by_rank_hard = build_topk_preds_by_rank(
            base_gen_single_hard.raw_lm_logits, len(base_gen_tokens_hard), k_for_decoder_preds, tok
        )

        # (4) Base model generation with hard context but without linear map
        left_ids, right_ids, left_emb, right_emb = dec.tokenize_and_embed_prompt(cfg["decoder_prompt"], tok)
        base_gen_single_hard_no_map = dec.generate_soft(A_i, 
                                                        max_length=cfg["t_text"], 
                                                        gumbel_tau=sch_args["tau"], 
                                                        override_model_base_and_out=orig, 
                                                        hard_left_emb=left_ids, 
                                                        hard_right_emb=right_ids, 
                                                        use_projection=False)

        base_token_ids_full_hard_no_map = base_gen_single_hard_no_map.hard_token_ids[0].tolist()
        base_gen_tokens_hard_no_map = [escape_newlines(tok.decode([tid])) for tid in base_token_ids_full_hard_no_map]

        base_preds_by_rank_hard_no_map = build_topk_preds_by_rank(
            base_gen_single_hard_no_map.raw_lm_logits, len(base_gen_tokens_hard_no_map), k_for_decoder_preds, tok
        )
        # ------------------------------------------------------------------
        # END generation blocks
        # ------------------------------------------------------------------

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
            "Base Model (A')": top_n_aprime_tokens,
            "Log w/o Ablation (A_hat)": top_n_lens_recon_tokens,
            "Log w/Resample Ablation (A_hat+Δ)": top_n_resample_ablation_tokens,
        }

        # Generate autoregressive continuation from position p
        autoregressive_continuation = None
        if generate_continuation:
            autoregressive_continuation = generate_autoregressive_continuation(
                orig, input_ids_seq, p, num_tokens=continuation_tokens, tok=tok, device=device
            )

        # Collect structured data if requested
        if return_structured_data:
            # Get decoder predictions for position p - topk_per_pos contains strings, not tuples
            decoder_preds = []
            if p < gen_single.raw_lm_logits.size(1):
                # Get actual token IDs and probabilities from the raw logits
                logits_at_p = gen_single.raw_lm_logits[0, p]
                probs = torch.softmax(logits_at_p, dim=-1)
                top_values, top_indices = torch.topk(probs, k=min(top_n_analysis, probs.size(-1)))
                for i in range(len(top_indices)):
                    token_id = top_indices[i].item()
                    prob = top_values[i].item()
                    token_str = escape_newlines(tok.decode([token_id]))
                    decoder_preds.append((token_str, prob))
            
            # For original and reconstructed logits, we need to get probabilities too
            # Since get_top_n_tokens only returns strings, we need to get the probabilities from the original logits
            
            # Original logits with probabilities
            original_logits_with_probs = []
            if logits_orig_at_p.numel() > 0:
                orig_probs = torch.softmax(logits_orig_at_p, dim=-1)
                orig_top_values, orig_top_indices = torch.topk(orig_probs, k=min(top_n_analysis, orig_probs.size(-1)))
                for i in range(len(orig_top_indices)):
                    token_id = orig_top_indices[i].item()
                    prob = orig_top_values[i].item()
                    token_str = escape_newlines(tok.decode([token_id]))
                    original_logits_with_probs.append((token_str, prob))
            
            # Reconstructed logits with probabilities
            reconstructed_logits_with_probs = []
            if logits_target_at_p.numel() > 0:
                recon_probs = torch.softmax(logits_target_at_p, dim=-1)
                recon_top_values, recon_top_indices = torch.topk(recon_probs, k=min(top_n_analysis, recon_probs.size(-1)))
                for i in range(len(recon_top_indices)):
                    token_id = recon_top_indices[i].item()
                    prob = recon_top_values[i].item()
                    token_str = escape_newlines(tok.decode([token_id]))
                    reconstructed_logits_with_probs.append((token_str, prob))
            
            sample_data = {
                "input_text": escape_newlines(original_string_cropped),
                "chosen_token": escape_newlines(original_token_at_p_str),
                "position": p,
                "decoded_text": escape_newlines(" ".join(gen_tokens)),
                "top_predictions": decoder_preds,
                "continuation": escape_newlines(autoregressive_continuation) if autoregressive_continuation else None,
                "original_logits": original_logits_with_probs,
                "reconstructed_logits": reconstructed_logits_with_probs,
            }
            structured_samples.append(sample_data)
        
        if capture_output:
            # Capture the output instead of printing directly
            import io
            from contextlib import redirect_stdout
            
            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                print_verbose_sample_details(
                    l=l,
                    p=p,
                    original_token_at_p_str=original_token_at_p_str,
                    context_display_range=context_display_range,
                    context_labels=context_labels,
                    context_data_rows=context_data_rows,
                    analysis_predictions=analysis_preds_dict,
                    decoder_tokens=gen_tokens,
                    decoder_preds_by_rank=decoder_preds_by_rank,
                    base_tokens=base_gen_tokens,
                    base_tokens_hard=base_gen_tokens_hard,
                    base_tokens_hard_no_map=base_gen_tokens_hard_no_map,
                    base_preds_by_rank=base_preds_by_rank,
                    base_preds_by_rank_hard=base_preds_by_rank_hard,
                    base_preds_by_rank_hard_no_map=base_preds_by_rank_hard_no_map,
                    top_n_analysis_val=top_n_analysis,
                    original_string_cropped=original_string_cropped,
                    autoregressive_continuation=autoregressive_continuation,
                    a_prime_string_cropped=a_prime_string_cropped,
                    cfg=cfg,
                )
            captured_output.append(output_buffer.getvalue())
            # Also print to console
            print(captured_output[-1], end='')
        else:
            print_verbose_sample_details(
                l=l,
                p=p,
                original_token_at_p_str=original_token_at_p_str,
                context_display_range=context_display_range,
                context_labels=context_labels,
                context_data_rows=context_data_rows,
                analysis_predictions=analysis_preds_dict,
                decoder_tokens=gen_tokens,
                decoder_preds_by_rank=decoder_preds_by_rank,
                base_tokens=base_gen_tokens,
                base_tokens_hard=base_gen_tokens_hard,
                base_tokens_hard_no_map=base_gen_tokens_hard_no_map,
                base_preds_by_rank=base_preds_by_rank,
                base_preds_by_rank_hard=base_preds_by_rank_hard,
                base_preds_by_rank_hard_no_map=base_preds_by_rank_hard_no_map,
                top_n_analysis_val=top_n_analysis,
                original_string_cropped=original_string_cropped,
                autoregressive_continuation=autoregressive_continuation,
                a_prime_string_cropped=a_prime_string_cropped,
                cfg=cfg,
            )
        num_printed_this_batch += 1
        if (printed_count_so_far + num_printed_this_batch) >= num_samples:
            break
    
    # After all samples, print soft prompt projections (only once per batch)
    if num_printed_this_batch > 0:
        soft_prompt_projections = get_soft_prompt_projections(dec, enc, orig, tok)
        soft_prompt_text = format_soft_prompt_projections(soft_prompt_projections)
        
        if capture_output:
            captured_output.append(soft_prompt_text)
            print(soft_prompt_text)
        else:
            print(soft_prompt_text)
    
    if return_structured_data:
        return num_printed_this_batch, structured_samples
    elif capture_output:
        return num_printed_this_batch, "\n".join(captured_output)
    return num_printed_this_batch


# -----------------------------------------------------------------------------
# NOTE: Add helper to generate using the original (base) model with identical
#       context handling (prompt left/right + activation splice) to the Decoder
#       `generate_soft` method. This allows comparison between the trained
#       Decoder and the frozen base model when supplied the same projected
#       activation.
# -----------------------------------------------------------------------------

# class BaseGenerated(NamedTuple):
#     """Lightweight container mirroring `Decoder.Generated` for the base model."""

#     hard_token_ids: torch.Tensor
#     raw_lm_logits: torch.Tensor


# def generate_soft_with_base(
#     orig_model: OrigWrapper,
#     activation_input: torch.Tensor,
#     *,
#     proj_layer: torch.nn.Linear,
#     prompt_left_emb: torch.Tensor | None,
#     prompt_right_emb: torch.Tensor | None,
#     prompt_len: int,
#     max_length: int,
#     gumbel_tau: float,
#     device: torch.device,
# ) -> BaseGenerated:
#     """Autoregressively sample *soft* tokens from the *frozen* base model.

#     This closely follows `Decoder.generate_soft`, but relies solely on the
#     base model's LM head (tied embeddings) instead of a separate `out` layer.
#     """

#     # Ensure dtype/device consistency
#     activation_input = activation_input.to(proj_layer.weight.dtype).to(device)
#     if prompt_left_emb is not None:
#         prompt_left_emb = prompt_left_emb.to(device)
#     if prompt_right_emb is not None:
#         prompt_right_emb = prompt_right_emb.to(device)

#     # Build initial sequence of embeddings: <prompt_left> + proj(A) + <prompt_right>
#     parts: list[torch.Tensor] = []
#     if prompt_left_emb is not None:
#         parts.append(prompt_left_emb.expand(activation_input.size(0), -1, -1))

#     a_proj = proj_layer(activation_input).unsqueeze(1)
#     parts.append(a_proj)

#     if prompt_right_emb is not None:
#         parts.append(prompt_right_emb.expand(activation_input.size(0), -1, -1))

#     seq_embs = torch.cat(parts, dim=1)  # (B, prompt_len+1, d_model)

#     emb_table = orig_model.model.get_output_embeddings().weight  # (V, d_model)

#     logits_list: list[torch.Tensor] = []
#     hard_ids_list: list[torch.Tensor] = []

#     for _ in range(max_length):
#         outputs = orig_model.model(inputs_embeds=seq_embs, output_hidden_states=True)
#         h_last = (
#             outputs.last_hidden_state
#             if hasattr(outputs, "last_hidden_state")
#             else outputs.hidden_states[-1]
#         )

#         logits_t = orig_model.model.lm_head(h_last[:, -1])  # (B, V)

#         # Straight-Through Gumbel-Softmax sampling
#         ste_token_dist = F.gumbel_softmax(logits_t, tau=gumbel_tau, hard=True)
#         emb_t = ste_token_dist @ emb_table  # (B, d_model)

#         seq_embs = torch.cat([seq_embs, emb_t.unsqueeze(1)], dim=1)

#         logits_list.append(logits_t)
#         hard_ids_list.append(ste_token_dist.argmax(dim=-1))

#     logits_seq = torch.stack(logits_list, dim=1)
#     hard_ids = torch.stack(hard_ids_list, dim=1)

#     return BaseGenerated(hard_token_ids=hard_ids, raw_lm_logits=logits_seq)


def project_embeddings_to_text(
    embeddings: torch.Tensor,
    embedding_table: torch.Tensor,
    tok: PreTrainedTokenizerBase,
    top_k: int = 3
) -> List[Tuple[str, List[Tuple[str, float]]]]:
    """Project embedding vectors back to text by finding nearest neighbors in embedding space.
    
    Args:
        embeddings: Embedding vectors to project [num_embeddings, d_model]
        embedding_table: Vocabulary embedding table [vocab_size, d_model]
        tok: Tokenizer
        top_k: Number of nearest neighbors to return
        
    Returns:
        List of (top1_token, [(token, distance), ...]) for each embedding
    """
    results = []
    
    # Normalize embeddings for cosine similarity
    embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    embedding_table_norm = torch.nn.functional.normalize(embedding_table, p=2, dim=-1)
    
    # Compute similarities
    similarities = torch.matmul(embeddings_norm, embedding_table_norm.T)  # [num_embeddings, vocab_size]
    
    # Get top-k for each embedding
    for i in range(embeddings.shape[0]):
        top_values, top_indices = torch.topk(similarities[i], k=min(top_k, similarities.shape[1]))
        
        top_tokens = []
        for j in range(len(top_indices)):
            token_id = top_indices[j].item()
            similarity = top_values[j].item()
            token_str = escape_newlines(tok.decode([token_id]))
            top_tokens.append((token_str, similarity))
        
        # The top-1 token is the primary result
        top1_token = top_tokens[0][0] if top_tokens else ""
        results.append((top1_token, top_tokens))
    
    return results


def get_soft_prompt_projections(
    dec: Decoder,
    enc: Encoder,
    orig: OrigWrapper,
    tok: PreTrainedTokenizerBase,
) -> Dict[str, Any]:
    """Get text projections of learned soft prompts from decoder and encoder.
    
    Returns:
        Dictionary with projection results for display
    """
    projections = {}
    
    # Decoder projections
    if dec.prompt_left_emb is not None or dec.prompt_right_emb is not None:
        # Get embedding tables
        input_emb_table = dec.base.get_input_embeddings().weight
        output_emb_table = dec.base.get_output_embeddings().weight
        
        decoder_projections = {}
        
        # Project left prompt
        if dec.prompt_left_emb is not None:
            # Using input embeddings
            left_input_proj = project_embeddings_to_text(
                dec.prompt_left_emb, input_emb_table, tok, top_k=3
            )
            decoder_projections['left_input'] = left_input_proj
            
            # Using output embeddings
            left_output_proj = project_embeddings_to_text(
                dec.prompt_left_emb, output_emb_table, tok, top_k=3
            )
            decoder_projections['left_output'] = left_output_proj
        
        # Project right prompt
        if dec.prompt_right_emb is not None:
            # Using input embeddings
            right_input_proj = project_embeddings_to_text(
                dec.prompt_right_emb, input_emb_table, tok, top_k=3
            )
            decoder_projections['right_input'] = right_input_proj
            
            # Using output embeddings
            right_output_proj = project_embeddings_to_text(
                dec.prompt_right_emb, output_emb_table, tok, top_k=3
            )
            decoder_projections['right_output'] = right_output_proj
        
        projections['decoder'] = decoder_projections
    
    # Encoder projections (only input embeddings needed)
    if enc.soft_prompt_embeddings is not None:
        # Get input embedding table from encoder's base model
        if enc._use_base:
            input_emb_table = enc.base.get_input_embeddings().weight
        else:
            # Fallback to original model if encoder base not used
            input_emb_table = orig.model.get_input_embeddings().weight
        
        encoder_proj = project_embeddings_to_text(
            enc.soft_prompt_embeddings, input_emb_table, tok, top_k=3
        )
        projections['encoder'] = encoder_proj
    
    return projections


def format_soft_prompt_projections(projections: Dict[str, Any]) -> str:
    """Format soft prompt projections for display."""
    lines = []
    lines.append("\n--- Learned Soft Prompt Projections ---")
    
    # Decoder projections
    if 'decoder' in projections:
        lines.append("\nDecoder Soft Prompts:")
        dec_proj = projections['decoder']
        
        # Format left prompt
        if 'left_input' in dec_proj:
            left_input = dec_proj['left_input']
            left_output = dec_proj.get('left_output', [])
            
            lines.append("  Left prompt:")
            # Show as a single line with <embed> marker
            input_text = ''.join([t[0] for t, _ in left_input])
            output_text = ''.join([t[0] for t, _ in left_output]) if left_output else ""
            
            lines.append(f"    Input emb:  \"{input_text}<embed>\"")
            lines.append(f"    Output emb: \"{output_text}<embed>\"")
            
            # Show top-3 for each token
            lines.append("    Per-token top-3 (input emb):")
            for i, (_, top_tokens) in enumerate(left_input):
                token_strs = [f"'{t}'({s:.3f})" for t, s in top_tokens[:3]]
                lines.append(f"      Token {i}: {', '.join(token_strs)}")
        
        # Format right prompt
        if 'right_input' in dec_proj:
            right_input = dec_proj['right_input']
            right_output = dec_proj.get('right_output', [])
            
            lines.append("  Right prompt:")
            input_text = ''.join([t[0] for t, _ in right_input])
            output_text = ''.join([t[0] for t, _ in right_output]) if right_output else ""
            
            lines.append(f"    Input emb:  \"<embed>{input_text}\"")
            lines.append(f"    Output emb: \"<embed>{output_text}\"")
            
            lines.append("    Per-token top-3 (input emb):")
            for i, (_, top_tokens) in enumerate(right_input):
                token_strs = [f"'{t}'({s:.3f})" for t, s in top_tokens[:3]]
                lines.append(f"      Token {i}: {', '.join(token_strs)}")
    
    # Encoder projections
    if 'encoder' in projections:
        lines.append("\nEncoder Soft Prompt:")
        enc_proj = projections['encoder']
        
        # Show as a single line
        prompt_text = ''.join([t[0] for t, _ in enc_proj])
        lines.append(f"  Text: \"{prompt_text}\"")
        
        # Show top-3 for each token
        lines.append("  Per-token top-3:")
        for i, (_, top_tokens) in enumerate(enc_proj):
            token_strs = [f"'{t}'({s:.3f})" for t, s in top_tokens[:3]]
            lines.append(f"    Token {i}: {', '.join(token_strs)}")
    
    if not projections:
        lines.append("  (No soft prompts configured)")
    
    lines.append("-" * 60)
    return '\n'.join(lines)