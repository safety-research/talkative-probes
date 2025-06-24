"""Shared verbose sample analysis functionality for evaluation and training."""

from __future__ import annotations

import torch
from typing import Any, Dict, List, Optional, NamedTuple, Union, Tuple, TYPE_CHECKING
from transformers import PreTrainedTokenizerBase
from torch.nn import functional as F  # Added for gumbel_softmax

from lens.models.decoder import Decoder
from lens.models.encoder import Encoder
from lens.models.orig import OrigWrapper
from lens.training.loop import compute_kl_divergence_robust

if TYPE_CHECKING:
    from tuned_lens import TunedLens


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
    top_p: Optional[float] = None,
    attention_mask: torch.Tensor | None = None,
) -> str:
    """Generate autoregressive continuation from a specific position.

    Args:
        orig_model: The original model wrapper.
        input_ids: Input token IDs [batch_size, seq_len].
        start_position: Position to start generation from. The token at this position
                      is included in the prompt.
        num_tokens: Number of tokens to generate.
        tok: Tokenizer.
        device: Device to run on (note: model's device is used by `generate`).
        temperature: Sampling temperature.
        top_p: Top-p sampling threshold. If None, this sampling method is disabled.
        attention_mask: Optional attention mask for input_ids.

    Returns:
        Generated text string.
    """
    # Get the prefix up to and including start_position
    if start_position >= input_ids.size(1):
        prefix_ids = input_ids
    else:
        prefix_ids = input_ids[:, :start_position + 1]

    prefix_len = prefix_ids.shape[1]

    # Slice the attention mask if provided
    prefix_attention_mask = None
    if attention_mask is not None:
        prefix_attention_mask = attention_mask[:, :prefix_len]
    elif tok.pad_token_id is not None:
        # Create mask from pad tokens if not provided
        prefix_attention_mask = (prefix_ids != tok.pad_token_id).long()

    with torch.no_grad():
        # Use the model's generate function for efficient, correct generation
        # with KV caching.
        generated_ids = orig_model.model.generate(
            input_ids=prefix_ids,
            attention_mask=prefix_attention_mask,
            max_new_tokens=num_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.eos_token_id,  # Use EOS token for padding
        )

    # Decode only the newly generated tokens
    newly_generated_ids = generated_ids[0, prefix_len:]
    return escape_newlines(tok.decode(newly_generated_ids, skip_special_tokens=True))


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
    """Prints a formatted table with row labels and data. Allows right-padded misalignment."""
    if not labels:
        return
    if len(labels) != len(data_rows):
        print("  (Error: Mismatch between number of labels and data rows for table printing)")
        return

    # Find the maximum number of columns in any row
    num_cols = 0
    for row_data in data_rows:
        if row_data:
            num_cols = max(num_cols, len(row_data))
    
    if num_cols == 0:  # All data rows are empty. Print labels only.
        max_label_width_only = max(len(s) for s in labels) + 2 if labels else 0
        for label in labels:
            print(f"{label:<{max_label_width_only}}")
        return

    # Compute column widths, allowing for ragged rows (right-padded)
    col_widths = [0] * num_cols
    for row_data in data_rows:
        for c_idx, cell in enumerate(row_data):
            col_widths[c_idx] = max(col_widths[c_idx], len(str(cell)))

    max_label_width = max(len(s) for s in labels) + 2
    for r_idx, label in enumerate(labels):
        row_str = f"{label:<{max_label_width}}"
        current_data_row = data_rows[r_idx]
        for c_idx in range(num_cols):
            if c_idx < len(current_data_row):
                cell_content = str(current_data_row[c_idx])
                row_str += f" {cell_content:<{col_widths[c_idx]}} |"
            else:
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
    analysis_metrics: Dict[str, Dict[str, Optional[float]]],
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
    sample_losses: Optional[Dict[str, float]] = None,
    kl_divergences: Optional[Dict[str, float]] = None,
    resample_ablation: bool = True, 
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
    
    # Print loss information if available
    if sample_losses is not None:
        print("Loss Components for this sample:")
        print(f"  Total Loss: {sample_losses['total']:.4f}")
        print(f"  - MSE Loss (A_i vs A_hat): {sample_losses['mse']:.4f} (weighted: {sample_losses['mse_weighted']:.4f})")
        print(f"  - LM Loss (KL[P_Orig||P_Dec]): {sample_losses['lm']:.4f} (weighted: {sample_losses['lm_weighted']:.4f})")
        print(f"  - KL Loss" + (" (A_hat+Δ)" if resample_ablation else "(A_hat)") + f": {sample_losses['kl']:.4f} (weighted: {sample_losses['kl_weighted']:.4f})")
        print(f"  - Entropy: {sample_losses['entropy']:.4f} (weighted: {sample_losses['entropy_weighted']:.4f})")
        print()

    print(f"Original Input Context (window around P={p}, showing positions {context_display_range}):")
    print_formatted_table(context_labels, context_data_rows)

    print(f"\nAnalysis for token following Position P={p} (i.e., predicting for seq position {p+1}):")
    for pred_type, top_n_tokens_list in analysis_predictions.items():
        tokens_str = ", ".join([f"'{t}'" for t in top_n_tokens_list])
        metrics_str_parts = []
        if pred_type in analysis_metrics:
            metrics = analysis_metrics[pred_type]
            mse_val = metrics.get("mse_vs_A")
            kl_A_val = metrics.get("kl_vs_A")
            kl_nat_val = metrics.get("kl_vs_natural")

            if mse_val is not None:
                metrics_str_parts.append(f"MSE(A_i,X): {mse_val:.4f}")
            if kl_A_val is not None:
                # Format with more precision for small KL values
                kl_A_fmt = f"{kl_A_val:.6f}" if abs(kl_A_val) < 0.001 and kl_A_val != 0 else f"{kl_A_val:.4f}"
                metrics_str_parts.append(f"KL(A||X): {kl_A_fmt}")
            if kl_nat_val is not None:
                kl_nat_fmt = f"{kl_nat_val:.6f}" if abs(kl_nat_val) < 0.001 and kl_nat_val != 0 else f"{kl_nat_val:.4f}"
                metrics_str_parts.append(f"KL(Nat||X): {kl_nat_fmt}")
        
        full_metrics_str = ""
        if metrics_str_parts:
            full_metrics_str = f" ({', '.join(metrics_str_parts)})"
            
        print(f"  - {pred_type} (Top {top_n_analysis_val}): {tokens_str}{full_metrics_str}")

    # Print KL divergences if available (original separate section)
    if kl_divergences is not None:
        print(f"\nKL Divergences (Grouped):")
        for group_name, group_values in kl_divergences.items():
            print(f"  {group_name}:")
            for kl_label, kl_value in group_values.items():
                if kl_value is None: # Handle cases where KL couldn't be computed
                    print(f"    - {kl_label}: N/A")
                    continue
                # Format with more precision to see small values
                if abs(kl_value) < 0.001 and kl_value != 0:
                    print(f"    - {kl_label}: {kl_value:.6f}")
                else:
                    print(f"    - {kl_label}: {kl_value:.4f}")
        
        if "From Natural Prediction (no intervention)" in kl_divergences:
            kl_a_natural_val = kl_divergences["From Natural Prediction (no intervention)"].get("KL(Natural || Natural)")
            if kl_a_natural_val is not None and kl_a_natural_val < 0.001: # Check if it's a small float
                 print("  Note: KL(Natural || Natural) ≈ 0 as expected (Natural is the original activation)")
        
        if sample_losses and "kl" in sample_losses:
            print(f"\n  ⚠️  Training KL loss for this sample" + (" (A_hat+Δ)" if resample_ablation else "(A_hat)") + f": {sample_losses['kl']:.4f}")
            if "From Original Activation A (training objective)" in kl_divergences:
                computed_kl_train = kl_divergences["From Original Activation A (training objective)"].get("KL(A || A_train) [TRAINING LOSS]")
                if computed_kl_train is not None and abs(computed_kl_train - sample_losses['kl']) > 0.01:
                    print(f"  ⚠️  WARNING: Computed training KL ({computed_kl_train:.4f}) doesn't match sample_losses KL ({sample_losses['kl']:.4f})!")
    add_current_token = cfg['trainable_components']['encoder']['add_current_token']
    print("\nGenerated Explanation (from Decoder using A_i):" + (" (with current token)" if add_current_token else ""))
    if decoder_tokens:
        dec_labels = ["Token:"] + [f"Dec Top {i+1}:" for i in range(len(decoder_preds_by_rank))]
        dec_data_rows = [decoder_tokens + ([f"[{original_token_at_p_str}]" if add_current_token else ""] if add_current_token else [])] + decoder_preds_by_rank
        print_formatted_table(dec_labels, dec_data_rows)
    else:
        print("  (No explanation generated or empty)")
    
    # ------------------------------------------------------------------
    # Base model generation comparison
    # ------------------------------------------------------------------

    print("\nGenerated Explanation (from BASE model using same context + still with soft tokens.):")
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


def compute_single_sample_losses(
    A_single: torch.Tensor,
    A_prime_single: Optional[torch.Tensor],
    input_ids_single: torch.Tensor,
    layer_idx: int,
    token_pos: int,
    models: Dict[str, torch.nn.Module],
    orig: OrigWrapper,
    sch_args: Dict[str, Any],
    device: torch.device,
    cached_prefix_ids: torch.Tensor | None = None,
    config: Dict[str, Any] = None,
    resample_ablation: bool = True,
    tokenizer = None,
) -> Dict[str, Any]:
    """Compute all loss components and relevant tensors for a single sample using train_step.
    
    Args:
        A_single: Single activation [1, hidden_size]
        A_prime_single: Alternative activation [1, hidden_size], or None.
        input_ids_single: Input IDs [seq_len]
        layer_idx: Layer index
        token_pos: Token position
        models: Dictionary with 'dec' and 'enc' models
        orig: Original model wrapper
        sch_args: Schedule arguments including 'tau', 'alpha', etc.
        device: Device to run on
        cached_prefix_ids: Pre-tokenized prefix for LM loss
        config: Configuration dictionary
        tokenizer: Tokenizer for natural prefix
        
    Returns:
        Dictionary with "losses" and "computed_values".
        "losses" contains various loss components including MSEs.
        "computed_values" contains tensors like generated text, A_hat, A_train, and logits.
    """
    from lens.training.loop import train_step
    
    # Ensure input_ids is 2D [1, seq_len]
    if input_ids_single.dim() == 1:
        input_ids_single = input_ids_single.unsqueeze(0)
    
    # Create a batch dictionary for train_step
    batch = {
        "A": A_single,
        "input_ids_A": input_ids_single,
        "layer_idx": torch.tensor([layer_idx], device=device),
        "token_pos_A": torch.tensor([token_pos], device=device),
    }
    
    # Add A_prime if available
    if A_prime_single is not None:
        batch["A_prime"] = A_prime_single
    else:
        # For verbose samples, we typically have A_prime, but handle the case where we don't
        # Use A itself as A_prime (no delta contribution)
        batch["A_prime"] = A_single.clone()
    
    # Prepare loss functions dict
    # _loss_fns = {
    #     "t_text": sch_args["t_text"],
    #     "tau": sch_args["tau"],
    #     "alpha": sch_args["alpha"],
    #     "kl_base_weight": sch_args["kl_base_weight"],
    #     "entropy_weight": sch_args["entropy_weight"],
    #     "mse_weight": sch_args["mse_weight"],
    #     "lm_base_weight": sch_args["lm_base_weight"],
    #     "GRPO_weight": sch_args["GRPO_weight"],
    #     "GRPO_beta": sch_args["GRPO_beta"],
    #     "group_n": sch_args["group_n"],
    # }
    _loss_fns = sch_args
    
    # Override t_text from config if available
    if config and "t_text" in config:
        _loss_fns["t_text"] = config["t_text"]
    
    # Call train_step with verbose_eval=True to get intermediate values
    with torch.no_grad():  # We don't need gradients for verbose samples
        losses = train_step(
            batch=batch,
            models=models,
            _loss_fns=_loss_fns,
            lm_loss_natural_prefix=config.get('lm_loss_natural_prefix') if config else None,
            tokenizer=tokenizer,
            cached_prefix_ids=cached_prefix_ids,
            resample_ablation=resample_ablation,
            eval_mode=False,  # Normal mode, not intervention mode
            verbose_eval=True,  # Get intermediate values
            GRPO_validate_mode=True,
        )
    
    # Extract intermediate values from verbose output
    verbose_intermediate = losses.get("verbose_intermediate", {})
    
    # Get the generation results and reconstructions from train_step
    gen_A_single = verbose_intermediate["gen"].detach().clone()
    A_hat_A_single = verbose_intermediate["A_hat"].detach().clone()
    A_train = verbose_intermediate["A_train"].detach().clone()
    logits_A_single_pos = verbose_intermediate["logits_orig"].detach().clone()
    logits_A_train_kl_pos = verbose_intermediate["logits_approx"].detach().clone()  
    del verbose_intermediate    
    
    # Compute additional MSEs that train_step doesn't provide
    mse_A_vs_zero = torch.nn.functional.mse_loss(A_single, torch.zeros_like(A_single)).item()
    mse_A_vs_Ahat = torch.nn.functional.mse_loss(A_single, A_hat_A_single.to(A_single.device)).item() if A_hat_A_single is not None else 0.0
    
    mse_A_vs_aprime = None
    if A_prime_single is not None and not torch.equal(A_prime_single, A_single):
        mse_A_vs_aprime = torch.nn.functional.mse_loss(A_single, A_prime_single).item()
    
    # Build loss dictionary matching the original format
    loss_dict = {
        "total": losses["total"].item() if torch.is_tensor(losses["total"]) else losses["total"],
        "mse": losses["mse"].item() if torch.is_tensor(losses["mse"]) else losses["mse"],
        "lm": losses["lm"].item() if torch.is_tensor(losses["lm"]) else losses["lm"],
        "kl": losses["kl"].item() if torch.is_tensor(losses["kl"]) else losses["kl"],
        "entropy": losses["entropy"].item() if torch.is_tensor(losses["entropy"]) else losses["entropy"],
        "mse_weighted": _loss_fns["mse_weight"] * (losses["mse"].item() if torch.is_tensor(losses["mse"]) else losses["mse"]),
        "lm_weighted": (_loss_fns["lm_base_weight"] * _loss_fns["alpha"]) * (losses["lm"].item() if torch.is_tensor(losses["lm"]) else losses["lm"]),
        "kl_weighted": _loss_fns["kl_base_weight"] * (losses["kl"].item() if torch.is_tensor(losses["kl"]) else losses["kl"]),
        "entropy_weighted": -_loss_fns["entropy_weight"] * (losses["entropy"].item() if torch.is_tensor(losses["entropy"]) else losses["entropy"]),
        "mse_A_vs_zero": mse_A_vs_zero,
        "mse_A_vs_aprime": mse_A_vs_aprime,
        "mse_A_vs_Ahat": mse_A_vs_Ahat,
        "mse_A_vs_A_train": losses["mse"].item() if torch.is_tensor(losses["mse"]) else losses["mse"],
    }
    del losses
    
    computed_values_dict = {
        "gen_A_single": gen_A_single,
        "A_hat_A_single": A_hat_A_single,
        "A_train_kl": A_train,
        "logits_A_single_pos": logits_A_single_pos,
        "logits_A_train_kl_pos": logits_A_train_kl_pos,
    }
    
    return {"losses": loss_dict, "computed_values": computed_values_dict}


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
    cached_prefix_ids: torch.Tensor | None = None,
    resample_ablation: bool = True,
    comparison_tuned_lens: Optional["TunedLens"] = None,
    random_from_batch: bool = True,
    do_soft_token_embeds: bool = True,
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
        cached_prefix_ids: Pre-tokenized cached prefix for LM loss
        resample_ablation: Whether to resample ablation
        comparison_tuned_lens: Optional comparison TunedLens
        
    Returns:
        If return_structured_data is False and capture_output is False: Number of samples printed from this batch
        If return_structured_data is True: Tuple of (num_printed, structured_samples_list)
        If capture_output is True: Tuple of (num_printed, captured_output_string)
    """
    with torch.no_grad():
        dec = models["dec"]
        enc = models["enc"]
        num_printed_this_batch = 0
        structured_samples = [] if return_structured_data else None
        captured_output = [] if capture_output else None
        if random_from_batch:
            num_samples = min(num_samples, batch["A"].size(0))
            sample_indices = torch.randperm(batch["A"].size(0), device=device)[:num_samples]
        else:
            sample_indices = range(min(num_samples - printed_count_so_far, batch["A"].size(0)))

        for i in sample_indices:
            l = int(batch["layer_idx"][i].item())
            p = int(batch["token_pos_A"][i].item())

            input_ids_seq = batch["input_ids_A"][i].unsqueeze(0).to(device)
            A_i = batch["A"][i : i + 1].to(device) 
            
            A_prime_i = None
            idx_for_aprime = -1
            if batch["A_prime"].size(0) > 0:
                idx_for_aprime = i % batch["A_prime"].size(0)
                A_prime_i = batch["A_prime"][idx_for_aprime : idx_for_aprime + 1].to(device)
            
            # Create models dict with orig included for train_step
            models_with_orig = {
                "dec": dec,
                "enc": enc,
                "orig": orig
            }
            
            results_from_compute = compute_single_sample_losses(
                A_single=A_i,
                A_prime_single=A_prime_i,
                input_ids_single=batch["input_ids_A"][i].to(device),
                layer_idx=l,
                token_pos=p,
                models=models_with_orig,
                orig=orig,
                sch_args=sch_args,
                device=device,
                cached_prefix_ids=cached_prefix_ids,
                config=cfg,
                resample_ablation=resample_ablation,
                tokenizer=tok,
            )
            sample_losses = results_from_compute["losses"]
            computed_tensors = results_from_compute["computed_values"]

            gen_single = computed_tensors["gen_A_single"].detach()
            A_hat_single = computed_tensors["A_hat_A_single"].detach()
            logits_orig_at_p_batched = computed_tensors["logits_A_single_pos"].detach() 
            A_train_i_for_kl = computed_tensors["A_train_kl"].detach()
            logits_train_at_p_batched = computed_tensors["logits_A_train_kl_pos"].detach()

            A_i_cast = A_i.to(orig.model.lm_head.weight.dtype)
            logit_lens_logits_from_A_i = orig.model.lm_head(A_i_cast)
            top_n_logit_lens_tokens = get_top_n_tokens(logit_lens_logits_from_A_i.squeeze(0), tok, top_n_analysis)
            del A_i_cast
            
            with torch.no_grad():
                logits_natural_all_pos = orig.model(input_ids=input_ids_seq).logits.detach().clone()
                logits_natural_at_p_batched = logits_natural_all_pos[:, p]

            preds_prefix_full_single_top = [
                escape_newlines(tok.decode([logits_natural_all_pos[0,t_idx].argmax().item()]))
                if t_idx < logits_natural_all_pos.size(1) else "N/A"
                for t_idx in range(input_ids_seq.size(-1))
            ]
            
            
            zero_activation = torch.zeros_like(A_i)
            logits_zero_at_p_batched = orig.forward_with_replacement(input_ids_seq, zero_activation, l, p).logits[:, p].detach().clone()
            top_n_zero_tokens = get_top_n_tokens(logits_zero_at_p_batched.squeeze(0), tok, top_n_analysis)
            
            
            top_n_orig_A_tokens = get_top_n_tokens(logits_orig_at_p_batched.squeeze(0), tok, top_n_analysis)
            
            top_n_aprime_tokens = ["N/A (A_prime not avail)"] * top_n_analysis
            logits_aprime_at_p_batched = None
            if A_prime_i is not None:
                logits_aprime_at_p_batched = orig.forward_with_replacement(input_ids_seq, A_prime_i, l, p).logits[:, p].detach().clone()
                top_n_aprime_tokens = get_top_n_tokens(logits_aprime_at_p_batched.squeeze(0), tok, top_n_analysis)
            
            logits_approx_at_p_batched = orig.forward_with_replacement(input_ids_seq, A_hat_single.to(device), l, p).logits[:, p].detach().clone()
            top_n_lens_recon_tokens = get_top_n_tokens(logits_approx_at_p_batched.squeeze(0), tok, top_n_analysis)

            top_n_train_ablation_tokens = ["N/A (A_train_kl not avail)"] * top_n_analysis
            if A_train_i_for_kl is not None and logits_train_at_p_batched is not None:
                top_n_train_ablation_tokens = get_top_n_tokens(logits_train_at_p_batched.squeeze(0), tok, top_n_analysis)
            
            raw_prefix_ids = input_ids_seq[0].tolist()
            
            top_n_natural_base_preds_for_p_plus_1: List[str]
            natural_base_logits_at_p = logits_natural_at_p_batched[0]
            top_n_natural_base_preds_for_p_plus_1 = get_top_n_tokens(natural_base_logits_at_p, tok, top_n_analysis)

            # Compute baseline using previous t_text tokens as encoder input
            t_text = cfg.get("t_text", 8)
            # Get the previous t_text tokens from position p
            start_token_idx = max(0, p - t_text + 1)  # Include token at position p
            end_token_idx = p + 1
            
            # Extract previous tokens (may be less than t_text if near beginning)
            prev_token_ids = raw_prefix_ids[start_token_idx:end_token_idx]
            
            # Pad with BOS token if needed
            if len(prev_token_ids) < t_text:
                bos_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
                padding_needed = t_text - len(prev_token_ids)
                prev_token_ids = [bos_id] * padding_needed + prev_token_ids
            
            # Convert token IDs to embeddings
            prev_token_ids_tensor = torch.tensor(prev_token_ids, device=device).unsqueeze(0)  # [1, t_text]
            
            # Get embeddings from the base model
            if hasattr(enc, 'base') and enc._use_base:
                emb_table = enc.base.get_input_embeddings().weight
            else:
                # Use original model's embeddings
                emb_table = orig.model.get_input_embeddings().weight
            
            prev_token_embeddings = emb_table[prev_token_ids_tensor]  # [1, t_text, d_model]
            
            # Pass through encoder to get A_hat_baseline
            A_hat_baseline = enc(prev_token_embeddings, current_token_ids=batch["input_ids_A"][i][p].to(device).unsqueeze(0) if enc.config.add_current_token else None).detach().clone()
            
            # Compute predictions and metrics for baseline
            logits_baseline_at_p_batched = orig.forward_with_replacement(input_ids_seq, A_hat_baseline.to(device), l, p).logits[:, p].detach().clone()
            top_n_baseline_tokens = get_top_n_tokens(logits_baseline_at_p_batched.squeeze(0), tok, top_n_analysis)
            
            # Compute MSE for baseline
            mse_baseline_vs_A = torch.nn.functional.mse_loss(A_i, A_hat_baseline.to(A_i.device)).item()
            
            # Clean up baseline tensors
            del prev_token_embeddings, A_hat_baseline
            
            # Compute shuffle intervention: shuffle first (n-3) tokens of decoder output
            t_text = cfg.get("t_text", 8)
            shuffle_count = max(0, t_text - 3)
            shuffled_embeds = gen_single.generated_text_embeddings.detach().clone().to(device)
            
            if shuffle_count > 0:
                # Shuffle the first shuffle_count tokens - indices on same device as embeddings
                indices = torch.randperm(shuffle_count, device=shuffled_embeds.device)
                shuffled_embeds[0, :shuffle_count] = shuffled_embeds[0, indices]
            
            # Pass shuffled embeddings through encoder
            A_hat_shuffled = enc(shuffled_embeds, current_token_ids=batch["input_ids_A"][i][p].to(device).unsqueeze(0) if enc.config.add_current_token else None).detach().clone()
            
            # Compute predictions for shuffled reconstruction
            logits_shuffled_all_pos = orig.forward_with_replacement(input_ids_seq, A_hat_shuffled.to(device), l, p).logits.detach().clone()
            logits_shuffled_at_p_batched = logits_shuffled_all_pos[:, p]
            top_n_shuffled_tokens = get_top_n_tokens(logits_shuffled_at_p_batched.squeeze(0), tok, top_n_analysis)
            
            # Compute MSE for shuffled
            mse_shuffled_vs_A = torch.nn.functional.mse_loss(A_i, A_hat_shuffled.to(A_i.device)).item()
            
            # Clean up shuffle tensors
            del shuffled_embeds, A_hat_shuffled
            
            # Compute full shuffle intervention: shuffle ALL tokens of decoder output
            full_shuffled_embeds = gen_single.generated_text_embeddings.clone().to(device)
            indices_full = torch.randperm(t_text, device=full_shuffled_embeds.device)
            full_shuffled_embeds[0] = full_shuffled_embeds[0, indices_full]
            
            # Pass full shuffled embeddings through encoder
            A_hat_full_shuffled = enc(full_shuffled_embeds, current_token_ids=batch["input_ids_A"][i][p].to(device).unsqueeze(0) if enc.config.add_current_token else None).detach().clone()
            
            # Compute predictions for full shuffled reconstruction
            logits_full_shuffled_all_pos = orig.forward_with_replacement(input_ids_seq, A_hat_full_shuffled.to(device), l, p).logits.detach().clone()   
            logits_full_shuffled_at_p_batched = logits_full_shuffled_all_pos[:, p]
            top_n_full_shuffled_tokens = get_top_n_tokens(logits_full_shuffled_at_p_batched.squeeze(0), tok, top_n_analysis)
            
            # Compute MSE for full shuffled
            mse_full_shuffled_vs_A = torch.nn.functional.mse_loss(A_i, A_hat_full_shuffled.to(A_i.device)).item()
            
            # Clean up full shuffle tensors
            del full_shuffled_embeds, A_hat_full_shuffled

            # Clean up full logits tensors that are no longer needed
            del logits_shuffled_all_pos, logits_full_shuffled_all_pos

            # TunedLens predictions
            top_n_tuned_lens_tokens = ["N/A"] * top_n_analysis
            logits_tuned_lens_at_p_batched = None
            tuned_lens_prediction_key = f"TunedLens (L{l})"

            if comparison_tuned_lens is not None:
                # comparison_tuned_lens is already on the correct device (moved by the caller)
                # A_i is also on the correct device.
                if hasattr(comparison_tuned_lens, 'config') and hasattr(comparison_tuned_lens.config, 'num_hidden_layers') and l < comparison_tuned_lens.config.num_hidden_layers:
                    try:
                        with torch.no_grad():
                            # Ensure A_i has the dtype TunedLens expects if there's a specific one
                            # Always cast A_i to float32 for TunedLens to avoid dtype mismatch
                            A_i_casted = A_i.to(torch.float32)
                            logits_tuned_lens_at_p_batched = comparison_tuned_lens(A_i_casted, idx=l).detach().clone()

                        top_n_tuned_lens_tokens = get_top_n_tokens(logits_tuned_lens_at_p_batched.squeeze(0), tok, top_n_analysis)
                        del A_i_casted
                    except Exception as e:
                        # import logging
                        # logging.getLogger(__name__).error(f"Error during TunedLens prediction for layer {l}: {e}", exc_info=True)
                        #log.error(f"Error during TunedLens prediction for layer {l}: {e}", exc_info=True)
                        top_n_tuned_lens_tokens = [f"ERR TL L{l}"] * top_n_analysis 
                        print(f"error during tuned lens prediction for layer {l}: {e}")
                else:
                     top_n_tuned_lens_tokens = [f"L{l} OOB TL"] * top_n_analysis

            analysis_preds_dict = {
                "Base Model's natural prediction": top_n_natural_base_preds_for_p_plus_1,
                "Zero Vector Baseline": top_n_zero_tokens,
                "Prev Tokens Baseline (Enc[prev t tokens])": top_n_baseline_tokens,
                "Shuffled Decoder Output (first n-3)": top_n_shuffled_tokens,
                "Shuffled Decoder Output (ALL tokens)": top_n_full_shuffled_tokens,
                "Logit Lens (from A_i)": top_n_logit_lens_tokens,
                "Base Model (orig A)": top_n_orig_A_tokens,
                "Base Model (A')": top_n_aprime_tokens,
                tuned_lens_prediction_key: top_n_tuned_lens_tokens,
                "Log w/o Ablation (A_hat)" + ("train" if resample_ablation else ""): top_n_lens_recon_tokens,
                "Log training " + ("(A_hat+Δ)" if resample_ablation else "(A_hat)"): top_n_train_ablation_tokens,
            }

            a_prime_string_cropped = None
            if A_prime_i is not None and "input_ids_A_prime" in batch and "token_pos_A_prime" in batch and \
               idx_for_aprime < batch["input_ids_A_prime"].size(0) and idx_for_aprime < batch["token_pos_A_prime"].size(0) :
                input_ids_A_prime_seq = batch["input_ids_A_prime"][idx_for_aprime].to(device)
                p_prime = int(batch["token_pos_A_prime"][idx_for_aprime].item())
                raw_prime_ids = input_ids_A_prime_seq.tolist()
                crop_start_idx_prime = max(0, p_prime - 100)
                crop_end_idx_prime = min(len(raw_prime_ids), p_prime + 30 + 1)
                before_p_prime = tok.decode(raw_prime_ids[crop_start_idx_prime:p_prime]) if p_prime > crop_start_idx_prime else ""
                token_at_p_prime = tok.decode([raw_prime_ids[p_prime]]) if p_prime < len(raw_prime_ids) else ""
                after_p_prime = tok.decode(raw_prime_ids[p_prime+1:crop_end_idx_prime]) if p_prime+1 < crop_end_idx_prime else ""
                a_prime_string_cropped = escape_newlines(before_p_prime + "*" + token_at_p_prime + "*" + after_p_prime)
                if crop_start_idx_prime > 0: a_prime_string_cropped = "..." + a_prime_string_cropped
                if crop_end_idx_prime < len(raw_prime_ids): a_prime_string_cropped = a_prime_string_cropped + "..."
            
            display_start_idx = max(0, p - 10)
            display_end_idx = min(len(raw_prefix_ids), p + 3 + 1)
            displayed_raw_ids = raw_prefix_ids[display_start_idx:display_end_idx]
            displayed_prefix_tokens = [escape_newlines(tok.decode([tid])) for tid in displayed_raw_ids]
            displayed_positions = [str(k) for k in range(display_start_idx, display_end_idx)]

            crop_start_idx = max(0, p - 100)
            crop_end_idx = min(len(raw_prefix_ids), p + 30 + 1)
            before_p = tok.decode(raw_prefix_ids[crop_start_idx:p]) if p > crop_start_idx else ""
            token_at_p_val = tok.decode([raw_prefix_ids[p]]) if p < len(raw_prefix_ids) else ""
            after_p = tok.decode(raw_prefix_ids[p+1:crop_end_idx]) if p+1 < crop_end_idx else ""
            original_string_cropped = escape_newlines(before_p + "*" + token_at_p_val + "*" + after_p)
            if crop_start_idx > 0: original_string_cropped = "..." + original_string_cropped
            if crop_end_idx < len(raw_prefix_ids): original_string_cropped = original_string_cropped + "..."
            original_token_at_p_str = escape_newlines(tok.decode([raw_prefix_ids[p]])) if p < len(raw_prefix_ids) else "N/A"

            gen_token_ids_full = gen_single.hard_token_ids[0].cpu().tolist()
            gen_tokens = [escape_newlines(tok.decode([tid])) for tid in gen_token_ids_full]
            k_for_decoder_preds = min(top_n_analysis, gen_single.raw_lm_logits.size(-1))
            decoder_preds_by_rank = build_topk_preds_by_rank(
                gen_single.raw_lm_logits.cpu(), len(gen_tokens), k_for_decoder_preds, tok
            )

            base_gen_single = dec.generate_soft_kv_cached_nondiff(A_i, max_length=cfg["t_text"], gumbel_tau=sch_args["tau"], override_model_base_and_out=orig, use_projection=True,return_logits=True ).detach().clone()
            base_token_ids_full = base_gen_single.hard_token_ids[0].cpu().tolist()
            base_gen_tokens = [escape_newlines(tok.decode([tid])) for tid in base_token_ids_full]
            base_preds_by_rank = build_topk_preds_by_rank(
                base_gen_single.raw_lm_logits.cpu(), len(base_gen_tokens), k_for_decoder_preds, tok
            )
            # Clean up generation results
            del base_gen_single

            left_ids, right_ids, _, _ = dec.tokenize_and_embed_prompt(cfg["decoder_prompt"], tok)
            base_gen_single_hard = dec.generate_soft_kv_cached_nondiff(A_i, max_length=cfg["t_text"], gumbel_tau=sch_args["tau"], override_model_base_and_out=orig, hard_left_emb=left_ids, hard_right_emb=right_ids, use_projection=True,return_logits=True).detach().clone()
            base_token_ids_full_hard = base_gen_single_hard.hard_token_ids[0].cpu().tolist()
            base_gen_tokens_hard = [escape_newlines(tok.decode([tid])) for tid in base_token_ids_full_hard]
            base_preds_by_rank_hard = build_topk_preds_by_rank(
                base_gen_single_hard.raw_lm_logits.cpu(), len(base_gen_tokens_hard), k_for_decoder_preds, tok
            )
            # Clean up generation results
            del base_gen_single_hard

            base_gen_single_hard_no_map = dec.generate_soft_kv_cached_nondiff(A_i, max_length=cfg["t_text"], gumbel_tau=sch_args["tau"], override_model_base_and_out=orig, hard_left_emb=left_ids, hard_right_emb=right_ids, use_projection=False,return_logits=True).detach().clone()
            base_token_ids_full_hard_no_map = base_gen_single_hard_no_map.hard_token_ids[0].cpu().tolist()
            base_gen_tokens_hard_no_map = [escape_newlines(tok.decode([tid])) for tid in base_token_ids_full_hard_no_map]
            base_preds_by_rank_hard_no_map = build_topk_preds_by_rank(
                base_gen_single_hard_no_map.raw_lm_logits.cpu(), len(base_gen_tokens_hard_no_map), k_for_decoder_preds, tok
            )
            # Clean up generation results
            del base_gen_single_hard_no_map
            
            context_display_range = f"{display_start_idx}-{display_end_idx-1}" if displayed_positions else "empty"
            context_labels = ["Position:", "Token:", "BaseLM (shift):"]
            
            shifted_preds_for_display = [
                preds_prefix_full_single_top[display_start_idx + k_rel - 1] 
                if display_start_idx + k_rel > 0 and display_start_idx + k_rel -1 < len(preds_prefix_full_single_top) 
                else ("" if display_start_idx + k_rel == 0 else "ERR_IDX")
                for k_rel in range(len(displayed_prefix_tokens))
            ]
            context_data_rows = [list(displayed_positions), list(displayed_prefix_tokens), list(shifted_preds_for_display)]
            del preds_prefix_full_single_top  # No longer needed
            relative_p = p - display_start_idx
            if 0 <= relative_p < len(displayed_prefix_tokens):
                context_data_rows[0][relative_p] = f"[{context_data_rows[0][relative_p]}]P"
                context_data_rows[1][relative_p] = f"*{context_data_rows[1][relative_p]}*"

            def safe_kl(l1, l2):
                if l1 is None or l2 is None: return None
                return compute_kl_divergence_robust(l1, l2).item()

            kl_zero_from_natural = safe_kl(logits_zero_at_p_batched, logits_natural_at_p_batched)
            kl_orig_from_natural = safe_kl(logits_orig_at_p_batched, logits_natural_at_p_batched)
            kl_aprime_from_natural = safe_kl(logits_aprime_at_p_batched, logits_natural_at_p_batched)
            kl_ahat_from_natural = safe_kl(logits_approx_at_p_batched, logits_natural_at_p_batched)
            kl_ahat_delta_from_natural = safe_kl(logits_train_at_p_batched, logits_natural_at_p_batched)
            kl_baseline_from_natural = safe_kl(logits_baseline_at_p_batched, logits_natural_at_p_batched)
            kl_shuffled_from_natural = safe_kl(logits_shuffled_at_p_batched, logits_natural_at_p_batched)
            kl_full_shuffled_from_natural = safe_kl(logits_full_shuffled_at_p_batched, logits_natural_at_p_batched)
            kl_tuned_lens_from_natural = safe_kl(logits_tuned_lens_at_p_batched, logits_natural_at_p_batched)

            kl_zero_from_orig = safe_kl(logits_zero_at_p_batched, logits_orig_at_p_batched)
            kl_aprime_from_orig = safe_kl(logits_aprime_at_p_batched, logits_orig_at_p_batched)
            kl_ahat_from_orig = safe_kl(logits_approx_at_p_batched, logits_orig_at_p_batched)
            kl_baseline_from_orig = safe_kl(logits_baseline_at_p_batched, logits_orig_at_p_batched)
            kl_shuffled_from_orig = safe_kl(logits_shuffled_at_p_batched, logits_orig_at_p_batched)
            kl_full_shuffled_from_orig = safe_kl(logits_full_shuffled_at_p_batched, logits_orig_at_p_batched)
            training_kl_loss_value = sample_losses['kl'] 
            kl_tuned_lens_from_orig = safe_kl(logits_tuned_lens_at_p_batched, logits_orig_at_p_batched)

            kl_divergences_for_print_section = {
                "From Natural Prediction (no intervention)": {
                    "KL(Natural || Zero)": kl_zero_from_natural,
                    "KL(Natural || Prev Tokens Baseline)": kl_baseline_from_natural,
                    "KL(Natural || Shuffled Decoder - first n-3)": kl_shuffled_from_natural,
                    "KL(Natural || Shuffled Decoder - ALL tokens)": kl_full_shuffled_from_natural,
                    "KL(Natural || A)": kl_orig_from_natural, 
                    "KL(Natural || A')": kl_aprime_from_natural,
                    "KL(Natural || A_hat)": kl_ahat_from_natural,
                    "KL(Natural || A_hat+Δ)": kl_ahat_delta_from_natural,
                    f"KL(Natural || {tuned_lens_prediction_key})": kl_tuned_lens_from_natural,
                },
                "From Original Activation A (training objective)": {
                    "KL(A || Zero)": kl_zero_from_orig,
                    "KL(A || Prev Tokens Baseline)": kl_baseline_from_orig,
                    "KL(A || Shuffled Decoder - first n-3)": kl_shuffled_from_orig,
                    "KL(A || Shuffled Decoder - ALL tokens)": kl_full_shuffled_from_orig,
                    "KL(A || A')": kl_aprime_from_orig,
                    "KL(A || A_hat)": kl_ahat_from_orig,
                    ("KL(A || A_hat+Δ) [TRAINING LOSS]" if resample_ablation else "KL(A || A_hat) [TRAINING LOSS]"): training_kl_loss_value,
                    f"KL(A || {tuned_lens_prediction_key})": kl_tuned_lens_from_orig,
                }
            }
            
            analysis_metrics_for_print = {
                "Base Model's natural prediction": { 
                    "mse_vs_A": 0.0, 
                    "kl_vs_A": safe_kl(logits_natural_at_p_batched, logits_orig_at_p_batched), 
                    "kl_vs_natural": 0.0,
                },
                "Zero Vector Baseline": {
                    "mse_vs_A": sample_losses["mse_A_vs_zero"],
                    "kl_vs_A": kl_zero_from_orig,
                    "kl_vs_natural": kl_zero_from_natural,
                },
                "Prev Tokens Baseline (Enc[prev t tokens])": {
                    "mse_vs_A": mse_baseline_vs_A,
                    "kl_vs_A": kl_baseline_from_orig,
                    "kl_vs_natural": kl_baseline_from_natural,
                },
                "Shuffled Decoder Output (first n-3)": {
                    "mse_vs_A": mse_shuffled_vs_A,
                    "kl_vs_A": kl_shuffled_from_orig,
                    "kl_vs_natural": kl_shuffled_from_natural,
                },
                "Shuffled Decoder Output (ALL tokens)": {
                    "mse_vs_A": mse_full_shuffled_vs_A,
                    "kl_vs_A": kl_full_shuffled_from_orig,
                    "kl_vs_natural": kl_full_shuffled_from_natural,
                },
                "Base Model (orig A)": { 
                    "mse_vs_A": 0.0, 
                    "kl_vs_A": 0.0, 
                    "kl_vs_natural": kl_orig_from_natural,
                },
                "Base Model (A')": { 
                    "mse_vs_A": sample_losses["mse_A_vs_aprime"], 
                    "kl_vs_A": kl_aprime_from_orig, 
                    "kl_vs_natural": kl_aprime_from_natural, 
                },
                "Log w/o Ablation (A_hat)": { 
                    "mse_vs_A": sample_losses["mse_A_vs_Ahat"], 
                    "kl_vs_A": kl_ahat_from_orig,
                    "kl_vs_natural": kl_ahat_from_natural,
                },
                "Log w/Resample Ablation (A_hat+Δ)": { 
                    "mse_vs_A": sample_losses["mse_A_vs_A_train"], 
                    "kl_vs_A": training_kl_loss_value, 
                    "kl_vs_natural": kl_ahat_delta_from_natural, 
                },
                tuned_lens_prediction_key: {
                    "mse_vs_A": None,
                    "kl_vs_A": kl_tuned_lens_from_orig,
                    "kl_vs_natural": kl_tuned_lens_from_natural,
                },
            }
            
            autoregressive_continuation = None
            if generate_continuation:
                autoregressive_continuation = generate_autoregressive_continuation(
                    orig, input_ids_seq, p, num_tokens=continuation_tokens, tok=tok, device=device
                )
            
            if return_structured_data:
                # Simplified structured data based on available variables for print
                # If original (token, prob) pairs are needed, logic from earlier versions needs to be re-added
                original_logits_topk_for_structured = []
                if logits_orig_at_p_batched is not None:
                    orig_probs = torch.softmax(logits_orig_at_p_batched.squeeze(0), dim=-1)
                    orig_top_values, orig_top_indices = torch.topk(orig_probs, k=min(top_n_analysis, orig_probs.size(-1)))
                    for val_idx in range(len(orig_top_indices)):
                        token_id = orig_top_indices[val_idx].item()
                        prob = orig_top_values[val_idx].item()
                        token_str = escape_newlines(tok.decode([token_id]))
                        original_logits_topk_for_structured.append((token_str, prob))

                reconstructed_logits_topk_for_structured = [] # For A_hat
                if logits_approx_at_p_batched is not None: # Logits from A_hat intervention
                    recon_probs = torch.softmax(logits_approx_at_p_batched.squeeze(0), dim=-1)
                    recon_top_values, recon_top_indices = torch.topk(recon_probs, k=min(top_n_analysis, recon_probs.size(-1)))
                    for val_idx in range(len(recon_top_indices)):
                        token_id = recon_top_indices[val_idx].item()
                        prob = recon_top_values[val_idx].item()
                        token_str = escape_newlines(tok.decode([token_id]))
                        reconstructed_logits_topk_for_structured.append((token_str, prob))
                
                # Decoder predictions (top-k for each generated token position)
                # For structured output, let's take the top-1 prediction for each token from the decoder
                # and its associated top-k probabilities for the first position if needed.
                # For simplicity, using the already built decoder_preds_by_rank (list of lists of top tokens)
                # and gen_tokens (list of generated tokens)
                
                decoder_top_preds_structured = [] # List of (token_str, prob) for the *first* generated position by decoder
                if gen_single.raw_lm_logits.numel() > 0 and gen_single.raw_lm_logits.size(1) > 0 : # Check if any logits generated
                    first_pos_logits = gen_single.raw_lm_logits[0, 0]
                    first_pos_probs = torch.softmax(first_pos_logits, dim=-1)
                    top_vals, top_ids = torch.topk(first_pos_probs, k=min(top_n_analysis, first_pos_probs.size(-1)))
                    for k_idx in range(len(top_ids)):
                        decoder_top_preds_structured.append(
                            (escape_newlines(tok.decode([top_ids[k_idx].item()])), top_vals[k_idx].item())
                        )

                sample_data = {
                    "input_text": escape_newlines(original_string_cropped),
                    "chosen_token": escape_newlines(original_token_at_p_str),
                    "position": p,
                    "decoded_text": escape_newlines(" ".join(gen_tokens)), # Decoder generated text
                    "decoder_top_predictions_first_pos": decoder_top_preds_structured, # Top-k for first token by decoder
                    "continuation": escape_newlines(autoregressive_continuation) if autoregressive_continuation else None,
                    "original_model_top_predictions_at_p": original_logits_topk_for_structured, # Top-k from A_i intervention
                    "reconstructed_A_hat_top_predictions_at_p": reconstructed_logits_topk_for_structured, # Top-k from A_hat intervention
                    "metrics_at_p": analysis_metrics_for_print, # Contains MSEs and KLs for various conditions
                    "all_losses": sample_losses, # Full loss dictionary
                }
                sample_data = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in sample_data.items()}
                structured_samples.append(sample_data)
            
            if capture_output:
                import io
                from contextlib import redirect_stdout
                output_buffer = io.StringIO()
                with redirect_stdout(output_buffer):
                    print_verbose_sample_details(
                        l=l, p=p, original_token_at_p_str=original_token_at_p_str,
                        context_display_range=context_display_range, context_labels=context_labels, context_data_rows=context_data_rows,
                        analysis_predictions=analysis_preds_dict,
                        analysis_metrics=analysis_metrics_for_print, 
                        decoder_tokens=gen_tokens, decoder_preds_by_rank=decoder_preds_by_rank,
                        base_tokens=base_gen_tokens, base_tokens_hard=base_gen_tokens_hard, base_tokens_hard_no_map=base_gen_tokens_hard_no_map,
                        base_preds_by_rank=base_preds_by_rank, base_preds_by_rank_hard=base_preds_by_rank_hard, base_preds_by_rank_hard_no_map=base_preds_by_rank_hard_no_map,
                        top_n_analysis_val=top_n_analysis, original_string_cropped=original_string_cropped,
                        autoregressive_continuation=autoregressive_continuation, a_prime_string_cropped=a_prime_string_cropped,
                        cfg=cfg, sample_losses=sample_losses, kl_divergences=kl_divergences_for_print_section, 
                        resample_ablation=resample_ablation
                    )
                captured_output.append(output_buffer.getvalue())
                print(captured_output[-1], end='')
            else:
                print_verbose_sample_details(
                    l=l, p=p, original_token_at_p_str=original_token_at_p_str,
                    context_display_range=context_display_range, context_labels=context_labels, context_data_rows=context_data_rows,
                    analysis_predictions=analysis_preds_dict,
                    analysis_metrics=analysis_metrics_for_print, 
                    decoder_tokens=gen_tokens, decoder_preds_by_rank=decoder_preds_by_rank,
                    base_tokens=base_gen_tokens, base_tokens_hard=base_gen_tokens_hard, base_tokens_hard_no_map=base_gen_tokens_hard_no_map,
                    base_preds_by_rank=base_preds_by_rank, base_preds_by_rank_hard=base_preds_by_rank_hard, base_preds_by_rank_hard_no_map=base_preds_by_rank_hard_no_map,
                    top_n_analysis_val=top_n_analysis, original_string_cropped=original_string_cropped,
                    autoregressive_continuation=autoregressive_continuation, a_prime_string_cropped=a_prime_string_cropped,
                    cfg=cfg, sample_losses=sample_losses, kl_divergences=kl_divergences_for_print_section,
                    resample_ablation=resample_ablation
                )
            num_printed_this_batch += 1
            
            # Force cleanup after each sample to prevent accumulation
            del (
                A_i, input_ids_seq, gen_single, A_hat_single, A_train_i_for_kl,
                logits_orig_at_p_batched, logits_train_at_p_batched,
                logits_natural_at_p_batched, logits_zero_at_p_batched,
                logits_approx_at_p_batched, logits_baseline_at_p_batched,
                logits_shuffled_at_p_batched, logits_full_shuffled_at_p_batched,
                logit_lens_logits_from_A_i, zero_activation, prev_token_ids_tensor,
                base_gen_tokens, base_gen_tokens_hard, base_gen_tokens_hard_no_map,
                results_from_compute, computed_tensors,
                analysis_preds_dict, analysis_metrics_for_print, sample_losses, kl_divergences_for_print_section,
                gen_tokens, decoder_preds_by_rank, base_preds_by_rank, base_preds_by_rank_hard, base_preds_by_rank_hard_no_map,
                context_data_rows,
            )
            if A_prime_i is not None:
                del A_prime_i
            if 'natural_base_logits_at_p' in locals():
                del natural_base_logits_at_p
            if logits_aprime_at_p_batched is not None:
                del logits_aprime_at_p_batched
            if logits_tuned_lens_at_p_batched is not None:
                del logits_tuned_lens_at_p_batched
            if autoregressive_continuation is not None:
                del autoregressive_continuation
            if a_prime_string_cropped is not None:
                del a_prime_string_cropped
            
            # Force GPU memory cleanup after each sample
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if (printed_count_so_far + num_printed_this_batch) >= num_samples:
                break
    
        if num_printed_this_batch > 0 and do_soft_token_embeds:
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
            input_text = ''.join([t for t, _ in left_input])
            output_text = ''.join([t for t, _ in left_output]) if left_output else ""
            
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
            input_text = ''.join([t for t, _ in right_input])
            output_text = ''.join([t for t, _ in right_output]) if right_output else ""
            
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
        prompt_text = ''.join([t for t, _ in enc_proj])
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
