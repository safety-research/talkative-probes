"""Evaluate a saved checkpoint on a held-out activation set."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split

from lens.data.collate import collate
from lens.data.dataset import ActivationDataset
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.orig import OrigWrapper
from lens.training.loop import train_step
from lens.utils import checkpoint as ckpt_util
from transformers import AutoTokenizer
from tqdm import tqdm
# Helper to escape newlines for display
def escape_newlines(text: str) -> str:
    return text.replace("\n", "\\n")


from typing import Any, Dict, List, Tuple, Union

from transformers import PreTrainedTokenizerBase
from lens.utils.embedding_remap import remap_embeddings


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


def _parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a Consistency-Lens checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument(
        "--config_path", type=str, default="consistency-lens/config/lens_simple.yaml", help="Path to the configuration YAML file."
    )
    parser.add_argument("--activation_dir", type=str, help="Override activation_dir from config.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation.")
    parser.add_argument("--num_batches", type=int, default=25, help="Number of batches to evaluate.")
    parser.add_argument("--verbose_samples", type=int, default=3, help="Number of verbose samples to print from the start of evaluation.")
    parser.add_argument("--top_n_analysis", type=int, default=3, help="Number of top-N predictions to show in verbose analysis.")
    parser.add_argument("--val_fraction", type=float, help="Fraction of dataset for validation (overrides config).")
    parser.add_argument("--split_seed", type=int, help="Seed for train/val split (overrides config).")
    return parser.parse_args()


def _setup_logging() -> logging.Logger:
    """Sets up basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def _load_config(config_path: str) -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _build_models(
    cfg: Dict[str, Any], device: torch.device
) -> Tuple[Dict[str, torch.nn.Module], PreTrainedTokenizerBase, OrigWrapper]:
    """Builds and initializes the models (Decoder, Encoder, Original Model Wrapper, Tokenizer)."""
    model_name = cfg["model_name"]
    
    # Decoder
    dec_cfg = DecoderConfig(model_name=model_name, n_prompt_tokens=cfg["decoder_n_prompt_tokens"])
    dec = Decoder(dec_cfg)
    tokenizer_name = cfg.get("tokenizer_name", model_name)
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    logging.getLogger(__name__).info(f"Tokenizer name: {tokenizer_name}")

    # Encoder (created before remap)
    enc_cfg = EncoderConfig(model_name=model_name)
    enc = Encoder(enc_cfg)

    # Original Model Wrapper
    orig = OrigWrapper(model_name, load_in_8bit=False)
    orig.model.to(device)

    new_vocab_size = tok.vocab_size
    # Ensure Decoder prompt embedding tokens updated for resized vocab
    dec.set_prompt(cfg.get("decoder_prompt", "Explain: "), tok)

    from lens.utils.embedding_remap import remap_embeddings
    base_tok = AutoTokenizer.from_pretrained(model_name)

    remap_embeddings(dec.base, base_tok, tok)
    remap_embeddings(enc.base, base_tok, tok)
    remap_embeddings(orig.model, base_tok, tok)
    logging.getLogger(__name__).info("Remapped all model embeddings to new tokenizer")

    # Refresh decoder prompt ids for new vocab
    dec.set_prompt(cfg.get("decoder_prompt", "Explain: "), tok)

    # Ensure Decoder's independent LM head matches new vocab
    if dec.out.weight.size(0) != new_vocab_size:
        import torch.nn as nn
        d_model = dec.base.config.hidden_size
        dec.out = nn.Linear(d_model, new_vocab_size, bias=False)
        with torch.no_grad():
            dec.out.weight.copy_(dec.base.get_output_embeddings().weight)
        logging.getLogger(__name__).info("Resized dec.out to new vocab size")

    models_dict = {"dec": dec.to(device), "enc": enc.to(device)}
    return models_dict, tok, orig


def _load_checkpoint(checkpoint_path: str, models: Dict[str, torch.nn.Module], device: torch.device) -> None:
    """Loads model weights from a checkpoint."""
    # Dummy optimizer for checkpoint loading (if optimizer state is saved)
    dec = models["dec"]
    enc = models["enc"]
    # Ensure all parameters are on the same device before creating optimizer
    params_on_device = list(p for p in dec.parameters() if p.device == device) + \
                       list(p for p in enc.parameters() if p.device == device)
    
    # If some params were not moved to device, this might be an issue.
    # However, _build_models should have moved them.
    all_params = list(dec.parameters()) + list(enc.parameters())
    if len(params_on_device) != len(all_params):
        logging.warning("Not all model parameters are on the target device for optimizer creation.")

    opt = torch.optim.AdamW(params=all_params, lr=1e-4) # Use all_params, assuming they are correctly on device
    
    ckpt_util.load(checkpoint_path, models=models, optim=opt, map_location=device)
    logging.info(f"Loaded checkpoint from {checkpoint_path}")


def _prepare_data(
    cfg: Dict[str, Any], args: argparse.Namespace, effective_act_dir: str, log: logging.Logger
) -> DataLoader:
    """Loads and prepares the dataset and DataLoader."""
    log.info(f"Loading activations from {effective_act_dir}")
    full_ds = ActivationDataset(effective_act_dir)

    val_fraction_cfg = cfg.get('val_fraction', 0.1)
    vf = args.val_fraction if args.val_fraction is not None else val_fraction_cfg
    vf = max(0.0, min(1.0, vf))

    split_seed_cfg = cfg.get('split_seed', 42)
    split_seed = args.split_seed if args.split_seed is not None else split_seed_cfg

    if 0 < vf < 1.0:
        vsz = int(len(full_ds) * vf)
        tsz = len(full_ds) - vsz
        if tsz == 0 or vsz == 0: # Avoid empty splits if possible
            log.warning(f"Validation split resulted in an empty train or validation set (tsz={tsz}, vsz={vsz}). Using full dataset for evaluation.")
            ds = full_ds
        else:
            _, ds = random_split(
                full_ds,
                [tsz, vsz],
                generator=torch.Generator().manual_seed(split_seed),
            )
    else: # vf is 0.0 or 1.0, or invalid (handled by max/min)
        ds = full_ds
        if vf == 0.0:
            log.info("val_fraction is 0, using full dataset for evaluation (as validation set).")
        elif vf == 1.0:
             log.info("val_fraction is 1, using full dataset for evaluation.")


    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    log.info(f"Loaded {len(ds)} samples for evaluation.")
    return loader


def _print_formatted_table(labels: List[str], data_rows: List[List[str]]) -> None:
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


def _print_verbose_sample_details(
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
) -> None:
    """Prints detailed information for a single verbose sample."""
    print("--- Verbose sample ---")
    # Print the wider text context, assumed to be pre-cropped appropriately
    print(f"Original Text (cropped): \"{escape_newlines(original_string_cropped)}\"")
    print(f"Layer {l}, Position {p} (0-indexed for activation A_i from original token '{original_token_at_p_str}')\n")

    print(f"Original Input Context (window around P={p}, showing positions {context_display_range}):")
    _print_formatted_table(context_labels, context_data_rows)

    print(f"\nAnalysis for token following Position P={p} (i.e., predicting for seq position {p+1}):")
    for pred_type, top_n_tokens_list in analysis_predictions.items():
        tokens_str = ", ".join([f"'{t}'" for t in top_n_tokens_list])
        print(f"  - {pred_type} (Top {top_n_analysis_val}): {tokens_str}")

    print("\nGenerated Explanation (from Decoder using A_i):")
    if gen_tokens:
        # Note: topk_per_pos for decoder explanation is currently hardcoded to top-3 in _process_and_print_verbose_batch_samples
        expl_labels = ["Token:", "Decoder Top-3 Preds:"]
        expl_data_rows = [gen_tokens, topk_per_pos]
        _print_formatted_table(expl_labels, expl_data_rows)
    else:
        print("  (No explanation generated or empty)")
    
    print("-" * 60)


def _process_and_print_verbose_batch_samples(
    batch: Dict[str, Any],
    cfg: Dict[str, Any],
    models: Dict[str, torch.nn.Module],
    orig: OrigWrapper,
    tok: PreTrainedTokenizerBase,
    args: argparse.Namespace,
    sch_args: Dict[str, Any],
    device: torch.device,
    printed_count_so_far: int
) -> int:
    """Processes and prints verbose samples from a batch."""
    dec = models["dec"]
    enc = models["enc"]
    num_printed_this_batch = 0
    top_n_for_analysis = args.top_n_analysis

    for i in range(min(args.verbose_samples - printed_count_so_far, batch["A"].size(0))):
        l = int(batch["layer_idx"][i].item())
        p = int(batch["token_pos_A"][i].item())

        input_ids_seq = batch["input_ids_A"][i].unsqueeze(0).to(device)
        A_i = batch["A"][i : i + 1].to(device) # Shape: [1, hidden_size]

        # Logit Lens prediction from A_i
        A_i_cast = A_i.to(orig.model.lm_head.weight.dtype)
        logit_lens_logits_from_A_i = orig.model.lm_head(A_i_cast) # Shape [1, vocab_size]
        top_n_logit_lens_tokens = get_top_n_tokens(logit_lens_logits_from_A_i.squeeze(0), tok, top_n_for_analysis)

        # Forward passes for other logits (interventions)
        # Base Model (orig A) - prediction using A_i at (l,p)
        logits_orig_all_pos = orig.forward_with_replacement(input_ids_seq, A_i, l, p).logits # Shape [1, seq_len, vocab_size]
        logits_orig_at_p = logits_orig_all_pos[:, p].squeeze(0) # Shape [vocab_size]
        top_n_orig_A_tokens = get_top_n_tokens(logits_orig_at_p, tok, top_n_for_analysis)
        
        # Lens Recon (A_hat) - prediction using A_hat_single at (l,p)
        gen_single = dec.generate_soft(A_i, max_length=cfg["t_text"], gumbel_tau=sch_args["tau"])
        A_hat_single = enc(gen_single.generated_text_embeddings)
        logits_target_all_pos = orig.forward_with_replacement(input_ids_seq, A_hat_single, l, p).logits
        logits_target_at_p = logits_target_all_pos[:, p].squeeze(0)
        top_n_lens_recon_tokens = get_top_n_tokens(logits_target_at_p, tok, top_n_for_analysis)

        # Resample Ablation (A_hat+Δ) - prediction using A_target_i at (l,p)
        alt_idx = (i + 1) % batch["A_prime"].size(0)
        A_prime_i = batch["A_prime"][alt_idx : alt_idx + 1].to(device)
        gen_ap = dec.generate_soft(A_prime_i, max_length=cfg["t_text"], gumbel_tau=sch_args["tau"])
        A_prime_hat = enc(gen_ap.generated_text_embeddings)
        delta_res = (A_prime_i - A_prime_hat).detach()
        A_target_i = A_hat_single + delta_res
        logits_resample_all_pos = orig.forward_with_replacement(input_ids_seq, A_target_i, l, p).logits
        logits_resample_at_p = logits_resample_all_pos[:, p].squeeze(0)
        top_n_resample_ablation_tokens = get_top_n_tokens(logits_resample_at_p, tok, top_n_for_analysis)
        
        # Original input sequence processing & Base Model's natural prediction
        raw_prefix_ids = input_ids_seq[0].tolist()
        base_model_full_logits = orig.model(input_ids=input_ids_seq).logits # Shape [1, seq_len, vocab_size]
        
        top_n_natural_base_preds_for_p_plus_1: List[str]
        if p < base_model_full_logits.size(1): # Ensure p is a valid index for logits
            natural_base_logits_at_p = base_model_full_logits[0, p] # Logits for predicting token at p+1
            top_n_natural_base_preds_for_p_plus_1 = get_top_n_tokens(natural_base_logits_at_p, tok, top_n_for_analysis)
        else:
            # This case means p is at or beyond the last token, so no "next token" prediction from base model.
            top_n_natural_base_preds_for_p_plus_1 = ["N/A"] * top_n_for_analysis


        display_start_idx = max(0, p - 10)
        display_end_idx = min(len(raw_prefix_ids), p + 3 + 1)
        displayed_raw_ids = raw_prefix_ids[display_start_idx:display_end_idx]
        displayed_prefix_tokens = [escape_newlines(tok.decode([tid])) for tid in displayed_raw_ids]
        displayed_positions = [str(k) for k in range(display_start_idx, display_end_idx)]

        #for printing the original string cropped
        crop_start_idx = max(0, p - 30  )
        crop_end_idx = min(len(raw_prefix_ids), p + 30 + 1)
        original_string_cropped = escape_newlines(tok.decode(raw_prefix_ids[crop_start_idx:crop_end_idx]))
        original_string_cropped = ("..." + original_string_cropped) if crop_start_idx > 0 else original_string_cropped
        original_string_cropped = original_string_cropped + ("..." if crop_end_idx < len(raw_prefix_ids) else "")
        
        original_token_at_p_str = escape_newlines(tok.decode([raw_prefix_ids[p]])) if p < len(raw_prefix_ids) else "N/A"

        # Generated explanation processing
        gen_token_ids_full = gen_single.hard_token_ids[0].tolist()
        gen_tokens = [escape_newlines(tok.decode([tid])) for tid in gen_token_ids_full]
        # For decoder's own predictions during generation, keep using top-3 for now, or make it another arg.
        # The current request is for the "Analysis" section.
        topk_per_pos = [
            ", ".join(f'"{escape_newlines(tok.decode([x_id]).strip())}"' for x_id in torch.topk(logit_slice, k=3).indices.tolist())
            for logit_slice in gen_single.raw_lm_logits[0][: len(gen_tokens)]
        ]

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
            "Lens Recon (A_hat)": top_n_lens_recon_tokens,
            "Resample Ablation (A_hat+Δ)": top_n_resample_ablation_tokens,
        }

        _print_verbose_sample_details(
            l, p, original_token_at_p_str,
            context_display_range, context_labels, context_data_rows,
            analysis_preds_dict,
            gen_tokens, topk_per_pos,
            top_n_for_analysis, original_string_cropped
        )
        num_printed_this_batch += 1
        if (printed_count_so_far + num_printed_this_batch) >= args.verbose_samples:
            break
    return num_printed_this_batch

def _evaluate_model(
    loader: DataLoader,
    models: Dict[str, torch.nn.Module],
    orig: OrigWrapper,
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    tok: PreTrainedTokenizerBase,
    log: logging.Logger,
) -> None:
    """Runs the evaluation loop."""
    total_loss = 0.0
    n_seen = 0
    printed_verbose_total = 0

    models["dec"].eval()
    models["enc"].eval()

    with torch.no_grad():
        for b_idx, batch in tqdm(enumerate(loader), total=args.num_batches, desc="Evaluating"):
            if b_idx >= args.num_batches:
                log.info(f"Reached max {args.num_batches} batches.")
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            sch_args = {
                "tau": cfg["gumbel_tau_schedule"]["end_value"],
                "T_text": cfg["t_text"],
                "alpha": cfg["alpha_schedule"].get("value", 0.1),
                "ce_weight": cfg.get("ce_weight", 0.01),
                "kl_base_weight": cfg.get("kl_base_weight", 1.0),
            }
            
            current_models_for_step = {"dec": models["dec"], "enc": models["enc"], "orig": orig}
            losses = train_step(batch, current_models_for_step, sch_args)
            
            if printed_verbose_total < args.verbose_samples:
                num_printed_this_batch = _process_and_print_verbose_batch_samples(
                    batch, cfg, models, orig, tok, args, sch_args, device, printed_verbose_total
                )
                printed_verbose_total += num_printed_this_batch

            total_loss += losses["total"].item() * batch["A"].size(0)
            n_seen += batch["A"].size(0)

    if n_seen > 0:
        avg_loss = total_loss / n_seen
        log.info("Eval loss_total: %.4f over %d samples", avg_loss, n_seen)
    else:
        log.info("No samples processed during evaluation.")


def main() -> None:  # noqa: D401
    # tok        = AutoTokenizer.from_pretrained("Qilex/tinyStories-10k-tokenizer")
    # orig_tok   = AutoTokenizer.from_pretrained("gpt2")  # or whatever the model used

    # for tid in range(20):     # compare a few low-id tokens
    #     print(tid, tok.convert_ids_to_tokens(tid), orig_tok.convert_ids_to_tokens(tid))
    """Main entry point for evaluating a Consistency-Lens checkpoint."""
    args = _parse_arguments()
    log = _setup_logging()
    cfg = _load_config(args.config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    models_dict, tok, orig_model = _build_models(cfg, device)
    _load_checkpoint(args.checkpoint, models_dict, device)

    effective_act_dir = args.activation_dir if args.activation_dir is not None else cfg["val_activation_dir"]
    loader = _prepare_data(cfg, args, effective_act_dir, log)

    _evaluate_model(loader, models_dict, orig_model, cfg, args, device, tok, log)


if __name__ == "__main__":
    main()