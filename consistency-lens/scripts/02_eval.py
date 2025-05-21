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
from lens.models.encoder import Encoder
from lens.models.orig import OrigWrapper
from lens.training.loop import train_step
from lens.utils import checkpoint as ckpt_util
from transformers import AutoTokenizer
from tqdm import tqdm
# Helper to escape newlines for display
def escape_newlines(text: str) -> str:
    return text.replace("\n", "\\n")



def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Evaluate a Consistency-Lens checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="consistency-lens/config/lens_simple.yaml")
    parser.add_argument("--activation_dir", type=str, help="Override activation_dir from config")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=25)
    parser.add_argument("--verbose_samples", type=int, default=3)
    parser.add_argument("--val_fraction", type=float, help="Fraction of dataset for validation")
    parser.add_argument("--split_seed",   type=int,   help="Seed for train/val split")
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Logging setup
    # ---------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    with open(args.config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Build models
    # ------------------------------------------------------------------
    model_name = cfg["model_name"]
    dec = Decoder(DecoderConfig(model_name=model_name, n_prompt_tokens=cfg["decoder_n_prompt_tokens"]))
    tok = AutoTokenizer.from_pretrained(model_name)
    dec.set_prompt(cfg.get("decoder_prompt", "Explain: "), tok)
    enc = Encoder(model_name)
    orig = OrigWrapper(model_name, load_in_8bit=False)
    orig.model.to(device)

    models = {"dec": dec.to(device), "enc": enc.to(device)}

    # Dummy optimiser just so checkpoint.load can restore state dict (optional)
    opt = torch.optim.AdamW(params=list(dec.parameters()) + list(enc.parameters()), lr=1e-4)

    ckpt_util.load(args.checkpoint, models=models, optim=opt, map_location=device)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    # Determine activation directory from CLI or config
    effective_act_dir = args.activation_dir if args.activation_dir is not None else cfg["val_activation_dir"]
    print(f"Loading activations from {effective_act_dir}")
    full_ds = ActivationDataset(effective_act_dir)
    vf = max(0.0, min(1.0, args.val_fraction if args.val_fraction is not None else cfg.get('val_fraction', 0.1)))
    if 0 < vf < 1.0:
        vsz = int(len(full_ds) * vf)
        tsz = len(full_ds) - vsz
        _, ds = random_split(
            full_ds,
            [tsz, vsz],
            generator=torch.Generator().manual_seed(args.split_seed if args.split_seed is not None else cfg.get('split_seed', 42)),
        )
    else:
        ds = full_ds
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    print(f"Loaded {len(ds)} samples")

    total_loss = 0.0
    n_seen = 0
    printed = 0

    dec.eval()
    enc.eval()
    with torch.no_grad():
        for b_idx, batch in tqdm(enumerate(loader), total=args.num_batches):
            if b_idx >= args.num_batches:
                print(f"Reached {args.num_batches} batches")
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            sch_args = {
                "tau": cfg["gumbel_tau_schedule"]["end_value"],
                "T_text": cfg["t_text"],
                "alpha": cfg["alpha_schedule"].get("value", 0.1),
                "ce_weight": cfg.get("ce_weight", 0.01), # Retained from original, though not explicitly in README formula for eval
                "kl_base_weight": cfg.get("kl_base_weight", 1.0), # Retained from original
            }

            losses = train_step(batch, {"dec": dec, "enc": enc, "orig": orig}, sch_args)
            # Verbose per-sample analysis ----------------------------------
            # The inner torch.no_grad() here is removed as it's redundant with the outer one.
            for i in range(min(args.verbose_samples - printed, batch["A"].size(0))):
                # Use 'i' directly instead of 'idx = i'
                l = int(batch["layer_idx"][i].item())
                p = int(batch["token_pos"][i].item())
                # The conditional skipping of long sequences (if p > 13) is removed;
                # context windowing below handles appropriate display.

                input_ids_seq = batch["input_ids_A"][i].unsqueeze(0)
                A_i = batch["A"][i : i + 1] # Activation from original model at (l,p) for this input

                # Forward passes to get logits for comparison
                # Logits at pos p, using original activation A_i
                logits_orig = orig.forward_with_replacement(input_ids_seq, A_i, l, p).logits[:, p]
                
                # Generate explanation and reconstruct A_hat
                gen_single = dec.generate_soft(A_i, max_length=cfg["t_text"], gumbel_tau=sch_args["tau"])
                A_hat_single = enc(gen_single.generated_text_embeddings)
                logits_target = orig.forward_with_replacement(input_ids_seq, A_hat_single, l, p).logits[:, p]

                # --- Resample ablation using A_prime from batch (wrap around if needed) ---
                alt_idx = (i + 1) % batch["A_prime"].size(0)
                A_prime_i = batch["A_prime"][alt_idx : alt_idx + 1]
                gen_ap = dec.generate_soft(A_prime_i, max_length=cfg["t_text"], gumbel_tau=sch_args["tau"])
                A_prime_hat = enc(gen_ap.generated_text_embeddings)
                delta_res = (A_prime_i - A_prime_hat).detach()
                A_target_i = A_hat_single + delta_res
                logits_resample = orig.forward_with_replacement(input_ids_seq, A_target_i, l, p).logits[:, p]
                top_resample_id = torch.argmax(logits_resample, dim=-1).item()

                top_orig_id = torch.argmax(logits_orig, dim=-1).item()
                top_tgt_id = torch.argmax(logits_target, dim=-1).item()

                # Original input sequence processing
                prefix_ids = input_ids_seq[0].tolist() # Full sequence of token IDs.
                # Store full, unescaped tokens; used for context by preds_prefix, not directly for display table.
                prefix_tokens_full_unescaped = [tok.decode([tid]) for tid in prefix_ids] 

                # Base model's next-token predictions for the original input sequence
                # full_logits are from the original model without any intervention on input_ids_seq
                base_model_full_logits = orig.model(input_ids=input_ids_seq).logits  # (1, seq_len, V)
                # preds_prefix[k] is the prediction for token k+1, given prefix_tokens_full_unescaped[0...k].
                preds_prefix = [tok.decode([base_model_full_logits[0, t_idx].argmax().item()]) for t_idx in range(len(prefix_ids))]

                # raw_prefix_ids is the same as prefix_ids, representing the full original sequence of token IDs.
                raw_prefix_ids = input_ids_seq[0].tolist()
                # eos_id was used for old truncation logic, no longer needed for the display window.

                # Define window for displaying original input context:
                # 10 tokens before p, token p itself, and 3 tokens after p.
                display_start_idx = max(0, p - 10)
                # p+3 is the last token index to include, so p+3+1 for exclusive slice end.
                display_end_idx = min(len(raw_prefix_ids), p + 3 + 1) 

                # Extract the window of token IDs from the full sequence.
                displayed_raw_ids = raw_prefix_ids[display_start_idx:display_end_idx]
                # Decode and escape tokens in the window for display in the table.
                displayed_prefix_tokens = [escape_newlines(tok.decode([tid])) for tid in displayed_raw_ids]
                # Create a list of stringified absolute positions for the displayed tokens.
                displayed_positions = [str(k) for k in range(display_start_idx, display_end_idx)]
                
                # Get the original token at position p (from untruncated sequence) for the title.
                # This is already escaped during its creation.
                original_token_at_p_str = "N/A" 
                if p < len(raw_prefix_ids):
                    original_token_at_p_str = escape_newlines(tok.decode([raw_prefix_ids[p]]))

                # Generated explanation processing (this part remains unchanged)
                gen_token_ids_full = gen_single.hard_token_ids[0].tolist()
                # Generated explanation tokens are escaped for consistent display.
                gen_tokens = [escape_newlines(tok.decode([tid])) for tid in gen_token_ids_full]
                
                topk_per_pos = []
                # Ensure we only get top-k for the actual length of gen_tokens.
                for t_idx, logit_slice in enumerate(gen_single.raw_lm_logits[0][: len(gen_tokens)]):
                    topk_ids = torch.topk(logit_slice, k=3).indices.tolist()
                    topk_per_pos.append(
                        ",".join(escape_newlines(tok.decode([x_id]).strip()) for x_id in topk_ids)
                    )

                print("--- Verbose sample ---")
                # original_token_at_p_str is already escaped where it's defined.
                print(f"Layer {l}, Position {p} (0-indexed for activation A_i from original token '{original_token_at_p_str}')\n")

                # Part 1: Original Input Context & Predictions at Target Position
                # Update title to reflect new windowing approach.
                context_display_range = f"{display_start_idx}-{display_end_idx-1}" if displayed_positions else "empty"
                print(f"Original Input Context (window around P={p}, showing positions {context_display_range}):")
                context_labels = ["Position:", "Token:", "BaseLM Pred (for next token):"]

                # Base model's next-token predictions for the *entire original* (untruncated) input sequence.
                # These predictions are escaped at source.
                # preds_prefix_full[k] is the prediction for the token *after* raw_prefix_ids[k].
                preds_prefix_full = [
                    escape_newlines(tok.decode([base_model_full_logits[0, t_idx].argmax().item()]))
                    for t_idx in range(len(raw_prefix_ids))
                ]
                
                # Shifted predictions for table display:
                # The prediction shown under Token[k] (absolute index k) is the prediction *for* Token[k],
                # which is generated given tokens up to k-1, i.e., preds_prefix_full[k-1].
                shifted_preds_for_display = []
                for k_rel in range(len(displayed_prefix_tokens)): # k_rel is the relative index within the displayed window
                    abs_idx = display_start_idx + k_rel # Absolute index in raw_prefix_ids
                    if abs_idx == 0: # The first token of the sequence has no preceding token for prediction.
                        shifted_preds_for_display.append("")
                    else:
                        # Prediction for token raw_prefix_ids[abs_idx] is preds_prefix_full[abs_idx - 1].
                        # Ensure abs_idx - 1 is a valid index for preds_prefix_full.
                        if abs_idx - 1 < len(preds_prefix_full):
                             shifted_preds_for_display.append(preds_prefix_full[abs_idx - 1])
                        else:
                             # This case implies an issue with preds_prefix_full length or indexing logic.
                             shifted_preds_for_display.append("ERR_IDX") 
                
                context_data_rows = [
                    displayed_positions,       # Absolute positions for the tokens in the window
                    list(displayed_prefix_tokens), # Windowed tokens (already escaped)
                    shifted_preds_for_display  # Corresponding predictions (already escaped)
                ]
                
                # Highlight target position P in the display.
                # relative_p is p's index within the displayed_prefix_tokens list (i.e., the window).
                relative_p = p - display_start_idx 
                if 0 <= relative_p < len(displayed_prefix_tokens): # Check if P is within the displayed window
                    # Mark the position number and the token itself.
                    context_data_rows[0][relative_p] = f"[{context_data_rows[0][relative_p]}]P"
                    context_data_rows[1][relative_p] = f"*{context_data_rows[1][relative_p]}*" # Tokens are already escaped

                max_context_label_width = max(len(s) for s in context_labels) + 2 if context_labels else 0

                # Determine column widths for the context table based on displayed tokens.
                num_context_cols = len(displayed_prefix_tokens)
                context_col_widths = [0] * num_context_cols
                if num_context_cols > 0: # Proceed only if there are columns to process (i.e., window is not empty).
                    for r_idx_table, data_row_table in enumerate(context_data_rows):
                        for c_idx_table, cell_table in enumerate(data_row_table):
                            # All cell content is assumed to be string and escaped.
                            context_col_widths[c_idx_table] = max(context_col_widths[c_idx_table], len(cell_table))
                
                # Print context table.
                for r_idx_table, label_table in enumerate(context_labels):
                    row_str_table = f"{label_table:<{max_context_label_width}}"
                    for c_idx_table, cell_content_table in enumerate(context_data_rows[r_idx_table]):
                        # Ensure column width is available; default to 0 if not (should not happen with num_context_cols check).
                        col_width = context_col_widths[c_idx_table] if c_idx_table < len(context_col_widths) else 0
                        row_str_table += f" {cell_content_table:<{col_width}} |"
                    print(row_str_table.rstrip(" |"))

                print(f"\nAnalysis for token following Position P={p} (i.e., predicting for seq position {p+1}):")
                # Natural prediction by base model for token P+1, from full (untruncated) predictions.
                # preds_prefix_full[p] is already escaped.
                natural_base_pred_for_p_plus_1 = "N/A"
                if p < len(preds_prefix_full): 
                    natural_base_pred_for_p_plus_1 = preds_prefix_full[p]
                print(f"  - Base Model's natural prediction: '{natural_base_pred_for_p_plus_1}'")
                print(f"  - Base Model (orig A): '{escape_newlines(tok.decode([top_orig_id]))}'")
                print(f"  - Lens Recon (A_hat): '{escape_newlines(tok.decode([top_tgt_id]))}'")
                print(f"  - Resample Ablation (A_hat+Δ): '{escape_newlines(tok.decode([top_resample_id]))}'")

                # Part 2: Generated Explanation
                # gen_tokens and topk_per_pos elements were escaped at their creation point.
                print("\nGenerated Explanation (from Decoder using A_i):")
                if gen_tokens: # Assuming gen_tokens elements are already escaped
                    expl_labels = ["Token:", "Decoder Top-3 Preds:"]
                    # Assuming topk_per_pos elements are already escaped
                    expl_data_rows = [gen_tokens, topk_per_pos]
                    
                    max_expl_label_width = max(len(s) for s in expl_labels) + 2 if expl_labels else 0

                    num_expl_cols = len(gen_tokens)
                    expl_col_widths = [0] * num_expl_cols
                    if num_expl_cols > 0: # Proceed only if there are columns.
                        for r_idx_table, data_row_table in enumerate(expl_data_rows):
                            for c_idx_table, cell_table in enumerate(data_row_table):
                                # All cell content is assumed to be string and escaped
                                expl_col_widths[c_idx_table] = max(expl_col_widths[c_idx_table], len(cell_table))
                            
                    for r_idx_table, label_table in enumerate(expl_labels):
                        row_str_table = f"{label_table:<{max_expl_label_width}}"
                        for c_idx_table, cell_content_table in enumerate(expl_data_rows[r_idx_table]):
                            col_width = expl_col_widths[c_idx_table] if c_idx_table < len(expl_col_widths) else 0
                            row_str_table += f" {cell_content_table:<{col_width}} |"
                        print(row_str_table.rstrip(" |"))
                else:
                    print("  (No explanation generated or empty)")
                
                print("-" * 60) # Separator for next sample
                printed += 1
                if printed >= args.verbose_samples:
                    break

            total_loss += losses["total"].item() * batch["A"].size(0)
            n_seen += batch["A"].size(0)

    avg_loss = total_loss / n_seen
    log.info("Eval loss_total: %.4f over %d samples", avg_loss, n_seen)


if __name__ == "__main__":
    main()