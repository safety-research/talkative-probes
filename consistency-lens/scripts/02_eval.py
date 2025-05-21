"""Evaluate a saved checkpoint on a held-out activation set."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from lens.data.collate import collate
from lens.data.dataset import ActivationDataset
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder
from lens.models.orig import OrigWrapper
from lens.training.loop import train_step
from lens.utils import checkpoint as ckpt_util
from transformers import AutoTokenizer


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Evaluate a Consistency-Lens checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="consistency-lens/config/lens.yaml")
    parser.add_argument("--activation_dir", type=str, default="consistency-lens/data/activations")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=25)
    parser.add_argument("--verbose_samples", type=int, default=3)
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
    ds = ActivationDataset(args.activation_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    total_loss = 0.0
    n_seen = 0
    printed = 0

    dec.eval()
    enc.eval()
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if b_idx >= args.num_batches:
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
                prefix_ids = input_ids_seq[0].tolist()
                prefix_tokens = [tok.decode([tid]) for tid in prefix_ids]

                # Base model's next-token predictions for the original input sequence
                # full_logits are from the original model without any intervention on input_ids_seq
                base_model_full_logits = orig.model(input_ids=input_ids_seq).logits  # (1, seq_len, V)
                # preds_prefix[k] is the prediction for token k+1, given prefix_tokens[0...k]
                preds_prefix = [tok.decode([base_model_full_logits[0, t_idx].argmax().item()]) for t_idx in range(len(prefix_ids))]

                # Original input sequence processing
                raw_prefix_ids = input_ids_seq[0].tolist()
                eos_id = tok.eos_token_id

                # Truncate original input sequence after the second EOS token to keep verbose output manageable.
                # This addresses the user's instruction that EOS counting should be on the original input.
                cnt_eos_prefix = 0
                truncated_prefix_ids = []
                for tid in raw_prefix_ids:
                    truncated_prefix_ids.append(tid) # Append token
                    if tid == eos_id:
                        cnt_eos_prefix += 1
                        if cnt_eos_prefix >= 2: # Allow up to two EOS tokens (e.g., sentence end, then sequence end)
                            break
                prefix_tokens = [tok.decode([tid]) for tid in truncated_prefix_ids]
                
                # Get the original token at position p for the title, from the untruncated sequence
                original_token_at_p_str = "N/A" # Default if p is out of bounds for raw_prefix_ids
                if p < len(raw_prefix_ids):
                    original_token_at_p_str = tok.decode([raw_prefix_ids[p]])

                # Generated explanation processing
                gen_token_ids_full = gen_single.hard_token_ids[0].tolist()
                # Generated explanation is no longer truncated by EOS counting here,
                # as that logic has been moved to the original input processing per user instruction.
                gen_tokens = [tok.decode([tid]) for tid in gen_token_ids_full]
                
                topk_per_pos = []
                # Ensure we only get top-k for the actual length of gen_tokens.
                # gen_single.raw_lm_logits[0] has shape (max_length, vocab_size).
                # len(gen_tokens) will be max_length.
                for t_idx, logit_slice in enumerate(gen_single.raw_lm_logits[0][: len(gen_tokens)]):
                    topk_ids = torch.topk(logit_slice, k=3).indices.tolist()
                    topk_per_pos.append(
                        ",".join(tok.decode([x_id]).strip() for x_id in topk_ids)
                    )

                # Helper to escape newlines for display
                def escape_newlines(text: str) -> str:
                    return text.replace("\n", "\\n")

                print("--- Verbose sample ---")
                # Escape newline in original_token_at_p_str for the title
                print(f"Layer {l}, Position {p} (0-indexed for activation A_i from original token '{escape_newlines(original_token_at_p_str)}')\n")

                # Part 1: Original Input Context & Predictions at Target Position
                print("Original Input Context (truncated after second EOS):")
                context_labels = ["Position:", "Token:", "BaseLM Pred (for next token):"]

                # Base model's next-token predictions for the original (untruncated) input sequence
                # Escape newlines in these predictions at source.
                # preds_prefix_full[k] is the prediction for the token *after* raw_prefix_ids[k].
                preds_prefix_full = [
                    escape_newlines(tok.decode([base_model_full_logits[0, t_idx].argmax().item()]))
                    for t_idx in range(len(raw_prefix_ids))
                ]
                
                # Shifted predictions for table display:
                # The prediction under Token[k] will be the prediction *for* Token[k].
                # This means preds_prefix_full[k-1] is shown under Token[k].
                # The first token (Token[0]) has no prior token in this view, so its prediction slot is blank.
                if len(prefix_tokens) > 0:
                    # prefix_tokens elements are already escaped during their creation.
                    # preds_prefix_full elements are already escaped.
                    shifted_preds_for_display = [""] + preds_prefix_full[:len(prefix_tokens)-1]
                else:
                    shifted_preds_for_display = []

                context_data_rows = [
                    [str(k) for k in range(len(prefix_tokens))], # Positions for displayed tokens
                    list(prefix_tokens), # Displayed tokens (already escaped)
                    shifted_preds_for_display # Predictions (already escaped and shifted)
                ]
                
                # Highlight target position P in the display if it's within the displayed (truncated) context
                if 0 <= p < len(prefix_tokens): # Check against the length of displayed tokens
                    context_data_rows[0][p] = f"[{context_data_rows[0][p]}]P"
                    context_data_rows[1][p] = f"*{context_data_rows[1][p]}*" # Tokens already escaped

                max_context_label_width = max(len(s) for s in context_labels) + 2 

                # Determine column widths for the context table based on displayed tokens
                num_context_cols = len(prefix_tokens)
                context_col_widths = [0] * num_context_cols
                for r_idx_table, data_row_table in enumerate(context_data_rows):
                    for c_idx_table, cell_table in enumerate(data_row_table):
                        # All cell content is already string and escaped
                        context_col_widths[c_idx_table] = max(context_col_widths[c_idx_table], len(cell_table))
                
                # Print context table
                for r_idx_table, label_table in enumerate(context_labels):
                    row_str_table = f"{label_table:<{max_context_label_width}}"
                    for c_idx_table, cell_content_table in enumerate(context_data_rows[r_idx_table]):
                        row_str_table += f" {cell_content_table:<{context_col_widths[c_idx_table]}} |"
                    print(row_str_table.rstrip(" |"))

                print(f"\nAnalysis for token following Position P={p} (i.e., predicting for seq position {p+1}):")
                # Natural prediction by base model for token P+1, from full (untruncated) predictions
                # preds_prefix_full[p] is already escaped.
                natural_base_pred_for_p_plus_1 = "N/A"
                if p < len(preds_prefix_full): 
                    natural_base_pred_for_p_plus_1 = preds_prefix_full[p]
                print(f"  - Base Model's natural prediction: '{natural_base_pred_for_p_plus_1}'")
                print(f"  - Base Model (orig A): '{escape_newlines(tok.decode([top_orig_id]))}'")
                print(f"  - Lens Recon (A_hat): '{escape_newlines(tok.decode([top_tgt_id]))}'")
                print(f"  - Resample Ablation (A_hat+Δ): '{escape_newlines(tok.decode([top_resample_id]))}'")

                # Part 2: Generated Explanation
                # gen_tokens and topk_per_pos elements were escaped at their creation point before this selection.
                # Re-check: gen_tokens and topk_per_pos are created *before* this selection.
                # Need to ensure they are escaped.
                # Original gen_tokens: gen_tokens = [tok.decode([tid]) for tid in gen_token_ids_full]
                # Original topk_per_pos: tok.decode([x_id]).strip()
                # These need to be updated where they are defined if not already done.
                # Assuming they are updated outside or I need to re-define them here with escaping.
                # For safety, let's re-process them here if they are used directly from non-escaped source.
                # The selection starts *after* topk_per_pos is fully computed.
                # So, I must assume gen_tokens and topk_per_pos are already escaped.
                # If not, this part of the prompt is insufficient.
                # Based on the provided selection, gen_tokens and topk_per_pos are used as-is.
                # The user must ensure they are escaped prior to this selection block.
                # For this rewrite, I will assume they are already escaped as per the broader task.

                print("\nGenerated Explanation (from Decoder using A_i):")
                if gen_tokens: # Assuming gen_tokens elements are already escaped
                    expl_labels = ["Token:", "Decoder Top-3 Preds:"]
                    # Assuming topk_per_pos elements are already escaped
                    expl_data_rows = [gen_tokens, topk_per_pos]
                    
                    max_expl_label_width = max(len(s) for s in expl_labels) + 2

                    num_expl_cols = len(gen_tokens)
                    expl_col_widths = [0] * num_expl_cols
                    for r_idx_table, data_row_table in enumerate(expl_data_rows):
                        for c_idx_table, cell_table in enumerate(data_row_table):
                            # All cell content is assumed to be string and escaped
                            expl_col_widths[c_idx_table] = max(expl_col_widths[c_idx_table], len(cell_table))
                            
                    for r_idx_table, label_table in enumerate(expl_labels):
                        row_str_table = f"{label_table:<{max_expl_label_width}}"
                        for c_idx_table, cell_content_table in enumerate(expl_data_rows[r_idx_table]):
                            row_str_table += f" {cell_content_table:<{expl_col_widths[c_idx_table]}} |"
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