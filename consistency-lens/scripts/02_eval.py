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
                "ce_weight": cfg.get("ce_weight", 0.01),
                "kl_base_weight": cfg.get("kl_base_weight", 1.0),
            }

            losses = train_step(batch, {"dec": dec, "enc": enc, "orig": orig}, sch_args)

            # Verbose per-sample analysis ----------------------------------
            with torch.no_grad():
                for i in range(min(args.verbose_samples - printed, batch["A"].size(0))):
                    idx = i
                    l = int(batch["layer_idx"][idx].item())
                    p = int(batch["token_pos"][idx].item())

                    input_ids_seq = batch["input_ids_A"][idx].unsqueeze(0)
                    A_i = batch["A"][idx : idx + 1]
                    # Forward passes
                    logits_orig = orig.forward_with_replacement(input_ids_seq, A_i, l, p).logits[:, p]
                    # Reconstruct path
                    gen_single = dec.generate_soft(A_i, max_length=cfg["t_text"], gumbel_tau=sch_args["tau"])
                    A_hat_single = enc(gen_single.generated_text_embeddings)
                    logits_target = orig.forward_with_replacement(input_ids_seq, A_hat_single, l, p).logits[:, p]

                    top_orig = torch.argmax(logits_orig, dim=-1).item()
                    top_tgt = torch.argmax(logits_target, dim=-1).item()

                    explain_text = tok.decode(gen_single.hard_token_ids[0].tolist(), skip_special_tokens=True)

                    # Original prefix tokens (0..P)
                    prefix_ids = input_ids_seq[0].tolist()
                    prefix_tokens = [tok.decode([tid]) for tid in prefix_ids]

                    # Base model next-token predictions for each prefix position
                    with torch.no_grad():
                        full_logits = orig.model(input_ids=input_ids_seq).logits  # (1, seq, V)

                    preds_prefix = [tok.decode([full_logits[0, t].argmax().item()]) for t in range(len(prefix_ids))]

                    # Generated explanation tokens + top-k
                    gen_token_ids = gen_single.hard_token_ids[0].tolist()
                    gen_tokens = [tok.decode([tid]) for tid in gen_token_ids]
                    topk_per_pos = []
                    for t_idx, logit in enumerate(gen_single.raw_lm_logits[0]):
                        topk_ids = torch.topk(logit, k=3).indices.tolist()
                        topk_per_pos.append(
                            ",".join(tok.decode([x]).strip() for x in topk_ids)
                        )

                    print("--- Verbose sample --------")
                    print(f"layer {l} | pos {p}\n")

                    print("Prefix tokens (0..P):")
                    print(" | ".join(prefix_tokens))
                    print("Base-LM top-1 next-token preds:")
                    print(" | ".join(preds_prefix))

                    print("\nToken at P original top-1 ->", tok.decode([top_orig]))
                    print("Token at P after replace   ->", tok.decode([top_tgt]))

                    print("\nGenerated explanation:")
                    print(" ".join(gen_tokens))
                    print("Top-3 per position:")
                    print(" | ".join(topk_per_pos))

                    printed += 1
                    if printed >= args.verbose_samples:
                        break

            total_loss += losses["total"].item() * batch["A"].size(0)
            n_seen += batch["A"].size(0)

    avg_loss = total_loss / n_seen
    log.info("Eval loss_total: %.4f over %d samples", avg_loss, n_seen)


if __name__ == "__main__":
    main()
