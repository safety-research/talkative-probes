"""MVP extractor.

Generates a *tiny* activation cache suitable for unit / smoke tests.  It runs a
handful of short prompts through a small HF CausalLM and stores tuples:

    {"A": <tensor>, "A_prime": <tensor>, "input_ids_A": <tensor>}

Each item is written to ``output_dir/{idx:05d}.pt`` so the dataset can lazily
load them.

This is *not* the full scalable extractor described in the README – it is the
minimum needed to unblock the end-to-end training loop on a laptop.
"""

from __future__ import annotations

import argparse
import random
import yaml
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _pad(seq: torch.Tensor, length: int, pad_id: int) -> torch.Tensor:
    """Right-pads *seq* to *length* along dim=1."""

    if seq.size(1) >= length:
        return seq[:, :length]

    pad = seq.new_full((seq.size(0), length - seq.size(1)), pad_id)
    return torch.cat([seq, pad], dim=1)


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Dump a minimal activation cache")
    parser.add_argument("--model_name", type=str, default="sshleifer/tiny-gpt2")
    parser.add_argument("--output_dir", type=str, default="consistency-lens/data/activations")
    parser.add_argument("--num_samples", type=int, help="Override number of samples, otherwise from config")
    parser.add_argument("--seq_len", type=int, help="Override sequence length, otherwise from config")
    parser.add_argument("--layer_idx", type=int, help="Override layer index, otherwise from config")
    parser.add_argument("--config_path", type=str, default="consistency-lens/config/lens.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Load YAML config
    with open(args.config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Inherit defaults from config, then CLI overrides
    seq_len = args.seq_len or cfg.get("activation_dumper", {}).get("seq_len", 32)
    num_samples = args.num_samples or cfg.get("activation_dumper", {}).get("num_samples", 256)
    layer_idx = args.layer_idx if args.layer_idx is not None else cfg.get("layer_l", 0)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Some tiny models (GPT-2) miss an explicit pad_token.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.eval()

    prompts: List[str] = [
        "The quick brown fox jumps over the lazy dog.",
        "Consistency Lens minimal test sentence.",
        "Hello world!",
        "Another short prompt for dumping activations.",
    ]

    for idx in range(num_samples):
        # --- Select prompts for A and A′ ---
        txt_A = random.choice(prompts)
        txt_Ap = random.choice(prompts)

        toks_A = tokenizer(txt_A, return_tensors="pt").input_ids
        toks_Ap = tokenizer(txt_Ap, return_tensors="pt").input_ids

        toks_A = _pad(toks_A, seq_len, tokenizer.pad_token_id)
        toks_Ap = _pad(toks_Ap, seq_len, tokenizer.pad_token_id)

        with torch.no_grad():
            out_A = model(toks_A, output_hidden_states=True)
            out_Ap = model(toks_Ap, output_hidden_states=True)

        hidden_A = out_A.hidden_states[layer_idx]  # (1, L, d_model)
        hidden_Ap = out_Ap.hidden_states[layer_idx]

        # Random valid token position P (>=10 and < seq_len-1)
        token_pos = random.randint(10, seq_len - 2)

        A = hidden_A[0, token_pos].half()  # (d_model,)
        A_prime = hidden_Ap[0, token_pos].half()

        sample = {
            "A": A,
            "A_prime": A_prime,
            "input_ids_A": toks_A.squeeze(0),
            "input_ids_A_prime": toks_Ap.squeeze(0),
            "token_pos": token_pos,
            "layer_idx": layer_idx,
        }

        torch.save(sample, out_dir / f"{idx:05d}.pt")


if __name__ == "__main__":
    main()
