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
import logging
import random
import yaml
from pathlib import Path
from typing import List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional dependency – defer import to runtime in case users only want the smoke-test mode
try:
    from datasets import load_dataset  # type: ignore

    _HAS_DATASETS = True
except ImportError:  # pragma: no cover
    _HAS_DATASETS = False


def _pad(seq: torch.Tensor, length: int, pad_id: int) -> torch.Tensor:
    """Right-pads *seq* to *length* along dim=1."""

    if seq.size(1) >= length:
        return seq[:, :length]

    pad = seq.new_full((seq.size(0), length - seq.size(1)), pad_id)
    return torch.cat([seq, pad], dim=1)


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Dump a minimal activation cache")
    parser.add_argument("--model_name", type=str, help="Override model_name from config.")
    parser.add_argument("--output_dir", type=str, default="consistency-lens/data/activations")
    parser.add_argument("--num_samples", type=int, help="Override number of samples, otherwise from config")
    parser.add_argument("--seq_len", type=int, help="Override sequence length, otherwise from config")
    parser.add_argument("--layer_idx", type=int, help="Override layer index, otherwise from config")
    parser.add_argument("--config_path", type=str, default="consistency-lens/config/lens.yaml")
    parser.add_argument("--seed", type=int, default=0)

    # HuggingFace dataset options
    parser.add_argument(
        "--hf_dataset_name",
        type=str,
        default="NeelNanda/pile-10k",
        help="HF dataset to draw raw text samples from (default: Pile-10k)",
    )
    parser.add_argument(
        "--use_hf_dataset",
        action="store_true",
        help="If set, load --hf_dataset_name via datasets.load_dataset() instead of the fixed prompt list.",
    )
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default="consistency-lens/data/corpus/pile-10k",
        help="Local directory where the HF dataset cache should be stored.",
    )

    args = parser.parse_args()

    # Load YAML config
    with open(args.config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Inherit defaults from config, then CLI overrides
    seq_len = args.seq_len or cfg.get("activation_dumper", {}).get("seq_len", 32)
    num_samples = args.num_samples or cfg.get("activation_dumper", {}).get("num_samples", 256)
    layer_idx = args.layer_idx if args.layer_idx is not None else cfg.get("layer_l", 0)

    # ------------------------------------------------------------------
    # Logging setup
    # ------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    # Some tiny models (GPT-2) miss an explicit pad_token.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info(f"Tokenizer: {tokenizer}")
    logging.info(f"Model name: {cfg['model_name']}")

    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"])
    model.eval()

    # ---------------------------------------------------------------------
    # Decide the source of raw text samples.
    # ---------------------------------------------------------------------

    prompts: Sequence[str]

    if args.use_hf_dataset:
        if not _HAS_DATASETS:
            raise RuntimeError(
                "datasets library not found – install with `pip install datasets` or omit --use_hf_dataset"
            )

        ds_split = "train"  # Pile-10k exposes only a single split

        log.info(
            "Loading '%s' (%s split) via HuggingFace datasets … this may take a moment.",
            args.hf_dataset_name,
            ds_split,
        )

        # Ensure cache directory exists so `datasets` can write without permission issues.
        cache_path = Path(args.dataset_cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Download/cache inside project tree so CI environments w/out $HOME permissions still work.
        # NOTE: We load the whole tiny (10 k rows) split into memory – acceptable for smoke tests.
        # TODO(kitf): when switching to a larger dataset, call `load_dataset(..., streaming=True)`
        #             and iterate lazily to avoid high RAM usage.
        ds = load_dataset(args.hf_dataset_name, split=ds_split, cache_dir=str(cache_path))

        # Keep just the 'text' field & strip trailing whitespace
        prompts = [str(x["text"]).strip() for x in ds]

        if len(prompts) == 0:
            raise ValueError(
                f"Dataset '{args.hf_dataset_name}' returned zero usable rows – check the column names or split."
            )

        log.info("Loaded %s text snippets from '%s'.", f"{len(prompts):,}", args.hf_dataset_name)
    else:
        log.info("Using hard-coded list of prompts for smoke tests.")
        # Fallback – tiny hard-coded list for smoke tests.
        prompts = [
            "The quick brown fox jumps over the lazy dog.",
            "Consistency Lens minimal test sentence.",
            "Hello world!",
            "Another short prompt for dumping activations.",
        ]

    log.info("Dumping %d activation samples to %s …", num_samples, out_dir)

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

        # ------------------------------------------------------------------
        # Choose a random token position that is *inside* the non-padding
        # region of *both* sequences to avoid extracting activations from PAD
        # tokens. Lower-bound at 5 to skip trivial prefix tokens.
        # ------------------------------------------------------------------

        nonpad_len_A = int((toks_A != tokenizer.pad_token_id).sum())
        nonpad_len_Ap = int((toks_Ap != tokenizer.pad_token_id).sum())

        max_valid = min(nonpad_len_A, nonpad_len_Ap, seq_len - 1) - 1  # exclusive upper bound previously -2
        min_valid = 5

        if max_valid < min_valid:
            # Fallback: pick middle of shortest sequence
            token_pos = max(0, (max_valid + 1) // 2)
        else:
            token_pos = random.randint(min_valid, max_valid)

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

    # ------------------------------------------------------------------
    # Log a quick example so users can inspect the structure.
    # ------------------------------------------------------------------
    example_path = out_dir / "00000.pt"
    if example_path.exists():
        ex = torch.load(example_path, map_location="cpu")

        # Build a concise summary: tensor shapes + token text
        def _shape(t):
            return list(t.shape) if torch.is_tensor(t) else "-"

        summary = {
            "A_shape": _shape(ex["A"]),
            "A_prime_shape": _shape(ex["A_prime"]),
            "input_ids_A_len": int(ex["input_ids_A"].numel()),
            "input_ids_A_prime_len": int(ex["input_ids_A_prime"].numel()),
            "token_pos": int(ex["token_pos"]),
            "layer_idx": int(ex["layer_idx"]),
            "text_A": tokenizer.decode(ex["input_ids_A"]),
            "text_A_prime": tokenizer.decode(ex["input_ids_A_prime"]),
        }

        log.info("Example sample (00000.pt): %s", summary)
    else:
        log.warning("Could not find example sample at %s; something went wrong?", example_path)


if __name__ == "__main__":
    main()
