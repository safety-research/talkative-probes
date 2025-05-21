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
from typing import Iterator # List, Sequence were for _pad, no longer needed here

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# Optional dependency – defer import to runtime in case users only want the smoke-test mode
try:
    from datasets import load_dataset  # type: ignore

    _HAS_DATASETS = True
except ImportError:  # pragma: no cover
    # load_dataset will be undefined if this happens.
    _HAS_DATASETS = False
    load_dataset = None # Make it explicit that load_dataset might be None


def iter_hf_text(dataset_name: str, cache_dir: str | None, split: str, num_samples: int) -> Iterator[str]:
    """Stream plain-text documents from a HuggingFace dataset until *num_samples* yielded."""
    if not _HAS_DATASETS or load_dataset is None: # Should be caught by main, but defensive
        raise ImportError("datasets library is not available or failed to import.")

    ds = load_dataset(dataset_name, split=split, streaming=True, cache_dir=cache_dir)
    count = 0
    for item in ds:
        text: str | None = None
        # heuristic: find first string field
        for v in item.values():
            if isinstance(v, str):
                text = v
                break
        if text is None:
            continue
        yield text
        count += 1
        if count >= num_samples:
            break


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Dump a minimal activation cache")
    parser.add_argument("--output_dir", type=str, help="Override output directory, otherwise from config")
    parser.add_argument("--num_samples", type=int, help="Override number of samples, otherwise from config")
    parser.add_argument("--seq_len", type=int, help="Override sequence length, otherwise from config")
    parser.add_argument("--layer_idx", type=int, help="Override layer index, otherwise from config")
    parser.add_argument("--config_path", type=str, default="consistency-lens/config/lens_simple.yaml")
    parser.add_argument("--seed", type=int, default=0)

    # HuggingFace dataset options
    parser.add_argument(
        "--hf_dataset_name",
        type=str,
        help="Override HF dataset to draw text from, otherwise from config",
    )
    parser.add_argument(
        "--use_hf_dataset",
        action="store_true",
        help="If set, load --hf_dataset_name via datasets.load_dataset() instead of the fixed prompt list.",
    )
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        help="Override HF dataset cache dir, otherwise from config",
    )
    parser.add_argument(
        "--hf_split",
        type=str,
        default=None,
        help="Override HF dataset split for streaming (e.g. train)",
    )

    # Optional validation dump in same run
    parser.add_argument("--val_hf_split", type=str, help="HF split for validation activations (e.g. validation)")
    parser.add_argument("--val_output_dir", type=str, help="Output directory for validation activations")
    parser.add_argument("--val_num_samples", type=int, help="Number of validation samples (defaults to num_samples)")

    args = parser.parse_args()

    # Load YAML config
    with open(args.config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    activation_dumper_cfg = cfg["activation_dumper"]

    # Determine effective config: CLI > YAML
    effective_output_dir = args.output_dir if args.output_dir is not None else activation_dumper_cfg["output_dir"]
    seq_len = args.seq_len if args.seq_len is not None else activation_dumper_cfg["seq_len"]
    num_samples = args.num_samples if args.num_samples is not None else activation_dumper_cfg["num_samples"]
    layer_idx = args.layer_idx if args.layer_idx is not None else cfg["layer_l"]
    effective_use_hf = args.use_hf_dataset or activation_dumper_cfg.get("use_hf_dataset", False)
    effective_hf_dataset_name = args.hf_dataset_name if args.hf_dataset_name is not None else activation_dumper_cfg.get("hf_dataset_name")
    effective_dataset_cache_dir = args.dataset_cache_dir if args.dataset_cache_dir is not None else activation_dumper_cfg.get("dataset_cache_dir")
    effective_hf_split = args.hf_split if args.hf_split is not None else activation_dumper_cfg.get("hf_split", "train")

    # ------------------------------------------------------------------
    # Build list of dataset splits to dump (e.g. train + optional validation)
    # ------------------------------------------------------------------
    splits_to_dump = [
        {
            "name": effective_hf_split,
            "output_dir": effective_output_dir,
            "num_samples": num_samples,
        }
    ]

    # Determine validation dump parameters either from CLI or YAML
    val_out_cfg = activation_dumper_cfg.get("val_output_dir")
    val_split_cfg = activation_dumper_cfg.get("val_hf_split", "validation")
    if args.val_output_dir or val_out_cfg:
        val_out_dir = args.val_output_dir if args.val_output_dir else val_out_cfg
        val_split = args.val_hf_split if args.val_hf_split else val_split_cfg
        val_samples = args.val_num_samples if args.val_num_samples is not None else activation_dumper_cfg.get("val_num_samples", num_samples)
        splits_to_dump.append({
            "name": val_split,
            "output_dir": val_out_dir,
            "num_samples": val_samples,
        })

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

    out_dir = Path(effective_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    # Some tiny models (GPT-2) miss an explicit pad_token.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info(f"Tokenizer: {tokenizer}")
    log.info(f"Model name: {cfg['model_name']}")

    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"]).to(device)
    model.eval()

    # ---------------------------------------------------------------------
    # Decide the source of raw text samples for the *pairs* (A, A_prime).
    # Both texts are now taken from the same iterator so that they are drawn
    # from the same underlying distribution without requiring two HF splits.
    # ---------------------------------------------------------------------
    if effective_use_hf:
        # Single split – create two independent iterators over the same split
        text_iter_A = iter_hf_text(
            effective_hf_dataset_name,
            effective_dataset_cache_dir,
            effective_hf_split,
            num_samples,
        )
        text_iter_B = iter_hf_text(
            effective_hf_dataset_name,
            effective_dataset_cache_dir,
            effective_hf_split,
            num_samples,
        )
        get_next_A = lambda: next(text_iter_A)
        get_next_B = lambda: next(text_iter_B)
    else:
        # Use fixed prompt list
        prompts = [
            "The quick brown fox jumps over the lazy dog.",
            "Consistency Lens minimal test sentence.",
            "Hello world!",
            "Another short prompt for dumping activations.",
        ]
        get_next_A = lambda: random.choice(prompts)
        get_next_B = lambda: random.choice(prompts)

    log.info("Dumping %d activation samples to %s …", num_samples, out_dir)

    # -----------------------
    # Helper: perform dump for a single split
    # -----------------------
    def dump_split(split_name: str, out_path: Path, n_samples: int):
        MIN_TOKEN_IDX_INCLUSIVE = 5
        log.info(f"Dumping split '{split_name}' to {out_path} with {n_samples} samples …")
        out_path.mkdir(parents=True, exist_ok=True)

        text_iter_A = iter_hf_text(
            effective_hf_dataset_name,
            effective_dataset_cache_dir,
            split_name,
            n_samples,
        )
        text_iter_B = iter_hf_text(
            effective_hf_dataset_name,
            effective_dataset_cache_dir,
            split_name,
            n_samples,
        )
        get_next_A = lambda: next(text_iter_A)
        get_next_B = lambda: next(text_iter_B)

        num_batches = (n_samples + activation_dumper_cfg["batch_size"] - 1) // activation_dumper_cfg["batch_size"]
        saved = 0
        for batch_idx in tqdm(range(num_batches), desc=f"Split {split_name}"):
            current_batch_actual_size = min(activation_dumper_cfg["batch_size"], n_samples - saved)
            if current_batch_actual_size <= 0:
                break
            # -- reuse existing batch processing code via inner function --
            batch_txt_A = [get_next_A() for _ in range(current_batch_actual_size)]
            batch_txt_Ap = [get_next_B() for _ in range(current_batch_actual_size)]
            toks_A_batch = tokenizer(batch_txt_A, padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt").input_ids.to(device)
            toks_Ap_batch = tokenizer(batch_txt_Ap, padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                out_A_batch = model(toks_A_batch, output_hidden_states=True)
                out_Ap_batch = model(toks_Ap_batch, output_hidden_states=True)
            hidden_A_batch = out_A_batch.hidden_states[layer_idx]
            hidden_Ap_batch = out_Ap_batch.hidden_states[layer_idx]
            nonpad_len_A_b = (toks_A_batch != tokenizer.pad_token_id).sum(dim=1)
            nonpad_len_Ap_b = (toks_Ap_batch != tokenizer.pad_token_id).sum(dim=1)
            max_idx_shared_b = torch.minimum(nonpad_len_A_b, nonpad_len_Ap_b) - 1
            token_pos_b = torch.clamp((max_idx_shared_b + MIN_TOKEN_IDX_INCLUSIVE) // 2, min=0)
            batch_indices = torch.arange(current_batch_actual_size, device=device)
            A_selected_b = hidden_A_batch[batch_indices, token_pos_b].cpu().half()
            Ap_selected_b = hidden_Ap_batch[batch_indices, token_pos_b].cpu().half()
            batch_samples = []
            for i in range(current_batch_actual_size):
                sample = {
                    "A": A_selected_b[i],
                    "A_prime": Ap_selected_b[i],
                    "input_ids_A": toks_A_batch[i].cpu(),
                    "input_ids_A_prime": toks_Ap_batch[i].cpu(),
                    "token_pos": token_pos_b[i].item(),
                    "layer_idx": layer_idx,
                }
                batch_samples.append(sample)
            torch.save(batch_samples, out_path / f"shard_{batch_idx:05d}.pt")
            saved += current_batch_actual_size

    # ------------------------------------------------------------------
    # Main dumping loop over requested splits
    # ------------------------------------------------------------------
    for split_cfg in splits_to_dump:
        dump_split(split_cfg["name"], Path(split_cfg["output_dir"]), split_cfg["num_samples"])

    log.info("All requested splits dumped successfully.")
    return


if __name__ == "__main__":
    main()
