"""MVP extractor.

Generates a *tiny* activation cache suitable for unit / smoke tests.  It runs a
handful of short prompts through a small HF CausalLM and stores tuples:

    {"A": <tensor>, "A_prime": <tensor>, "input_ids_A": <tensor>, "input_ids_A_prime": <tensor>, "token_pos_A": int, "token_pos_A_prime": int}

Each item is written to ``output_dir/shards/shard_{idx:05d}.pt`` so the dataset can lazily
load them. A metadata file `output_dir/metadata.json` tracks shards and total samples.

This is *not* the full scalable extractor described in the README – it is the
minimum needed to unblock the end-to-end training loop on a laptop.
"""

from __future__ import annotations

import argparse
import logging
import random
import yaml
import json
from pathlib import Path
from typing import Iterator, Callable # Added Callable
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def iter_hf_text(dataset_name: str, cache_dir: str | None, split: str, num_samples: int) -> Iterator[str]:
    """Stream plain-text documents from a HuggingFace dataset until *num_samples* yielded."""
    if num_samples == 0: # Avoid issues if 0 samples requested
        return iter([])

    ds = load_dataset(dataset_name, split=split, streaming=True, cache_dir=cache_dir)
    count = 0
    for item in ds:
        if count >= num_samples: # Ensure we don't overshoot
            break
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
        action="store_true", # If flag is present, args.use_hf_dataset is True, else False.
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
        default=None, # If not set, YAML 'hf_split' or default 'train' is used.
        help="Override HF dataset split for streaming (e.g. train)",
    )

    # Optional validation dump in same run
    parser.add_argument("--val_hf_split", type=str, help="HF split for validation activations (e.g. validation)")
    parser.add_argument("--val_output_dir", type=str, help="Output directory for validation activations")
    parser.add_argument("--val_num_samples", type=int, help="Number of validation samples (defaults to main num_samples)")

    args = parser.parse_args()

    # Load YAML config
    with open(args.config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    activation_dumper_cfg = cfg["activation_dumper"]

    # Determine effective config: CLI > YAML > Default
    # Get base output_dir (CLI or YAML)
    base_output_dir_str = args.output_dir if args.output_dir is not None else activation_dumper_cfg["output_dir"]
    seq_len = args.seq_len if args.seq_len is not None else activation_dumper_cfg["seq_len"]
    num_samples = args.num_samples if args.num_samples is not None else activation_dumper_cfg["num_samples"]
    layer_idx = args.layer_idx if args.layer_idx is not None else cfg["layer_l"]
    
    # Construct final path with tokenizer_name and layer index
    tokenizer_name = cfg.get("tokenizer_name", cfg["model_name"])
    effective_output_dir = Path(Path(base_output_dir_str).parent / tokenizer_name / f"layer_{layer_idx}" / Path(base_output_dir_str).name)
    
    # effective_use_hf: True if CLI flag set, else YAML value (defaulting to False if not in YAML)
    effective_use_hf = args.use_hf_dataset or activation_dumper_cfg.get("use_hf_dataset", False)
    
    effective_hf_dataset_name = args.hf_dataset_name if args.hf_dataset_name is not None else activation_dumper_cfg.get("hf_dataset_name")
    effective_dataset_cache_dir = args.dataset_cache_dir if args.dataset_cache_dir is not None else activation_dumper_cfg.get("dataset_cache_dir")
    effective_hf_split = args.hf_split if args.hf_split is not None else activation_dumper_cfg.get("hf_split", "train")


    # ------------------------------------------------------------------
    # Build list of dataset splits to dump (e.g. train + optional validation)
    # ------------------------------------------------------------------
    splits_to_dump = [
        {
            "name": effective_hf_split, # This is the name of the split (e.g., "train")
            "output_dir": effective_output_dir,
            "num_samples": num_samples,
        }
    ]

    # Determine validation dump parameters either from CLI or YAML
    val_out_base_cfg = activation_dumper_cfg.get("val_output_dir")
    val_split_cfg = activation_dumper_cfg.get("val_hf_split", "validation") # Default val split name if not specified
    
    base_val_output_dir_str = None
    if args.val_output_dir:
        base_val_output_dir_str = args.val_output_dir
    elif val_out_base_cfg:
        base_val_output_dir_str = val_out_base_cfg

    if base_val_output_dir_str: # If a base validation output dir is specified (CLI or YAML)
        val_out_dir = Path(Path(base_val_output_dir_str).parent / tokenizer_name / f"layer_{layer_idx}" / Path(base_val_output_dir_str).name)
        val_split = args.val_hf_split if args.val_hf_split else val_split_cfg
        # Default val_num_samples to main num_samples if not specified
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    log.info(f"Layer index: {layer_idx}")
    log.info(f"Output directories will include layer: e.g., {effective_output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    log.info(f"Tokenizer name: {tokenizer_name}")
    # Some tiny models (GPT-2) miss an explicit pad_token.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # log.info(f"Tokenizer: {tokenizer}") # This can be very verbose, consider removing or reducing.
    log.info(f"Model name: {cfg['model_name']}")

    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"])
    # Ensure embedding & LM head match tokenizer size
    if model.get_input_embeddings().weight.size(0) != tokenizer.vocab_size:
        model.resize_token_embeddings(tokenizer.vocab_size)
        log.info(f"Resized model token embeddings to {tokenizer.vocab_size}")
    model.to(device)
    model.eval()
    
    # Fixed prompt list for non-HF mode, defined once in main scope
    fixed_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Consistency Lens minimal test sentence.",
        "Hello world!",
        "Another short prompt for dumping activations.",
    ]

    # -----------------------
    # Helper: perform dump for a single split
    # -----------------------
    def dump_split(split_name_arg: str, out_path: Path, n_samples_for_split: int):
        # Minimum token index (inclusive) to consider for activation, from start of sequence.
        MIN_TOKEN_IDX_INCLUSIVE = 5 
        log.info(f"Dumping split '{split_name_arg}' to {out_path} with {n_samples_for_split} samples …")
        out_path.mkdir(parents=True, exist_ok=True)

        # Initialize metadata storage for this split
        metadata = {
            "total_samples": 0,
            "shards": [],
            "config": { # Log key parameters for this split dump
                "split_name": split_name_arg,
                "num_samples": n_samples_for_split,
                "seq_len": seq_len,
                "layer_idx": layer_idx,
                "model_name": cfg["model_name"],
                "tokenizer_name": tokenizer_name,
                "use_hf_dataset": effective_use_hf,
                "hf_dataset_name": effective_hf_dataset_name if effective_use_hf else None,
                "dataset_cache_dir": effective_dataset_cache_dir if effective_use_hf else None,
                "min_token_idx_inclusive": MIN_TOKEN_IDX_INCLUSIVE,
                "seed": args.seed,
            }
        }
        
        batch_size_cfg = activation_dumper_cfg["batch_size"]
        
        num_iterator_items_needed: int
        if effective_use_hf:
            if n_samples_for_split == 0:
                num_iterator_items_needed = 0
            elif batch_size_cfg == 1:
                num_iterator_items_needed = n_samples_for_split * 2
            else: # batch_size_cfg > 1
                # If the last batch (or any batch that becomes size 1 due to n_samples_for_split)
                # has exactly one sample, it needs an extra item from the iterator.
                if n_samples_for_split > 0 and (n_samples_for_split % batch_size_cfg == 1):
                    num_iterator_items_needed = n_samples_for_split + 1
                else:
                    num_iterator_items_needed = n_samples_for_split
        else:
            # For fixed_prompts, this count isn't strictly used as random.choice doesn't consume an iterator.
            num_iterator_items_needed = n_samples_for_split # Placeholder, not strictly enforced for fixed_prompts

        get_next_text_fn: Callable[[], str]
        if effective_use_hf:
            text_iter = iter_hf_text(
                effective_hf_dataset_name,
                effective_dataset_cache_dir,
                split_name_arg, 
                num_iterator_items_needed, # Use the calculated amount
            )
            get_next_text_fn = lambda: next(text_iter)
        else:
            # Use fixed prompt list (defined in main scope)
            get_next_text_fn = lambda: random.choice(fixed_prompts)


        num_batches = (n_samples_for_split + batch_size_cfg - 1) // batch_size_cfg
        saved_samples_count = 0
        for batch_idx in tqdm(range(num_batches), desc=f"Split {split_name_arg}"):
            current_batch_actual_size = min(batch_size_cfg, n_samples_for_split - saved_samples_count)
            if current_batch_actual_size <= 0:
                break
            
            batch_txt_A = []
            batch_txt_Ap = []
            
            # Strategy for selecting A and A_prime (Ap)
            # If batch size allows ( > 1), A_prime is a different item from the same base batch.
            # Otherwise (batch size == 1, or last partial batch is size 1), A and A_prime are drawn independently.
            if batch_size_cfg > 1 and current_batch_actual_size > 1:
                # Fetch a single batch of base texts
                batch_base_texts = [get_next_text_fn() for _ in range(current_batch_actual_size)]
                for i in range(current_batch_actual_size):
                    text_A = batch_base_texts[i]
                    
                    # Select A_prime from a different index in batch_base_texts
                    possible_j_indices = [j for j in range(current_batch_actual_size) if j != i]
                    # This list is guaranteed non-empty because current_batch_actual_size > 1.
                    chosen_j = random.choice(possible_j_indices)
                    text_Ap = batch_base_texts[chosen_j]
                    
                    batch_txt_A.append(text_A)
                    batch_txt_Ap.append(text_Ap)
            else: 
                # Handles:
                # 1. batch_size_cfg == 1 (so current_batch_actual_size is always 1)
                # 2. batch_size_cfg > 1 BUT current_batch_actual_size == 1 (last partial batch is size 1)
                # In these cases, A and A_prime must be sourced independently to ensure they can be different.
                for _ in range(current_batch_actual_size): # Loop runs current_batch_actual_size times
                    text_A = get_next_text_fn()
                    text_Ap = get_next_text_fn() # Get a second, independent text

                    # For fixed prompts, try harder to make them different if there are multiple unique prompts
                    if not effective_use_hf and len(set(fixed_prompts)) > 1 and text_A == text_Ap:
                        # Retry fetching A_prime until it's different from A.
                        # This ensures that if text_A and text_Ap are drawn the same, and other options exist, we re-draw text_Ap.
                        while text_Ap == text_A: 
                            text_Ap = random.choice(fixed_prompts) # Effectively get_next_text_fn() again for fixed prompts
                    
                    batch_txt_A.append(text_A)
                    batch_txt_Ap.append(text_Ap)
            
            # thread-parallel tokenisation + single call
            all_texts = batch_txt_A + batch_txt_Ap
            enc = tokenizer(
                all_texts,
                padding="max_length",
                truncation=True,
                max_length=seq_len,
                return_tensors="pt",
            )
            toks_all = enc.input_ids.to(device)
            toks_A_batch, toks_Ap_batch = toks_all.split(current_batch_actual_size, dim=0)
            
            with torch.no_grad():
                out_A_batch = model(toks_A_batch, output_hidden_states=True)
                out_Ap_batch = model(toks_Ap_batch, output_hidden_states=True)
            
            hidden_A_batch = out_A_batch.hidden_states[layer_idx]
            hidden_Ap_batch = out_Ap_batch.hidden_states[layer_idx]
            
            nonpad_len_A_b = (toks_A_batch != tokenizer.pad_token_id).sum(dim=1)
            nonpad_len_Ap_b = (toks_Ap_batch != tokenizer.pad_token_id).sum(dim=1)

            # Calculate token positions for A
            # upper_bound_X_b is the last valid index in the sequence (length - 1)
            upper_bound_A_b = nonpad_len_A_b - 1 
            # Determine the earliest token index to consider for A.
            # It should be at least MIN_TOKEN_IDX_INCLUSIVE, but not exceed sequence bounds or be negative.
            _temp_min_idx_A = torch.full_like(upper_bound_A_b, MIN_TOKEN_IDX_INCLUSIVE)
            _temp_min_idx_A = torch.minimum(_temp_min_idx_A, upper_bound_A_b) # Cap by sequence length
            effective_min_idx_A_b = torch.maximum(_temp_min_idx_A, torch.zeros_like(upper_bound_A_b)) # Ensure non-negative
            # Select token position as midpoint of [effective_min_idx, upper_bound]
            token_pos_A_b = (effective_min_idx_A_b + upper_bound_A_b) // 2
            # Fallback to 0 if calculation results < 0 (e.g. for empty sequence where upper_bound is -1)
            token_pos_A_b = torch.maximum(token_pos_A_b, torch.zeros_like(token_pos_A_b)) 

            # Calculate token positions for A_prime (Ap) independently using the same logic
            upper_bound_Ap_b = nonpad_len_Ap_b - 1
            _temp_min_idx_Ap = torch.full_like(upper_bound_Ap_b, MIN_TOKEN_IDX_INCLUSIVE)
            _temp_min_idx_Ap = torch.minimum(_temp_min_idx_Ap, upper_bound_Ap_b)
            effective_min_idx_Ap_b = torch.maximum(_temp_min_idx_Ap, torch.zeros_like(upper_bound_Ap_b))
            token_pos_Ap_b = (effective_min_idx_Ap_b + upper_bound_Ap_b) // 2
            token_pos_Ap_b = torch.maximum(token_pos_Ap_b, torch.zeros_like(token_pos_Ap_b))

            batch_indices = torch.arange(current_batch_actual_size, device=device)
            A_selected_b = hidden_A_batch[batch_indices, token_pos_A_b].cpu().half()
            Ap_selected_b = hidden_Ap_batch[batch_indices, token_pos_Ap_b].cpu().half()
            
            batch_samples_to_save = []
            for i in range(current_batch_actual_size):
                sample = {
                    "A": A_selected_b[i],
                    "A_prime": Ap_selected_b[i],
                    "input_ids_A": toks_A_batch[i].cpu(),
                    "input_ids_A_prime": toks_Ap_batch[i].cpu(),
                    "token_pos_A": token_pos_A_b[i].item(),
                    "token_pos_A_prime": token_pos_Ap_b[i].item(),
                    "layer_idx": layer_idx,
                }
                batch_samples_to_save.append(sample)
            
            shard_filename = f"shard_{batch_idx:05d}.pt"
            torch.save(batch_samples_to_save, out_path / shard_filename)
            
            # Update metadata
            metadata["shards"].append({
                "name": shard_filename,
                "num_samples": current_batch_actual_size
            })
            metadata["total_samples"] += current_batch_actual_size
            
            saved_samples_count += current_batch_actual_size

        # Save metadata to JSON file for this split
        metadata_path = out_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        log.info(f"Saved metadata for split '{split_name_arg}' to {metadata_path}")

    # ------------------------------------------------------------------
    # Main dumping loop over requested splits
    # ------------------------------------------------------------------
    for split_cfg_item in splits_to_dump:
        dump_split(
            split_cfg_item["name"], 
            Path(split_cfg_item["output_dir"]), # Ensure it's a Path object
            split_cfg_item["num_samples"]
        )

    log.info("All requested splits dumped successfully.")
    return


if __name__ == "__main__":
    main()
