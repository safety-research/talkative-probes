"""Multi-GPU optimized activation dumper for 8x H100 nodes.

Distributes model inference across multiple GPUs for efficient activation extraction.
Includes layer labeling in output directories and supports distributed processing.
"""

from __future__ import annotations

import argparse
import logging
import random
import yaml
import json
from pathlib import Path
from typing import Iterator, Callable
import os
import socket

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings in multi-process


def setup_distributed():
    """Initialize distributed processing."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed processing."""
    if dist.is_initialized():
        dist.destroy_process_group()


def iter_hf_text_distributed(
    dataset_name: str, 
    cache_dir: str | None, 
    split: str, 
    num_samples: int,
    rank: int,
    world_size: int
) -> Iterator[str]:
    """Stream text from HuggingFace dataset, distributed across ranks."""
    if num_samples == 0:
        return iter([])
    
    # Each rank processes a subset of samples
    samples_per_rank = (num_samples + world_size - 1) // world_size
    start_idx = rank * samples_per_rank
    end_idx = min(start_idx + samples_per_rank, num_samples)
    
    ds = load_dataset(dataset_name, split=split, streaming=True, cache_dir=cache_dir)
    count = 0
    global_count = 0
    
    for item in ds:
        if global_count >= end_idx:
            break
        
        if global_count >= start_idx:
            text: str | None = None
            for v in item.values():
                if isinstance(v, str):
                    text = v
                    break
            if text is not None:
                yield text
                count += 1
        
        global_count += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-GPU optimized activation dumper")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--num_samples", type=int, help="Override number of samples")
    parser.add_argument("--seq_len", type=int, help="Override sequence length")
    parser.add_argument("--layer_idx", type=int, help="Override layer index")
    parser.add_argument("--config_path", type=str, default="consistency-lens/config/lens_simple.yaml")
    parser.add_argument("--seed", type=int, default=0)
    
    # Multi-GPU options
    parser.add_argument("--model_parallel", action="store_true", help="Use model parallelism across GPUs")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1, help="Pipeline parallelism size")
    
    # HuggingFace dataset options
    parser.add_argument("--hf_dataset_name", type=str, help="Override HF dataset")
    parser.add_argument("--use_hf_dataset", action="store_true", help="Use HF dataset")
    parser.add_argument("--dataset_cache_dir", type=str, help="Override HF dataset cache dir")
    parser.add_argument("--hf_split", type=str, default=None, help="Override HF dataset split")
    
    # Validation split options
    parser.add_argument("--val_hf_split", type=str, help="HF split for validation")
    parser.add_argument("--val_output_dir", type=str, help="Output directory for validation")
    parser.add_argument("--val_num_samples", type=int, help="Number of validation samples")
    
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    
    # Load config
    with open(args.config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    activation_dumper_cfg = cfg["activation_dumper"]
    
    # Determine effective config
    base_output_dir_str = args.output_dir if args.output_dir is not None else activation_dumper_cfg["output_dir"]
    seq_len = args.seq_len if args.seq_len is not None else activation_dumper_cfg["seq_len"]
    num_samples = args.num_samples if args.num_samples is not None else activation_dumper_cfg["num_samples"]
    layer_idx = args.layer_idx if args.layer_idx is not None else cfg["layer_l"]
    
    # Include layer index in output directory name
    tokenizer_name = cfg.get("tokenizer_name", cfg["model_name"])
    effective_output_dir = Path(Path(base_output_dir_str).parent / tokenizer_name / f"layer_{layer_idx}" / Path(base_output_dir_str).name)
    
    effective_use_hf = args.use_hf_dataset or activation_dumper_cfg.get("use_hf_dataset", False)
    effective_hf_dataset_name = args.hf_dataset_name if args.hf_dataset_name is not None else activation_dumper_cfg.get("hf_dataset_name")
    effective_dataset_cache_dir = args.dataset_cache_dir if args.dataset_cache_dir is not None else activation_dumper_cfg.get("dataset_cache_dir")
    effective_hf_split = args.hf_split if args.hf_split is not None else activation_dumper_cfg.get("hf_split", "train")
    
    # Logging setup
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"%(asctime)s | Rank {rank} | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)
    
    # Set seeds with rank offset for diversity
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        log.info(f"World size: {world_size}, using device: {device}")
        log.info(f"Tokenizer name: {tokenizer_name}")
        log.info(f"Model name: {cfg['model_name']}")
        log.info(f"Layer index: {layer_idx}")
        log.info(f"Output directory will include layer: {effective_output_dir}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optimizations
    if rank == 0:
        log.info("Loading model...")
    
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": None,  # We'll handle device placement manually
    }
    
    # Load model efficiently
    if args.model_parallel and world_size > 1:
        # Model parallelism: split model across GPUs
        model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], **model_kwargs)
    else:
        # Data parallelism: replicate model on each GPU
        model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], **model_kwargs)
        model = model.to(device)
    
    # Ensure embedding size matches
    if model.get_input_embeddings().weight.size(0) != tokenizer.vocab_size:
        model.resize_token_embeddings(tokenizer.vocab_size)
        if rank == 0:
            log.info(f"Resized model token embeddings to {tokenizer.vocab_size}")
    
    model.eval()
    
    # Try to use Flash Attention if available
    try:
        model = model.to_bettertransformer()
        if rank == 0:
            log.info("Enabled BetterTransformer (Flash Attention)")
    except:
        pass
    
    # Fixed prompts for non-HF mode
    fixed_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Consistency Lens minimal test sentence.",
        "Hello world!",
        "Another short prompt for dumping activations.",
    ]
    
    # Build splits to dump
    splits_to_dump = [
        {
            "name": effective_hf_split,
            "output_dir": effective_output_dir,
            "num_samples": num_samples,
        }
    ]
    
    # Add validation split if specified
    val_out_base_cfg = activation_dumper_cfg.get("val_output_dir")
    val_split_cfg = activation_dumper_cfg.get("val_hf_split", "validation")
    
    base_val_output_dir_str = None
    if args.val_output_dir:
        base_val_output_dir_str = args.val_output_dir
    elif val_out_base_cfg:
        base_val_output_dir_str = val_out_base_cfg
    
    if base_val_output_dir_str:
        val_out_dir = Path(Path(base_val_output_dir_str).parent / tokenizer_name / f"layer_{layer_idx}" / Path(base_val_output_dir_str).name)
        val_split = args.val_hf_split if args.val_hf_split else val_split_cfg
        val_samples = args.val_num_samples if args.val_num_samples is not None else activation_dumper_cfg.get("val_num_samples", num_samples)
        splits_to_dump.append({
            "name": val_split,
            "output_dir": val_out_dir,
            "num_samples": val_samples,
        })
    
    # Dump helper function adapted for distributed
    def dump_split_distributed(split_name_arg: str, out_path: Path, n_samples_for_split: int):
        MIN_TOKEN_IDX_INCLUSIVE = 5
        
        if rank == 0:
            log.info(f"Dumping split '{split_name_arg}' to {out_path} with {n_samples_for_split} samples across {world_size} GPUs...")
            out_path.mkdir(parents=True, exist_ok=True)
        
        # Synchronize after directory creation
        if world_size > 1:
            dist.barrier()
        
        # Calculate samples per rank
        samples_per_rank = (n_samples_for_split + world_size - 1) // world_size
        rank_start_idx = rank * samples_per_rank
        rank_num_samples = min(samples_per_rank, n_samples_for_split - rank_start_idx)
        
        if rank_num_samples <= 0:
            return
        
        # Initialize metadata for this rank
        rank_metadata = {
            "rank": rank,
            "total_samples": 0,
            "shards": [],
            "config": {
                "split_name": split_name_arg,
                "num_samples": rank_num_samples,
                "seq_len": seq_len,
                "layer_idx": layer_idx,
                "model_name": cfg["model_name"],
                "tokenizer_name": tokenizer_name,
                "use_hf_dataset": effective_use_hf,
                "hf_dataset_name": effective_hf_dataset_name if effective_use_hf else None,
                "dataset_cache_dir": effective_dataset_cache_dir if effective_use_hf else None,
                "min_token_idx_inclusive": MIN_TOKEN_IDX_INCLUSIVE,
                "seed": args.seed + rank,
            }
        }
        
        batch_size_cfg = activation_dumper_cfg.get("batch_size", 8)
        
        # Setup text iterator
        get_next_text_fn: Callable[[], str]
        if effective_use_hf:
            text_iter = iter_hf_text_distributed(
                effective_hf_dataset_name,
                effective_dataset_cache_dir,
                split_name_arg,
                n_samples_for_split,
                rank,
                world_size
            )
            get_next_text_fn = lambda: next(text_iter, None)
        else:
            get_next_text_fn = lambda: random.choice(fixed_prompts)
        
        num_batches = (rank_num_samples + batch_size_cfg - 1) // batch_size_cfg
        saved_samples_count = 0
        
        # Create rank-specific subdirectory
        rank_out_path = out_path / f"rank_{rank}"
        rank_out_path.mkdir(parents=True, exist_ok=True)
        
        for batch_idx in tqdm(range(num_batches), desc=f"Rank {rank} - Split {split_name_arg}", disable=(rank != 0)):
            current_batch_actual_size = min(batch_size_cfg, rank_num_samples - saved_samples_count)
            if current_batch_actual_size <= 0:
                break
            
            batch_txt_A = []
            batch_txt_Ap = []
            
            # Collect texts for batch
            if batch_size_cfg > 1 and current_batch_actual_size > 1:
                batch_base_texts = []
                for _ in range(current_batch_actual_size):
                    text = get_next_text_fn()
                    if text is None:
                        break
                    batch_base_texts.append(text)
                
                if len(batch_base_texts) < current_batch_actual_size:
                    current_batch_actual_size = len(batch_base_texts)
                
                for i in range(current_batch_actual_size):
                    text_A = batch_base_texts[i]
                    possible_j_indices = [j for j in range(current_batch_actual_size) if j != i]
                    chosen_j = random.choice(possible_j_indices)
                    text_Ap = batch_base_texts[chosen_j]
                    
                    batch_txt_A.append(text_A)
                    batch_txt_Ap.append(text_Ap)
            else:
                for _ in range(current_batch_actual_size):
                    text_A = get_next_text_fn()
                    text_Ap = get_next_text_fn()
                    
                    if text_A is None or text_Ap is None:
                        break
                    
                    if not effective_use_hf and len(set(fixed_prompts)) > 1 and text_A == text_Ap:
                        while text_Ap == text_A:
                            text_Ap = random.choice(fixed_prompts)
                    
                    batch_txt_A.append(text_A)
                    batch_txt_Ap.append(text_Ap)
            
            if not batch_txt_A:
                break
            
            current_batch_actual_size = len(batch_txt_A)
            
            # Tokenize
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
            
            # Forward pass
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    out_A_batch = model(toks_A_batch, output_hidden_states=True)
                    out_Ap_batch = model(toks_Ap_batch, output_hidden_states=True)
            
            hidden_A_batch = out_A_batch.hidden_states[layer_idx]
            hidden_Ap_batch = out_Ap_batch.hidden_states[layer_idx]
            
            # Calculate positions
            nonpad_len_A_b = (toks_A_batch != tokenizer.pad_token_id).sum(dim=1)
            nonpad_len_Ap_b = (toks_Ap_batch != tokenizer.pad_token_id).sum(dim=1)
            
            upper_bound_A_b = nonpad_len_A_b - 1
            _temp_min_idx_A = torch.full_like(upper_bound_A_b, MIN_TOKEN_IDX_INCLUSIVE)
            _temp_min_idx_A = torch.minimum(_temp_min_idx_A, upper_bound_A_b)
            effective_min_idx_A_b = torch.maximum(_temp_min_idx_A, torch.zeros_like(upper_bound_A_b))
            token_pos_A_b = (effective_min_idx_A_b + upper_bound_A_b) // 2
            token_pos_A_b = torch.maximum(token_pos_A_b, torch.zeros_like(token_pos_A_b))
            
            upper_bound_Ap_b = nonpad_len_Ap_b - 1
            _temp_min_idx_Ap = torch.full_like(upper_bound_Ap_b, MIN_TOKEN_IDX_INCLUSIVE)
            _temp_min_idx_Ap = torch.minimum(_temp_min_idx_Ap, upper_bound_Ap_b)
            effective_min_idx_Ap_b = torch.maximum(_temp_min_idx_Ap, torch.zeros_like(upper_bound_Ap_b))
            token_pos_Ap_b = (effective_min_idx_Ap_b + upper_bound_Ap_b) // 2
            token_pos_Ap_b = torch.maximum(token_pos_Ap_b, torch.zeros_like(token_pos_Ap_b))
            
            batch_indices = torch.arange(current_batch_actual_size, device=device)
            A_selected_b = hidden_A_batch[batch_indices, token_pos_A_b].cpu().half()
            Ap_selected_b = hidden_Ap_batch[batch_indices, token_pos_Ap_b].cpu().half()
            
            # Save batch
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
            
            # Global batch index includes rank offset
            global_batch_idx = rank * 10000 + batch_idx
            shard_filename = f"shard_{global_batch_idx:08d}.pt"
            torch.save(batch_samples_to_save, rank_out_path / shard_filename)
            
            # Update metadata
            rank_metadata["shards"].append({
                "name": shard_filename,
                "num_samples": current_batch_actual_size
            })
            rank_metadata["total_samples"] += current_batch_actual_size
            
            saved_samples_count += current_batch_actual_size
        
        # Save rank-specific metadata
        rank_metadata_path = rank_out_path / "metadata.json"
        with open(rank_metadata_path, "w") as f:
            json.dump(rank_metadata, f, indent=4)
        
        # Synchronize and combine metadata on rank 0
        if world_size > 1:
            dist.barrier()
        
        if rank == 0:
            # Combine all rank metadata into global metadata
            global_metadata = {
                "total_samples": 0,
                "shards": [],
                "ranks": world_size,
                "config": rank_metadata["config"],
            }
            
            for r in range(world_size):
                rank_path = out_path / f"rank_{r}" / "metadata.json"
                if rank_path.exists():
                    with open(rank_path, "r") as f:
                        r_meta = json.load(f)
                    global_metadata["total_samples"] += r_meta["total_samples"]
                    for shard in r_meta["shards"]:
                        shard["rank"] = r
                        global_metadata["shards"].append(shard)
            
            # Save global metadata
            global_metadata_path = out_path / "metadata.json"
            with open(global_metadata_path, "w") as f:
                json.dump(global_metadata, f, indent=4)
            
            log.info(f"Saved global metadata for split '{split_name_arg}' to {global_metadata_path}")
            log.info(f"Total samples across all ranks: {global_metadata['total_samples']}")
    
    # Main dumping loop
    for split_cfg_item in splits_to_dump:
        dump_split_distributed(
            split_cfg_item["name"],
            Path(split_cfg_item["output_dir"]),
            split_cfg_item["num_samples"]
        )
    
    if rank == 0:
        log.info("All requested splits dumped successfully.")
    
    cleanup_distributed()


if __name__ == "__main__":
    main() 