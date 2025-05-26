"""Multi-GPU optimized activation dumper for 8x H100 nodes.

Distributes model inference across multiple GPUs for efficient activation extraction.
Includes layer labeling in output directories and supports distributed processing.
"""

from __future__ import annotations

import logging
import random
import json
from pathlib import Path
from typing import Iterator, Callable
import os
import socket
import time
from types import SimpleNamespace

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

# Enable tokenizer parallelism within each process (we handle multi-process coordination)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Ensure PyTorch thread count honors OMP configuration (helpful for tokenization CPU usage)
try:
    _req_threads = int(os.environ.get("OMP_NUM_THREADS", "0"))
    if _req_threads > 0:
        torch.set_num_threads(_req_threads)
        # Force HF tokenizers to use requested threads
        os.environ["RAYON_NUM_THREADS"] = str(_req_threads)
        # Also set for potential nested parallelism
        os.environ["MKL_NUM_THREADS"] = str(_req_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(_req_threads)
        os.environ["OMP_NUM_THREADS"] = str(_req_threads)  # Ensure it's set
except Exception:
    pass


def get_project_root() -> Path:
    """Get the project root directory (consistency-lens folder)."""
    # This script is in consistency-lens/scripts/, so go up one level
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    return project_root


def resolve_path(path_str: str) -> Path:
    """Resolve a path relative to the project root if it's a relative path."""
    path = Path(path_str)
    if not path.is_absolute():
        # Make it relative to project root
        project_root = get_project_root()
        return project_root / path
    return path


def setup_distributed():
    """Initialize distributed processing."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank_env = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank = 0
        world_size = 1
        local_rank_env = 0

    num_cuda = torch.cuda.device_count()
    if num_cuda == 0:
        effective_local_rank = -1
    else:
        effective_local_rank = local_rank_env % num_cuda

    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        if effective_local_rank >= 0:
            torch.cuda.set_device(effective_local_rank)

    return rank, world_size, effective_local_rank


def cleanup_distributed():
    """Clean up distributed processing."""
    if dist.is_initialized():
        dist.destroy_process_group()


def iter_hf_text_distributed(
    dataset_name: str, 
    cache_dir: str | None, 
    split: str, 
    num_samples: int | None,
    rank: int,
    world_size: int
) -> Iterator[str]:
    """Stream text from HuggingFace dataset, distributed across ranks."""
    if num_samples == 0:
        return iter([])
    
    # If num_samples is None, process entire dataset
    if num_samples is None:
        # For streaming, we'll process until exhausted
        start_idx = None
        end_idx = None
    else:
        # Each rank processes a subset of samples
        samples_per_rank = (num_samples + world_size - 1) // world_size
        start_idx = rank * samples_per_rank
        end_idx = min(start_idx + samples_per_rank, num_samples)
    
    ds = load_dataset(dataset_name, split=split, streaming=True, cache_dir=cache_dir)
    count = 0
    global_count = 0
    
    for item in ds:
        # For distributed processing when num_samples is None
        if num_samples is None:
            # Each rank processes every world_size-th item
            if global_count % world_size == rank:
                text: str | None = None
                for v in item.values():
                    if isinstance(v, str):
                        text = v
                        break
                if text is not None:
                    yield text
                    count += 1
        else:
            # Original logic for fixed num_samples
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


def iter_pretokenized_distributed(
    dataset_path: str,
    split: str,
    num_samples: int | None,
    rank: int,
    world_size: int
) -> Iterator[dict]:
    """Stream pre-tokenized data, distributed across ranks."""
    
    # Load the dataset
    dataset = load_from_disk(f"{dataset_path}/{split}")
    
    if num_samples is None:
        # Process entire dataset
        for idx, item in enumerate(dataset):
            if idx % world_size == rank:
                yield item
    else:
        # Process fixed number
        samples_per_rank = (num_samples + world_size - 1) // world_size
        start_idx = rank * samples_per_rank
        end_idx = min(start_idx + samples_per_rank, num_samples, len(dataset))
        
        for idx in range(start_idx, end_idx):
            yield dataset[idx]


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra-powered multi-GPU activation dumper."""
    
    # Build args namespace from Hydra config for compatibility
    args = SimpleNamespace()
    
    # Extract activation_dumper config
    activation_dumper_cfg = cfg.activation_dumper
    
    # Command-line overrides (these are automatically handled by Hydra if specified)
    args.output_dir = activation_dumper_cfg.get('output_dir')
    args.num_samples = activation_dumper_cfg.get('num_samples', -1)
    args.seq_len = activation_dumper_cfg.get('seq_len', 64)
    args.layer_idx = cfg.get('layer_l', 5)
    args.seed = cfg.get('seed', 0)
    
    # Multi-GPU options
    args.model_parallel = cfg.get('model_parallel', False)
    args.pipeline_parallel_size = cfg.get('pipeline_parallel_size', 1)
    
    # HuggingFace dataset options
    args.hf_dataset_name = activation_dumper_cfg.get('hf_dataset_name')
    args.use_hf_dataset = activation_dumper_cfg.get('use_hf_dataset', False)
    args.use_pretokenized = activation_dumper_cfg.get('use_pretokenized', False)
    args.pretokenized_path = activation_dumper_cfg.get('pretokenized_path')
    args.dataset_cache_dir = activation_dumper_cfg.get('dataset_cache_dir')
    args.hf_split = activation_dumper_cfg.get('hf_split', 'train')
    
    # Validation split options
    args.val_hf_split = activation_dumper_cfg.get('val_hf_split')
    args.val_output_dir = activation_dumper_cfg.get('val_output_dir')
    args.val_num_samples = activation_dumper_cfg.get('val_num_samples')
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    
    # Convert OmegaConf to dict for compatibility with existing code
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Determine effective config
    base_output_dir_str = args.output_dir if args.output_dir is not None else activation_dumper_cfg['output_dir']
    seq_len = args.seq_len if args.seq_len is not None else activation_dumper_cfg['seq_len']
    # If num_samples not provided or is -1, we'll process the entire dataset
    num_samples_cfg = activation_dumper_cfg.get("num_samples", -1)
    if args.num_samples is not None:
        num_samples = None if args.num_samples == -1 else args.num_samples
    else:
        num_samples = None if num_samples_cfg == -1 else num_samples_cfg
    layer_idx = args.layer_idx if args.layer_idx is not None else cfg['layer_l']
    
    # Build output directory with model and layer information
    # Start with the base path from config and add model/layer structure
    base_path = resolve_path(base_output_dir_str)
    model_name_clean = cfg["model_name"].replace("/", "_")
    
    # Insert model and layer info into the path
    # If the path ends with _train or _test, preserve that
    path_parts = base_path.parts
    if path_parts[-1].endswith(('_train', '_test', '_val')):
        dataset_suffix = path_parts[-1]
        parent_path = Path(*path_parts[:-1])
        effective_output_dir = parent_path / model_name_clean / f"layer_{layer_idx}" / dataset_suffix
    else:
        effective_output_dir = base_path.parent / model_name_clean / f"layer_{layer_idx}" / base_path.name
    
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
    
    device = torch.device(f"cuda:{local_rank}") if local_rank >= 0 else torch.device("cpu")
    
    if rank == 0:
        log.info(f"World size: {world_size}, using device: {device}")
        log.info(f"CPU threads per process: torch.get_num_threads()={torch.get_num_threads()}, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'not set')}")
        log.info(f"Tokenizer parallelism: TOKENIZERS_PARALLELISM={os.environ.get('TOKENIZERS_PARALLELISM')}, RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS', 'not set')}")
        log.info(f"Tokenizer name: {tokenizer_name}")
        log.info(f"Model name: {cfg['model_name']}")
        log.info(f"Layer index: {layer_idx}")
        log.info(f"Output directory will include layer: {effective_output_dir}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test tokenizer parallelism on rank 0
    if rank == 0:
        test_texts = ["Test sentence for thread verification"] * 100
        import time as _time
        _start = _time.time()
        _ = tokenizer(test_texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        _elapsed = _time.time() - _start
        log.info(f"Tokenizer test: 100 samples in {_elapsed:.3f}s = {100/_elapsed:.0f} samples/sec")
        log.info(f"Actual PyTorch threads in use: {torch.get_num_threads()}")
    
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
        # Build validation output directory with model and layer information
        base_val_path = resolve_path(base_val_output_dir_str)
        
        # Insert model and layer info into the path
        val_path_parts = base_val_path.parts
        if val_path_parts[-1].endswith(('_train', '_test', '_val')):
            val_dataset_suffix = val_path_parts[-1]
            val_parent_path = Path(*val_path_parts[:-1])
            val_out_dir = val_parent_path / model_name_clean / f"layer_{layer_idx}" / val_dataset_suffix
        else:
            val_out_dir = base_val_path.parent / model_name_clean / f"layer_{layer_idx}" / base_val_path.name
        val_split = args.val_hf_split if args.val_hf_split else val_split_cfg
        val_samples_cfg = activation_dumper_cfg.get("val_num_samples", -1)
        if args.val_num_samples is not None:
            val_samples = None if args.val_num_samples == -1 else args.val_num_samples
        else:
            val_samples = None if val_samples_cfg == -1 else val_samples_cfg
        splits_to_dump.append({
            "name": val_split,
            "output_dir": val_out_dir,
            "num_samples": val_samples,
        })
    
    # Dump helper function adapted for distributed
    def dump_split_distributed(split_name_arg: str, out_path: Path, n_samples_for_split: int):
        MIN_TOKEN_IDX_INCLUSIVE = 5
        
        if rank == 0:
            if n_samples_for_split is None:
                log.info(f"Dumping split '{split_name_arg}' to {out_path} - processing ENTIRE dataset across {world_size} GPUs...")
            else:
                log.info(f"Dumping split '{split_name_arg}' to {out_path} with {n_samples_for_split} samples across {world_size} GPUs...")
            out_path.mkdir(parents=True, exist_ok=True)
        
        # Synchronize after directory creation
        if world_size > 1:
            dist.barrier()
        
        # Calculate samples per rank
        if n_samples_for_split is None:
            # When processing entire dataset, we don't know exact count
            rank_num_samples = None
        else:
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
        
        batch_size_cfg = activation_dumper_cfg.get("batch_size", 256)  # Match single-GPU default
        
        # Adapt batch size based on model size if not explicitly set
        model_params = sum(p.numel() for p in model.parameters())
        if rank == 0:
            log.info(f"Model has {model_params/1e6:.1f}M parameters")
            if model_params < 100e6:  # < 100M params
                suggested_batch = 128
            elif model_params < 1e9:  # < 1B params
                suggested_batch = 256
            else:  # >= 1B params
                suggested_batch = 512
            
            if batch_size_cfg > suggested_batch * 2:
                log.warning(f"Batch size {batch_size_cfg} may be too large for {model_params/1e6:.1f}M model. Consider {suggested_batch}")
        
        # Setup data iterator
        use_pretokenized = args.use_pretokenized
        pretokenized_path = args.pretokenized_path or f"data/pretokenized/{effective_hf_dataset_name.replace('/', '_')}"
        
        if use_pretokenized:
            # Use pre-tokenized data - much faster!
            if rank == 0:
                log.info(f"Using pre-tokenized data from: {pretokenized_path}")
            pretok_iter = iter_pretokenized_distributed(
                pretokenized_path,
                split_name_arg,
                n_samples_for_split,
                rank,
                world_size
            )
            get_next_data_fn = lambda: next(pretok_iter, None)
            data_type = "pretokenized"
        else:
            # Original text-based approach
            get_next_text_fn: Callable[[], str | None]
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
                # For fixed prompts, we never return None
                get_next_text_fn = lambda: random.choice(fixed_prompts)
            get_next_data_fn = get_next_text_fn
            data_type = "text"
        
        if rank_num_samples is None:
            # We'll process until the iterator is exhausted
            num_batches = float('inf')
        else:
            num_batches = (rank_num_samples + batch_size_cfg - 1) // batch_size_cfg
        saved_samples_count = 0
        
        # Create rank-specific subdirectory
        rank_out_path = out_path / f"rank_{rank}"
        rank_out_path.mkdir(parents=True, exist_ok=True)
        
        batch_idx = 0
        pbar = tqdm(
            desc=f"Rank {rank} - {split_name_arg}",
            disable=(rank != 0),
            unit="batch",
            unit_scale=False,
            total=int(num_batches) if num_batches != float('inf') else None,
            bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )
        batch_start_time = time.time()
        last_update_time = time.time()
        samples_since_update = 0
        
        while True:
            if rank_num_samples is not None:
                current_batch_actual_size = min(batch_size_cfg, rank_num_samples - saved_samples_count)
                if current_batch_actual_size <= 0:
                    break
            else:
                current_batch_actual_size = batch_size_cfg
            
            batch_txt_A = []
            batch_txt_Ap = []
            batch_tokens_A = []
            batch_tokens_Ap = []
            
            # Collect texts for batch
            if use_pretokenized:
                # Direct token loading
                for _ in range(current_batch_actual_size):
                    data_A = get_next_data_fn()
                    data_Ap = get_next_data_fn()
                    
                    if data_A is None or data_Ap is None:
                        break
                    
                    batch_tokens_A.append(torch.tensor(data_A["input_ids"]))
                    batch_tokens_Ap.append(torch.tensor(data_Ap["input_ids"]))
                
                current_batch_actual_size = len(batch_tokens_A)
                if current_batch_actual_size == 0:
                    break  # No more data
            else:
                # Original text-based collection
                if batch_size_cfg > 1 and current_batch_actual_size > 1:
                    batch_base_texts = []
                    for _ in range(current_batch_actual_size):
                        text = get_next_data_fn()
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
                        text_A = get_next_data_fn()
                        text_Ap = get_next_data_fn()
                        
                        if text_A is None or text_Ap is None:
                            break
                        
                        if not effective_use_hf and len(set(fixed_prompts)) > 1 and text_A == text_Ap:
                            while text_Ap == text_A:
                                text_Ap = random.choice(fixed_prompts)
                        
                        batch_txt_A.append(text_A)
                        batch_txt_Ap.append(text_Ap)
            
            # Check if we have any data
            if use_pretokenized:
                if not batch_tokens_A:
                    break
                # current_batch_actual_size already updated above
            else:
                if not batch_txt_A:
                    break
                current_batch_actual_size = len(batch_txt_A)
            
            # Tokenize
            if use_pretokenized:
                # Already have tokens, just stack and move to device
                toks_A_batch = torch.stack(batch_tokens_A).to(device)
                toks_Ap_batch = torch.stack(batch_tokens_Ap).to(device)
            else:
                # Tokenize text
                all_texts = batch_txt_A + batch_txt_Ap
                # Try to use multiple threads for tokenization
                if hasattr(tokenizer, "_tokenizer"):
                    # For fast tokenizers, try to set parallelism
                    try:
                        tokenizer._tokenizer.enable_truncation(seq_len)
                        tokenizer._tokenizer.enable_padding(length=seq_len)
                    except:
                        pass
                
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
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
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
            samples_since_update += current_batch_actual_size
            batch_idx += 1
            pbar.update(1)
            
            # Update detailed stats every 5 batches
            if batch_idx % 5 == 0:
                current_time = time.time()
                elapsed = time.time() - batch_start_time
                instant_elapsed = current_time - last_update_time
                
                # Calculate rates
                avg_samples_per_sec = saved_samples_count / elapsed if elapsed > 0 else 0
                instant_samples_per_sec = samples_since_update / instant_elapsed if instant_elapsed > 0 else 0
                
                # Estimate total samples and ETA
                if rank_num_samples is not None:
                    pct_complete = (saved_samples_count / rank_num_samples) * 100
                    eta_seconds = (rank_num_samples - saved_samples_count) / instant_samples_per_sec if instant_samples_per_sec > 0 else 0
                    eta_str = f"{int(eta_seconds//60)}m{int(eta_seconds%60)}s"
                else:
                    pct_complete = 0
                    eta_str = "unknown"
                
                pbar.set_postfix({
                    "batch": batch_size_cfg,
                    "avg_s/s": f"{avg_samples_per_sec:.0f}",
                    "cur_s/s": f"{instant_samples_per_sec:.0f}",
                    "done": f"{saved_samples_count:,}",
                    "%": f"{pct_complete:.1f}" if pct_complete > 0 else "?",
                    "ETA": eta_str
                })
                
                last_update_time = current_time
                samples_since_update = 0
            
            # If we have a fixed number of samples and reached it, break
            if rank_num_samples is not None and saved_samples_count >= rank_num_samples:
                break
        
        pbar.close()
        
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