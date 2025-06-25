#!/usr/bin/env python3
"""Training script for Consistency Lens MVP."""

import logging
import math
import os
import time
from collections import Counter, deque
from contextlib import contextmanager, nullcontext
from pathlib import Path
import sys
import argparse
import re
from datetime import datetime
import psutil
try:
    import GPUtil
except ImportError:
    GPUtil = None

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
# Enable TF32 for better performance on Ampere GPUs (A100, H100)
torch.set_float32_matmul_precision('high')
from lens.data.collate import collate
from lens.data.dataset import ActivationDataset
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.orig import OrigWrapper
from lens.training.loop import train_step
from torch.utils.data import DataLoader, random_split
from lens.training.optim import param_groups
from lens.utils.logging import init as log_init, log as log_metrics
from lens.training.schedules import (
    get_schedule_value, 
    get_lr_scheduler,
    parse_schedule_config, 
    parse_schedule_value, 
    resolve_schedule_at_step,
    get_schedule_value_for_logging,
    get_autocast_context,
    optimizer_step,
    parse_schedule_to_steps,
    spec_to_steps,
    should_unfreeze_any_component,
    apply_unfreeze_warmup,
    unfreeze_non_adapters,
    apply_gradient_scaling,
)
from lens.utils.checkpoint_manager import CheckpointManager
import yaml
from lens.evaluation.verbose_samples import process_and_print_verbose_batch_samples
from lens.evaluation.wandb_logger import verbose_samples_logger
from transformers import AutoTokenizer # Not used in this selection
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import math # For math.ceil
import torch # For torch.utils.data, torch.Generator
from torch.utils.data import DataLoader, random_split, Dataset, Subset # Explicit imports for clarity
from lens.data.collate import collate
from lens.data.dataset import ActivationDataset
import logging # For type hinting Logger
import torch.nn as nn
from lens.utils.embedding_remap import remap_embeddings
from types import SimpleNamespace
from datasets import load_from_disk, Dataset as HFDataset

from lens.data.on_the_fly_datasets import (
    InMemoryValidationDataset,
    RankInMemoryTrainingCache,
    _generate_activations_batched,
    _dataset_log_fn
)
from typing import Optional, Callable
from transformers import PreTrainedTokenizer

# Make _rank_log_fn a top-level function
def global_rank_log_fn(message: str, level: str = "info", current_rank: Optional[int] = None, logger_instance: Optional[logging.Logger] = None):
    """
    A picklable logging function that can be used by worker processes.
    """
    if logger_instance is None:
        # Fallback if no logger is passed, though it's better to pass one.
        # This might not integrate perfectly with Hydra's logging if used directly by workers without setup.
        logger_instance = logging.getLogger(__name__) # Or a specific name
        if not logger_instance.hasHandlers(): # Basic setup if no handlers
            handler = logging.StreamHandler()
            formatter = logging.Formatter(f"[Rank {current_rank if current_rank is not None else 'N/A'}] %(levelname)s | %(message)s")
            handler.setFormatter(formatter)
            logger_instance.addHandler(handler)
            logger_instance.setLevel(logging.INFO)


    prefix = f"[Rank {current_rank}] " if current_rank is not None else ""
    
    if level == "error":
        logger_instance.error(f"{prefix}{message}")
    elif level == "warning":
        logger_instance.warning(f"{prefix}{message}")
    else:
        logger_instance.info(f"{prefix}{message}")


def _resolve_schedule_to_steps(schedule_str: str | int, steps_per_epoch: int, log: logging.Logger, setting_name: str, grad_accum_steps: int) -> int:
    """Helper to parse schedule strings (e.g., "100s", "2e") into steps."""
    if isinstance(schedule_str, int):
        return schedule_str
    try:
        spec = parse_schedule_value(schedule_str)
        return spec_to_steps(spec, steps_per_epoch, grad_accum_steps)
    except Exception as e:
        log.warning(f"Failed to parse {setting_name} '{schedule_str}': {e}. Using raw value if integer, else raising error.")
        if isinstance(schedule_str, str) and schedule_str.isdigit():
            return int(schedule_str)
        raise ValueError(f"Invalid format for {setting_name}: '{schedule_str}'") from e


def _log_epoch_token_statistics(epoch_decoded_tokens: list[int], tokenizer: AutoTokenizer, current_epoch_num: int, step: int, log_interval: int, log: logging.Logger):
    """Logs statistics about token occurrences at the end of an epoch."""
    if not epoch_decoded_tokens:
        return

    token_counts = Counter(epoch_decoded_tokens)
    if not token_counts:
        return

    most_common_token_id, most_common_count = token_counts.most_common(1)[0]
    total_tokens_in_epoch = len(epoch_decoded_tokens)
    frequency = most_common_count / total_tokens_in_epoch

    if step % log_interval == 0:  # Check if current step aligns with log_interval for console logging
        log.info(
            f"Epoch {current_epoch_num} most common token: ID {most_common_token_id} = `{tokenizer.decode([most_common_token_id])}` "
            f"(Count: {most_common_count}/{total_tokens_in_epoch}, Freq: {frequency:.4f})"
        )
    
    log_metrics({
        "epoch_stats/most_common_token_id": most_common_token_id,
        "epoch_stats/most_common_token_count": most_common_count,
        "epoch_stats/most_common_token_freq": frequency,
        "epoch_stats/total_tokens_in_epoch": total_tokens_in_epoch,
    }, step=step) # Log with the current step, which marks the end of the epoch


def get_project_root() -> Path:
    """Get the project root directory (consistency-lens folder)."""
    # This script is in consistency-lens/scripts/, so go up one level
    script_dir = Path(__file__).parent.parent.absolute()
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


def extract_dataset_info(activation_dir: str) -> dict:
    """Extract model name, layer, and dataset info from activation directory path.
    
    Expected formats:
    - .../SimpleStories_SimpleStories-5M/layer_5/SimpleStories_train
    - .../dataset_name/model_name/layer_X/split_name/
    - .../SimpleStories_train (direct path)
    """
    parts = Path(activation_dir).parts
    info = {
        'model_name': None,
        'layer': None,
        'dataset': None,
        'split': None
    }
    
    # Find layer_X pattern
    for i, part in enumerate(parts):
        if part.startswith('layer_'):
            layer_match = re.match(r'layer_(\d+)', part)
            if layer_match:
                info['layer'] = int(layer_match.group(1))
                # Model name should be one level up
                if i > 0:
                    model_part = parts[i-1]
                    # Handle combined dataset_model names
                    if '_' in model_part:
                        # e.g., SimpleStories_SimpleStories-5M
                        dataset_prefix, model_name = model_part.split('_', 1)
                        info['dataset'] = dataset_prefix
                        info['model_name'] = model_name.replace('_', '/')
                    else:
                        info['model_name'] = model_part
                        # Dataset should be two levels up
                        if i > 1:
                            info['dataset'] = parts[i-2]
                
                # Split should be one level down or embedded in the name
                if i < len(parts) - 1:
                    split_part = parts[i+1]
                    # Handle names like SimpleStories_train
                    if '_' in split_part:
                        dataset_name, split = split_part.rsplit('_', 1)
                        if split in ['train', 'test', 'val', 'validation']:
                            info['split'] = split
                            if not info['dataset']:
                                info['dataset'] = dataset_name
                    else:
                        info['split'] = split_part
                break
    
    # Fallback: try to extract from the final directory name
    if not info['dataset'] and parts:
        final_part = parts[-1]
        if '_' in final_part:
            dataset_name, split = final_part.rsplit('_', 1)
            if split in ['train', 'test', 'val', 'validation']:
                info['dataset'] = dataset_name
                info['split'] = split
    
    return info


def get_system_metrics(device: torch.device) -> dict:
    """Get current system performance metrics."""
    metrics = {}
    
    # CPU metrics
    metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
    metrics['memory_percent'] = psutil.virtual_memory().percent
    
    # GPU metrics if available
    if device.type == 'cuda' and GPUtil is not None:
        try:
            gpus = GPUtil.getGPUs()
            if gpus and device.index is not None and device.index < len(gpus):
                gpu = gpus[device.index]
                metrics['gpu_utilization'] = gpu.load * 100
                metrics['gpu_memory_percent'] = gpu.memoryUtil * 100
                metrics['gpu_temperature'] = gpu.temperature
            elif gpus and len(gpus) > 0:
                # If no specific device index, use first GPU
                gpu = gpus[0]
                metrics['gpu_utilization'] = gpu.load * 100
                metrics['gpu_memory_percent'] = gpu.memoryUtil * 100
                metrics['gpu_temperature'] = gpu.temperature
        except:
            # GPUtil might fail in some environments
            pass
    
    return metrics


def format_time(seconds: float) -> str:
    """Format seconds into human readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def _get_hydra_config_name() -> str | None:
    """Attempts to get the config name from Hydra's context."""
    config_name = None
    try:
        hydra_cfg = HydraConfig.get()
        if hasattr(hydra_cfg, 'job') and hasattr(hydra_cfg.job, 'config_name'):
            config_name = hydra_cfg.job.config_name
        elif hasattr(hydra_cfg, 'runtime') and hasattr(hydra_cfg.runtime, 'choices'):
            config_name = hydra_cfg.runtime.choices.get('config_name', 'config')
    except Exception: # Broad exception as the original code does
        # HydraConfig might not be available or might fail in some contexts
        pass
    return config_name


def generate_run_name(config: dict, dataset_info: dict, resume_from: str = None, config_name: str = None, run_suffix: str = None) -> str:
    """Generate a descriptive run name based on config and dataset info."""
    components = []
    
    # Config name first (if provided and not default)
    if config_name and config_name != 'config':
        # Remove common suffixes
        clean_config = config_name.replace('_config', '').replace('.yaml', '')
        components.append(clean_config)
    
    # Dataset name (more important than model for experiments)
    if dataset_info['dataset']:
        dataset_name = dataset_info['dataset']
        # Handle common dataset names
        dataset_map = {
            'SimpleStories': 'SS',
            'openwebtext': 'OWT',
            'pile': 'Pile',
        }
        dataset_short = dataset_map.get(dataset_name, dataset_name)
        if len(dataset_short) > 8 and dataset_short not in dataset_map.values():
            # Abbreviate long names not in map
            dataset_short = ''.join([w[0].upper() for w in dataset_short.replace('_', ' ').replace('-', ' ').split()])
        components.append(dataset_short)
    
    # Model name (clearer abbreviation)
    if dataset_info['model_name']:
        model_name = dataset_info['model_name']
        # Common model mappings
        model_map = {
            'SimpleStories/SimpleStories-5M': '5M',
            'openai-community/gpt2': 'GPT2',
            'gpt2': 'GPT2',
            'gpt2-medium': 'GPT2-M',
            'gpt2-large': 'GPT2-L',
            'gpt2-xl': 'GPT2-XL',
        }
        model_short = model_map.get(model_name, model_name.split('/')[-1])
        components.append(model_short)
    
    # Layer
    if dataset_info['layer'] is not None:
        components.append(f"L{dataset_info['layer']}")

    if config['trainable_components']['encoder']['output_layer'] is not None:
        components.append(f"e{config['trainable_components']['encoder']['output_layer']}")
    
    # Freeze status (important for experiments)
    freeze_schedule = config.get('freeze_schedule', {})
    if freeze_schedule.get('enabled', False):
        # Progressive unfreeze
        unfreeze_at = freeze_schedule.get('unfreeze_at', '')
        if 'epoch' in str(unfreeze_at).lower():
            components.append('unfreeze')
        else:
            components.append('prog-unfreeze')
    else:
        # Check if base models are trainable
        decoder_cfg = config.get('trainable_components', {}).get('decoder', {})
        encoder_cfg = config.get('trainable_components', {}).get('encoder', {})
        if decoder_cfg.get('base_model', False) or encoder_cfg.get('base_model', False):
            components.append('full')
        else:
            components.append('frozen')
    
    # Key hyperparameters
    lr = config.get('learning_rate', 1e-4)
    components.append(f"lr{lr:.0e}".replace('e-0', 'e-').replace('e+0', 'e'))
    
    # Text width (decoder output length)
    t_text = config.get('t_text', 10)
    components.append(f"t{t_text}")
    
    # Training duration info
    num_epochs = config.get('num_train_epochs', 0)
    max_steps = config.get('max_train_steps', 0)
    if num_epochs > 0:
        components.append(f"{num_epochs}ep")
    elif max_steps > 0:
        if max_steps >= 1000:
            components.append(f"{max_steps//1000}k")
        else:
            components.append(f"{max_steps}s")
    
    # If resuming, add 'resume'
    if resume_from:
        components.append("resume")
    
    # Add timestamp (shorter format)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    components.append(timestamp)
    
    # Add suffix if provided
    if run_suffix:
        components.append(run_suffix)
    
    return "_".join(components)


def _prepare_dataloaders(
    config: dict,
    activation_dir: str,
    effective_val_activation_dir: str | None,
    max_train_samples_req: int | None,
    max_val_samples_req: int | None,
    log: logging.Logger,
    orig_model_for_gen: Optional["OrigWrapper"] = None,
    tokenizer_for_gen: Optional[PreTrainedTokenizer] = None,
    generation_device: Optional["torch.device"] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    samples_per_regeneration_cycle: Optional[int] = None
):
    on_the_fly_cfg = config.get('dataset', {}).get('on_the_fly', {'enabled': False})
    dataset_cfg = config.get('dataset', {})

    train_ds, val_ds = None, None

    if on_the_fly_cfg.get('enabled', False):
        _dataset_log_fn(log, "Preparing datasets for on-the-fly activation generation (Map-style Cache model).", rank=rank)
        
        pretok_path_config = on_the_fly_cfg.get('pretokenized_path') # Store in a variable for check

        if not all([orig_model_for_gen, 
                    pretok_path_config, # Use the variable here
                    generation_device is not None, rank is not None, world_size is not None]):
            missing_args_map = {
                    "orig_model_for_gen": orig_model_for_gen,
                    "pretok_path": pretok_path_config, 
                    "generation_device": generation_device, 
                    "rank": rank, 
                    "world_size": world_size,
                }
            missing_args = [ arg_name for arg_name, arg_val in missing_args_map.items() if arg_val is None ]
            error_msg = f"Missing required arguments for on-the-fly Map-style Cache: {missing_args}."
            _dataset_log_fn(log, error_msg, "error", rank=rank)
            raise ValueError(error_msg)
        
        # pretok_path is guaranteed to be non-None here due to the check above
        pretok_path: str = pretok_path_config # type: ignore 

        if tokenizer_for_gen: 
            val_split_name = dataset_cfg.get('val_split', 'validation')
            #num_val_samples_to_generate = on_the_fly_cfg.get('validation_samples_override', max_val_samples_req)
            num_val_samples_to_generate = max_val_samples_req
            #if num_val_samples_to_generate is None: 
            #    num_val_samples_to_generate = config.get('activation_dumper',{}).get('val_num_samples', 5000)
            
            if num_val_samples_to_generate > 0 :
                val_ds = InMemoryValidationDataset(
                    orig_model_for_gen=orig_model_for_gen, tokenizer=tokenizer_for_gen,
                    pretok_dataset_path=pretok_path, pretok_split_name=val_split_name, 
                    num_val_samples_to_generate=num_val_samples_to_generate,
                    on_the_fly_config=on_the_fly_cfg, generation_device=generation_device, 
                    rank=rank, world_size=world_size 
                )
                if len(val_ds) == 0 and num_val_samples_to_generate > 0: # Added check for num_val_samples_to_generate
                     _dataset_log_fn(log, f"InMemoryValidationDataset is empty after creation, despite requesting {num_val_samples_to_generate} samples. Check pretokenized data '{pretok_path}/{val_split_name}' and config.", "warning", rank=rank)
            else: 
                _dataset_log_fn(log, f"Validation samples to generate is {num_val_samples_to_generate}, so val_ds will be None.", "info", rank=rank)
                val_ds = None
        else:
            _dataset_log_fn(log, "tokenizer_for_gen not provided for on-the-fly validation; skipping val_ds.", "warning", rank=rank)
            val_ds = None
        
        train_split_name = dataset_cfg.get('train_split', 'train')
        
        initial_cache_fill_size = samples_per_regeneration_cycle
        # if initial_cache_fill_size is None or initial_cache_fill_size <= 0:
        #     initial_cache_fill_size = on_the_fly_cfg.get('training_initial_cache_size_per_rank', 100000)
        #     _dataset_log_fn(log, f"max_train_samples not specified or <=0 for on-the-fly. Using training_initial_cache_size_per_rank: {initial_cache_fill_size}", "info", rank=rank)
        
        train_ds = RankInMemoryTrainingCache(
            orig_model_for_gen=orig_model_for_gen, 
            pretok_dataset_path=pretok_path, 
            pretok_split_name=train_split_name,
            on_the_fly_config=on_the_fly_cfg, 
            generation_device=generation_device, 
            rank=rank, world_size=world_size, 
            initial_cache_size=initial_cache_fill_size, # Correctly passed
            logger=log
        )
        # The initial fill is now handled by RankInMemoryTrainingCache's __init__
        # No need for: train_ds.regenerate_cache() here.

        if len(train_ds) == 0 and initial_cache_fill_size > 0:
             _dataset_log_fn(log, f"On-the-fly training cache (RankInMemoryTrainingCache) is empty after initial population attempt. Training might not proceed if data source is exhausted or too small.", "warning", rank=rank)
        elif len(train_ds) > 0:
             _dataset_log_fn(log, f"On-the-fly training cache (RankInMemoryTrainingCache) initially populated with {len(train_ds)} samples.", "info", rank=rank)

    else: 
        _dataset_log_fn(log, "Loading datasets from disk (standard behavior).", rank=rank)
        train_dataset_path = Path(activation_dir)
        train_ds = ActivationDataset(train_dataset_path, max_samples=max_train_samples_req, desc="Loading train dataset")
        if len(train_ds) == 0 and (max_train_samples_req is None or max_train_samples_req > 0):
             _dataset_log_fn(log, "Training dataset from disk is empty.", "warning", rank=rank)

        if effective_val_activation_dir:
            val_dataset_path = Path(effective_val_activation_dir)
            val_ds = ActivationDataset(val_dataset_path, max_samples=max_val_samples_req, desc="Loading val dataset")
            if len(val_ds) == 0 and (max_val_samples_req is None or max_val_samples_req > 0):
                _dataset_log_fn(log, "Validation dataset from disk is empty.", "warning", rank=rank)
        else:
            val_ds = None
            
    return train_ds, val_ds

def run_validation_step(
    dec: nn.Module,
    enc: nn.Module,
    orig: OrigWrapper,
    val_loader: DataLoader,
    config: dict,
    tokenizer: AutoTokenizer, # Not used in this selection
    cached_prefix_ids: torch.Tensor | None,
    device: torch.device,
    current_step: int,
    current_epoch: int, # 0-based epoch
    max_steps: int,
    steps_per_epoch: int,
    log: logging.Logger
) -> dict:
    """Runs a validation step and returns a dictionary of metrics."""
    dec.eval()
    enc.eval()
    val_loss = val_mse = val_lm = val_kl = 0.0
    val_seen = 0
    
    t_text = config['t_text']
    lm_base_weight = config['lm_base_weight']
    kl_base_weight = config['kl_base_weight']
    entropy_weight = config['entropy_weight']

    with torch.no_grad():
        for vbatch in val_loader:
            vbatch = {k: v.to(device) for k, v in vbatch.items()}
            sch_args = {
                "tau": get_schedule_value(config['gumbel_tau_schedule'], current_step, max_steps,
                                         current_epoch, steps_per_epoch),
                "t_text": t_text,
                "alpha": get_schedule_value(config['alpha_schedule'], current_step, max_steps,
                                           current_epoch, steps_per_epoch),
                "lm_base_weight": lm_base_weight,
                "kl_base_weight": kl_base_weight,
                "entropy_weight": entropy_weight,
                "mse_weight": config.get('mse_weight', 0.0),
            }
            v_losses = train_step(vbatch, {"dec": dec, "enc": enc, "orig": orig}, sch_args,
                                 lm_loss_natural_prefix=config.get('lm_loss_natural_prefix'),
                                 tokenizer=tokenizer,
                                 cached_prefix_ids=cached_prefix_ids,
                                 resample_ablation=config.get('resample_ablation'))
            bsz = vbatch["A"].size(0)
            val_loss += v_losses["total"].item() * bsz
            val_mse  += v_losses["mse"].item()   * bsz
            val_lm   += v_losses["lm"].item()    * bsz
            val_kl   += v_losses["kl"].item()    * bsz
            val_seen += bsz
            
    avg_val_loss = val_loss / val_seen if val_seen else float("nan")
    avg_val_mse  = val_mse  / val_seen if val_seen else float("nan")
    avg_val_lm   = val_lm   / val_seen if val_seen else float("nan")
    avg_val_kl   = val_kl   / val_seen if val_seen else float("nan")

    # Calculate normalized validation loss for checkpointing
    alpha_config = config.get('alpha_schedule', {})
    if alpha_config.get('type') == 'linear_warmup':
        final_alpha = alpha_config.get('end_value', 0.1)
    else:
        final_alpha = alpha_config.get('value', 0.1)
    
    normalized_val_loss = (lm_base_weight * final_alpha) * avg_val_lm + kl_base_weight * avg_val_kl
    
    log.info(
        f"Validation – loss {avg_val_loss:.4f}, mse {avg_val_mse:.4f}, lm {avg_val_lm:.4f}, kl {avg_val_kl:.4f}"
    )
    log.info(f"Normalized validation loss (for checkpointing): {normalized_val_loss:.4f}")
    
    metrics_to_log = {
        "eval/loss/total": avg_val_loss,
        "eval/loss/normalized": normalized_val_loss,
        "eval/loss/mse":    avg_val_mse,
        "eval/loss/lm":     avg_val_lm,
        "eval/loss/kl":     avg_val_kl,
    }
    log_metrics(metrics_to_log, step=current_step)

    dec.train()
    enc.train()
    
    return {
        "avg_val_loss": avg_val_loss,
        "normalized_val_loss": normalized_val_loss,
        "avg_val_mse": avg_val_mse,
        "avg_val_lm": avg_val_lm,
        "avg_val_kl": avg_val_kl,
    }


def log_parameter_counts(dec_raw, enc_raw, orig, decoder_config, encoder_config, log):
    """Log detailed parameter counts for all models.
    
    Args:
        dec_raw: Raw Decoder model (before compilation)
        enc_raw: Raw Encoder model (before compilation)
        orig: Original model wrapper
        decoder_config: Decoder configuration
        encoder_config: Encoder configuration
        log: Logger instance
    
    Returns:
        dict: Summary statistics including total trainable parameters
    """
    trainable_params_list = [p for p in dec_raw.parameters() if p.requires_grad] + \
                            [p for p in enc_raw.parameters() if p.requires_grad]
    total_trainable_params_val = sum(p.numel() for p in trainable_params_list)
    num_params_orig_total = sum(p.numel() for p in orig.model.parameters())

    log.info(f"Total trainable parameters (Decoder + Encoder combined, based on current requires_grad): {total_trainable_params_val:,}")

    # Initialize counts for decoder
    num_params_dec_total = sum(p.numel() for p in dec_raw.parameters())
    num_params_dec_base_trainable, num_params_dec_proj_trainable, num_params_dec_out_trainable, \
    num_params_dec_prompts_trainable, num_params_dec_embeddings_trainable, num_params_dec_other_trainable = 0, 0, 0, 0, 0, 0
    
    num_params_dec_base_frozen, num_params_dec_proj_frozen, num_params_dec_out_frozen, \
    num_params_dec_prompts_frozen, num_params_dec_embeddings_frozen, num_params_dec_other_frozen = 0, 0, 0, 0, 0, 0

    for n, p in dec_raw.named_parameters():
        numel = p.numel()
        is_trainable = p.requires_grad

        if 'prompt_left_emb' in n or 'prompt_right_emb' in n:
            if is_trainable: num_params_dec_prompts_trainable += numel
            else: num_params_dec_prompts_frozen += numel
        elif n.startswith('proj.'):
            if is_trainable: num_params_dec_proj_trainable += numel
            else: num_params_dec_proj_frozen += numel
        elif n.startswith('out.'):
            if is_trainable: num_params_dec_out_trainable += numel
            else: num_params_dec_out_frozen += numel
        elif 'embed' in n or 'wte' in n or 'wpe' in n or 'lm_head.weight' in n or (hasattr(dec_raw, 'base') and hasattr(dec_raw.base, 'shared') and 'shared.weight' in n): # common embedding names
            if is_trainable: num_params_dec_embeddings_trainable += numel
            else: num_params_dec_embeddings_frozen += numel
        elif 'base.' in n: # Should be after more specific checks like embeddings if they are part of 'base'
            if is_trainable: num_params_dec_base_trainable += numel
            else: num_params_dec_base_frozen += numel
        else:
            if is_trainable: num_params_dec_other_trainable += numel
            else: num_params_dec_other_frozen += numel
            
    current_dec_trainable_total = (num_params_dec_base_trainable + num_params_dec_proj_trainable + 
                                   num_params_dec_out_trainable + num_params_dec_prompts_trainable +
                                   num_params_dec_embeddings_trainable + num_params_dec_other_trainable)
    current_dec_frozen_total = (num_params_dec_base_frozen + num_params_dec_proj_frozen +
                                num_params_dec_out_frozen + num_params_dec_prompts_frozen +
                                num_params_dec_embeddings_frozen + num_params_dec_other_frozen)
    
    log.info(f"Decoder - Total parameters: {num_params_dec_total:,}")
    log.info(f"  Decoder - Categorized Trainable parameters: {current_dec_trainable_total:,}")
    log.info(f"    Decoder base trainable: {num_params_dec_base_trainable:,} (Config: {decoder_config.base_model})")
    log.info(f"    Decoder proj trainable: {num_params_dec_proj_trainable:,} (Config: {decoder_config.projection_layer})")
    log.info(f"    Decoder out trainable: {num_params_dec_out_trainable:,} (Config: {decoder_config.output_head})")
    log.info(f"    Decoder prompts trainable: {num_params_dec_prompts_trainable:,} (Config: {decoder_config.trainable_prompts})")
    log.info(f"    Decoder embeddings trainable: {num_params_dec_embeddings_trainable:,}")
    log.info(f"    Decoder other trainable: {num_params_dec_other_trainable:,}")
    log.info(f"  Decoder - Categorized Frozen parameters: {current_dec_frozen_total:,}")
    log.info(f"    Decoder base frozen: {num_params_dec_base_frozen:,}")
    log.info(f"    Decoder proj frozen: {num_params_dec_proj_frozen:,}")
    log.info(f"    Decoder out frozen: {num_params_dec_out_frozen:,}")
    log.info(f"    Decoder prompts frozen: {num_params_dec_prompts_frozen:,}")
    log.info(f"    Decoder embeddings frozen: {num_params_dec_embeddings_frozen:,}")
    log.info(f"    Decoder other frozen: {num_params_dec_other_frozen:,}")

    # Initialize counts for encoder
    num_params_enc_total = sum(p.numel() for p in enc_raw.parameters())
    num_params_enc_base_trainable, num_params_enc_proj_trainable, \
    num_params_enc_prompts_trainable, num_params_enc_embeddings_trainable, num_params_enc_other_trainable = 0, 0, 0, 0, 0

    num_params_enc_base_frozen, num_params_enc_proj_frozen, \
    num_params_enc_prompts_frozen, num_params_enc_embeddings_frozen, num_params_enc_other_frozen = 0, 0, 0, 0, 0

    for n, p in enc_raw.named_parameters():
        numel = p.numel()
        is_trainable = p.requires_grad

        if 'soft_prompt_embeddings' in n:
            if is_trainable: num_params_enc_prompts_trainable += numel
            else: num_params_enc_prompts_frozen += numel
        elif n.startswith('proj.'):
            if is_trainable: num_params_enc_proj_trainable += numel
            else: num_params_enc_proj_frozen += numel
        elif 'embed' in n or 'wte' in n or 'wpe' in n: # Common embedding names
            if is_trainable: num_params_enc_embeddings_trainable += numel
            else: num_params_enc_embeddings_frozen += numel
        elif 'base.' in n:
            if is_trainable: num_params_enc_base_trainable += numel
            else: num_params_enc_base_frozen += numel
        else:
            if is_trainable: num_params_enc_other_trainable += numel
            else: num_params_enc_other_frozen += numel

    current_enc_trainable_total = (num_params_enc_base_trainable + num_params_enc_proj_trainable +
                                   num_params_enc_prompts_trainable + num_params_enc_embeddings_trainable + 
                                   num_params_enc_other_trainable)
    current_enc_frozen_total = (num_params_enc_base_frozen + num_params_enc_proj_frozen +
                                num_params_enc_prompts_frozen + num_params_enc_embeddings_frozen +
                                num_params_enc_other_frozen)

    log.info(f"Encoder - Total parameters: {num_params_enc_total:,}")
    log.info(f"  Encoder - Categorized Trainable parameters: {current_enc_trainable_total:,}")
    log.info(f"    Encoder base trainable: {num_params_enc_base_trainable:,} (Config: {encoder_config.base_model}, present: {encoder_config.use_base_model})")
    log.info(f"    Encoder proj trainable: {num_params_enc_proj_trainable:,} (Config: {encoder_config.projection_layer})")
    log.info(f"    Encoder prompts trainable: {num_params_enc_prompts_trainable:,} (Config: {encoder_config.trainable_soft_prompt})")
    log.info(f"    Encoder embeddings trainable: {num_params_enc_embeddings_trainable:,}")
    log.info(f"    Encoder other trainable: {num_params_enc_other_trainable:,}")
    log.info(f"  Encoder - Categorized Frozen parameters: {current_enc_frozen_total:,}")
    log.info(f"    Encoder base frozen: {num_params_enc_base_frozen:,}")
    log.info(f"    Encoder proj frozen: {num_params_enc_proj_frozen:,}")
    log.info(f"    Encoder prompts frozen: {num_params_enc_prompts_frozen:,}")
    log.info(f"    Encoder embeddings frozen: {num_params_enc_embeddings_frozen:,}")
    log.info(f"    Encoder other frozen: {num_params_enc_other_frozen:,}")
    
    sum_of_categorized_trainable = current_dec_trainable_total + current_enc_trainable_total
    sum_of_categorized_frozen = current_dec_frozen_total + current_enc_frozen_total
    sum_of_categorized_total = sum_of_categorized_trainable + sum_of_categorized_frozen
    
    if total_trainable_params_val != sum_of_categorized_trainable:
        log.warning(
            f"Parameter count mismatch (trainable): total_trainable_params_val (sum of all requires_grad=True) is {total_trainable_params_val:,}, "
            f"but sum of categorized trainable parameters (Decoder + Encoder) is {sum_of_categorized_trainable:,}. "
            "This should ideally match. If 'other_trainable' is non-zero, it indicates parameters not fitting standard categories. "
            "If 'other_trainable' is zero and there's still a mismatch, the categorization logic needs review."
        )
    
    total_model_params = num_params_dec_total + num_params_enc_total
    # Check if the sum of all categorized parameters matches the actual total model parameters
    if total_model_params != sum_of_categorized_total:
        log.warning(
            f"Total parameter count mismatch: actual total (Decoder + Encoder) is {total_model_params:,}, "
            f"but sum of all categorized parameters (trainable + frozen) is {sum_of_categorized_total:,}. "
            "This indicates an issue in the categorization logic failing to cover all parameters."
        )

    log.info(f"Original LLM (frozen) parameters: {num_params_orig_total:,}")
    
    return {
        'total_trainable': total_trainable_params_val, # This is based on current requires_grad status
        'trainable_params_list': trainable_params_list, # The actual list of params
        'decoder_total': num_params_dec_total,
        'encoder_total': num_params_enc_total,
        'orig_total': num_params_orig_total
    }

def do_all_initial_validation(batch, orig, tokenizer, device, log, activation_dir):
    from lens.training.test import diagnose_activation_mismatch
    diagnosis = diagnose_activation_mismatch(
        batch, orig, tokenizer, device, sample_idx=0, verbose=True
    )
    log.warning(diagnosis)
    # In your training script, after loading a batch:
    from lens.training.test import diagnose_activation_save_load
    from lens.training.test import check_dataset_activation_format
    print("\n=== Save/Load Cycle Diagnosis ===")
    i = 0  # First sample
    l = int(batch["layer_idx"][i].item())
    p = int(batch["token_pos_A"][i].item())
    input_ids = batch["input_ids_A"][i].unsqueeze(0).to(device)

    save_load_results, fresh_act = diagnose_activation_save_load(orig, input_ids, l, p, device)
    for k, v in save_load_results.items():
        print(f"{k}: {v}")

    print("\n=== Dataset Format Check ===")
    # Use the actual activation directory
    check_dataset_activation_format(activation_dir)

    print("\n=== Batch Activation Info ===")
    print(f"Batch A shape: {batch['A'].shape}")
    print(f"Batch A dtype: {batch['A'].dtype}")
    print(f"Batch A[0] norm: {batch['A'][0].norm().item():.4f}")
    from lens.training.test import test_autocast_difference
    test_autocast_difference(orig, input_ids, l, p, device)

    from lens.training.test import check_layer_indexing
    check_layer_indexing(orig, input_ids, device)





def get_initial_model_state(model: torch.nn.Module) -> dict:
    """Capture initial trainable parameters (always stored on CPU)."""
    state = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            state[name] = p.detach().cpu().clone()   #  <-- keep on CPU
    return state



class Timer:
    def __init__(self, name: str, logger: logging.Logger, main_process: bool = True,
                 log_wandb: bool = False, wandb_step: int = None):
        self.name = name
        self.logger = logger
        self.main_process = main_process
        self.start_time = None
        self.elapsed_time = None
        self.log_wandb = log_wandb
        self.wandb_step = wandb_step

    def __enter__(self):
        if self.main_process:
            self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.main_process and self.start_time is not None:
            self.elapsed_time = time.time() - self.start_time
            self.logger.info(f"Phase '{self.name}' took {self.elapsed_time:.2f}s")
            if self.log_wandb and self.wandb_step is not None:
                metric_name = f"time/{self.name.replace(' ', '_').lower()}_duration_s"
                log_metrics({metric_name: self.elapsed_time}, step=self.wandb_step)


def convert_batch_dtype(batch, config, device, logger=None, is_main_process=True):
    """Convert batch tensors to the appropriate dtype based on config.
    
    Args:
        batch: Dictionary containing batch data
        config: Training configuration
        device: Target device
        logger: Optional logger for first-time conversion message
        is_main_process: Whether this is the main process
        
    Returns:
        tuple: (converted_count, target_dtype) or (0, None) if no conversion
    """
    global _first_conversion_logged
    
    if not config.get('force_data_conversion', False):
        return 0, None
    
    # Determine target dtype based on mixed precision config
    mixed_precision_config = config.get('mixed_precision', {'enabled': True, 'dtype': 'auto'})
    target_dtype = torch.float32  # Default
    
    if device.type == "cuda" and mixed_precision_config.get('enabled', True):
        dtype_str = mixed_precision_config.get('dtype', 'auto')
        if dtype_str == 'float16':
            target_dtype = torch.float16
        elif dtype_str == 'bfloat16':
            target_dtype = torch.bfloat16
        # For 'auto' and 'float32', keep float32
    
    # Convert batch tensors to target dtype
    converted_count = 0
    source_dtype = None
    for key, val in batch.items():
        if isinstance(val, torch.Tensor) and val.dtype in [torch.float16, torch.bfloat16, torch.float32]:
            if val.dtype != target_dtype:
                if source_dtype is None:
                    source_dtype = val.dtype
                batch[key] = val.to(dtype=target_dtype)
                converted_count += 1
    
    # Log first conversion
    if converted_count > 0 and not _first_conversion_logged and logger and is_main_process:
        logger.info(f"Converting batch data from {source_dtype} to {target_dtype} (first batch)")
        _first_conversion_logged = True
    
    return converted_count, target_dtype


def check_model_dtypes(models_dict, expected_dtype, logger, step=None):
    """Debug function to check all model tensors have the expected dtype."""
    issues = []
    for model_name, model in models_dict.items():
        # Check parameters
        for name, param in model.named_parameters():
            if param.dtype != expected_dtype:
                issues.append(f"{model_name}.{name}: {param.dtype} (expected {expected_dtype})")
        
        # Check buffers
        for name, buffer in model.named_buffers():
            if buffer.dtype in [torch.float16, torch.bfloat16, torch.float32] and buffer.dtype != expected_dtype:
                issues.append(f"{model_name}.{name} (buffer): {buffer.dtype} (expected {expected_dtype})")
    
    if issues:
        logger.error(f"Dtype mismatches found at step {step}:")
        for issue in issues[:10]:  # Show first 10 issues
            logger.error(f"  - {issue}")
        if len(issues) > 10:
            logger.error(f"  ... and {len(issues) - 10} more")
        return False
    return True


