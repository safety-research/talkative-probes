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


# Helper function to prepare datasets and dataloaders
def _prepare_dataloaders(
    config: dict,
    activation_dir: str,
    effective_val_activation_dir: str | None,
    max_train_samples_req: int | None,
    max_val_samples_req: int | None,
    log: logging.Logger
) -> tuple[DataLoader | None, DataLoader | None, Dataset | None, Dataset | None]:
    """Loads, splits (if necessary), and creates DataLoaders for train/validation."""

    train_ds: Dataset | None = None
    val_ds: Dataset | None = None

    # Retrieve dataset configuration from the main config
    split_seed = config['split_seed']
    val_fraction = config.get('val_fraction', 0.1) # Default val_fraction if not in config
    batch_size = config['batch_size']
    group_n = config.get('group_n', 1)

    if effective_val_activation_dir and Path(effective_val_activation_dir).exists():
        log.info(f"Loading training data from {activation_dir} (limit: {max_train_samples_req if max_train_samples_req is not None else 'all'}).")
        train_ds = ActivationDataset(activation_dir, max_samples=max_train_samples_req, desc="Loading train activations")
        log.info(f"Loading validation data from {effective_val_activation_dir} (limit: {max_val_samples_req if max_val_samples_req is not None else 'all'}).")
        val_ds = ActivationDataset(effective_val_activation_dir, max_samples=max_val_samples_req, desc="Loading val activations")
        
        if not train_ds or len(train_ds) == 0:
            raise RuntimeError(
                f"No .pt files found or loaded from train directory {activation_dir} (limit: {max_train_samples_req})."
            )
        if val_ds is not None and len(val_ds) == 0: # Check if val_ds was attempted but empty
            log.warning(
                f"No .pt files found or loaded from validation directory {effective_val_activation_dir} (limit: {max_val_samples_req}). Validation will be skipped."
            )
            val_ds = None # Ensure val_loader becomes None later

    else:
        if effective_val_activation_dir and not Path(effective_val_activation_dir).exists():
            log.warning(f"Validation activations directory {effective_val_activation_dir} not found. Falling back to random split from {activation_dir}.")
        
        log.info(f"Preparing to split data from {activation_dir} with val_fraction={val_fraction:.2f}, seed={split_seed}.")

        initial_load_n = None
        # Determine total number of samples to load initially to satisfy requests + fraction
        if max_train_samples_req is not None and max_val_samples_req is not None:
            initial_load_n = max_train_samples_req + max_val_samples_req
        elif max_train_samples_req is not None: # Only train count specified
            if 0.0 <= val_fraction < 1.0 and (1.0 - val_fraction) > 1e-9: # train_part > 0
                initial_load_n = math.ceil(max_train_samples_req / (1.0 - val_fraction))
            else: # val_fraction is 1.0 (all val) or train_part is 0. Load at least max_train_req.
                  # The split logic later will handle if max_train_req is incompatible with val_fraction.
                initial_load_n = max_train_samples_req 
        elif max_val_samples_req is not None: # Only val count specified
            if 0.0 < val_fraction <= 1.0 and val_fraction > 1e-9: # val_part > 0
                initial_load_n = math.ceil(max_val_samples_req / val_fraction)
            else: # val_fraction is 0.0 (all train) or val_part is 0. Load at least max_val_req.
                initial_load_n = max_val_samples_req
        
        log.info(f"Initial load limit for splitting: {initial_load_n if initial_load_n is not None else 'all available'}.")
        full_dataset_loaded = ActivationDataset(activation_dir, max_samples=initial_load_n, desc="Loading activations for split")

        if not full_dataset_loaded or len(full_dataset_loaded) == 0:
            raise RuntimeError(
                f"No .pt files found or loaded from {activation_dir} (limit: {initial_load_n}). Run scripts/00_dump_activations.py first."
            )

        available_total = len(full_dataset_loaded)
        log.info(f"Loaded {available_total} samples for splitting.")

        # Determine target sizes for train and val based on requests and available data
        final_val_size = 0
        if max_val_samples_req is not None:
            final_val_size = min(max_val_samples_req, available_total)
        else:
            final_val_size = int(available_total * val_fraction)
        final_val_size = max(0, min(final_val_size, available_total))

        final_train_size = 0
        if max_train_samples_req is not None:
            final_train_size = min(max_train_samples_req, available_total - final_val_size)
        else:
            final_train_size = available_total - final_val_size
        final_train_size = max(0, final_train_size)

        # Adjust if sum of targets differs from available_total or requested sum
        # This ensures the dataset to be split matches the sum of final train/val sizes.
        dataset_to_actually_split: Dataset = full_dataset_loaded
        current_total_target = final_train_size + final_val_size

        if current_total_target > available_total:
            # This case implies requests were too high for available data.
            # Re-evaluate based on available_total, prioritizing val_fraction.
            log.warning(f"Sum of calculated train ({final_train_size}) and val ({final_val_size}) "
                        f"exceeds available ({available_total}). Re-adjusting based on val_fraction.")
            final_val_size = int(available_total * val_fraction)
            final_train_size = available_total - final_val_size
            # dataset_to_actually_split is already full_dataset_loaded (i.e., all available)
        elif current_total_target < available_total:
            # Loaded more than needed by final_train_size + final_val_size. Take a subset.
            log.info(f"Loaded {available_total} but target sum is {current_total_target}. "
                     f"Taking subset of {current_total_target} before splitting.")
            dataset_to_actually_split = Subset(full_dataset_loaded, range(current_total_target))
        # Else (current_total_target == available_total), dataset_to_actually_split is full_dataset_loaded.
        
        # Perform the split on dataset_to_actually_split
        # The lengths for random_split must be final_train_size and final_val_size.
        # Their sum is len(dataset_to_actually_split) due to the logic above.
        
        if final_val_size > 0 and final_train_size > 0:
            log.info(f"Splitting {len(dataset_to_actually_split)} samples into train: {final_train_size}, val: {final_val_size}.")
            # Type ignore below because random_split can return List[Subset[T]]
            # and we are destructuring it.
            train_ds, val_ds = random_split( # type: ignore[assignment]
                dataset_to_actually_split,
                [final_train_size, final_val_size],
                generator=torch.Generator().manual_seed(split_seed),
            )
        elif final_train_size > 0: # Only train data
            train_ds = dataset_to_actually_split 
            val_ds = None
            log.info(f"Using all {len(train_ds)} samples from loaded/subsetted data for training, no validation split.")
        elif final_val_size > 0: # Only val data
            val_ds = dataset_to_actually_split
            train_ds = None
            log.warning(f"Training set is empty. Using all {len(val_ds)} samples from loaded/subsetted data for validation.")
        else: # Both are zero
            log.warning("Both train and validation target sizes are 0. No data to load.")
            train_ds = None
            val_ds = None

    # Create DataLoaders
    train_loader: DataLoader | None = None
    if train_ds and len(train_ds) > 0:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    else:
        # This path should ideally not be reached if checks above are robust,
        # but as a safeguard:
        log.error("Training dataset is empty or None after processing. Cannot create DataLoader.")
        raise RuntimeError("Training dataset is empty. Check data paths, limits, and split configuration.")

    val_loader: DataLoader | None = None
    if val_ds and len(val_ds) > 0:
        val_loader = DataLoader(val_ds, batch_size=batch_size*group_n, shuffle=False, collate_fn=collate)
    else:
        # This is an expected outcome if no validation data was configured or found.
        log.info("Validation dataset is empty or None. Validation will be skipped during training.")
    
    return train_loader, val_loader, train_ds, val_ds

# ... existing imports and helper functions ...

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



@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:  # noqa: D401
    """Hydra-powered entry point.

    The body below relies on the `config` dict, derived from the Hydra
    configuration.
    """

    # Convert Hydra config to a plain Python dict.
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Parse flexible schedule notations (e.g., "1000s", "5e") into detailed format
    config = parse_schedule_config(config)

    # ---------------------------------------------------------------
    # Logging setup (console). W&B handled via lens.utils.logging.
    # ---------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # From this point onward the original training logic remains intact
    # and can keep using `config`.
    # ------------------------------------------------------------------

    # Use overridden values or defaults from config
    model_name = config['model_name']
    tokenizer_name = config.get("tokenizer_name", model_name) # Moved here for earlier access
    layer_l = config.get('layer_l', 5)  # Get layer number from config
    
    # Handle activation_dir override carefully: CLI takes precedence over config.
    cli_activation_dir = config.get('activation_dir')
    base_activation_dir_str = cli_activation_dir if cli_activation_dir is not None else config['activation_dumper']['output_dir']
    # Resolve path relative to project root
    base_activation_path = resolve_path(base_activation_dir_str)
    # Include layer in the path: parent / model_name / layer_X / name
    model_name_clean = config['model_name'].replace("/", "_")
    activation_dir = str(base_activation_path.parent / model_name_clean / f"layer_{layer_l}" / base_activation_path.name)
    
    # Handle val_activation_dir: CLI takes precedence over config.
    base_val_activation_dir_str = config.get('val_activation_dir')
    effective_val_activation_dir: str | None = None
    if base_val_activation_dir_str:
        # Resolve path relative to project root
        base_val_path = resolve_path(base_val_activation_dir_str)
        # Include layer in the validation path too
        effective_val_activation_dir = str(base_val_path.parent / model_name_clean / f"layer_{layer_l}" / base_val_path.name)
    
    max_steps = config['max_train_steps']
    learning_rate = config['learning_rate']
    t_text = config['t_text']
    wandb_config = config.get('wandb', {}) # Ensure wandb_config is a dict
    lm_base_weight = config['lm_base_weight']
    kl_base_weight = config['kl_base_weight']
    entropy_weight = config['entropy_weight']
    gradient_accumulation_steps = config['gradient_accumulation_steps']

    # Extract trainable_components and custom_lr_multipliers from config
    trainable_components_config = config.get('trainable_components', {})
    decoder_train_cfg = trainable_components_config.get('decoder', {})
    encoder_train_cfg = trainable_components_config.get('encoder', {})
    custom_lr_multipliers = config.get('custom_lr_multipliers', {})
    projection_lr_multiplier = custom_lr_multipliers.get('projection_layers', 1.0)
    embedding_lr_multiplier = custom_lr_multipliers.get('embedding_layers', 1.0)
    prompt_lr_multiplier = custom_lr_multipliers.get('prompt_layers', 1.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract dataset info from activation directory
    dataset_info = extract_dataset_info(activation_dir)
    
    # Get config name from Hydra if available
    config_name = _get_hydra_config_name()
    
    # Generate run name (or use override)
    run_name_override = config.get('run_name')
    if run_name_override:
        run_name = run_name_override
        log.info(f"Using user-specified run name: {run_name}")
    else:
        run_name = generate_run_name(config, dataset_info, config.get('resume'), config_name, config.get('run_suffix'))
    
    # Append SLURM job ID if running under SLURM
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    if slurm_job_id:
        run_name = f"{run_name}_slurm{slurm_job_id}"
        log.info(f"Appended SLURM job ID to run name: {run_name}")
    
    # Update checkpoint output directory to include run name
    checkpoint_config = config.get('checkpoint', {})
    base_checkpoint_dir = resolve_path(checkpoint_config.get('output_dir', 'outputs'))
    run_checkpoint_dir = base_checkpoint_dir / run_name
    checkpoint_config['output_dir'] = str(run_checkpoint_dir)
    config['checkpoint'] = checkpoint_config
    
    # Log run information
    log.info("=" * 60)
    log.info(f"Run Name: {run_name}")
    log.info(f"Dataset: {dataset_info.get('dataset', 'unknown')}")
    log.info(f"Model: {dataset_info.get('model_name', 'unknown')}")
    log.info(f"Layer: {dataset_info.get('layer', 'unknown')}")
    log.info(f"Checkpoint Dir: {run_checkpoint_dir}")
    log.info("=" * 60)
    
    # Handle wandb resume
    wandb_run_id = config.get('wandb_resume_id')
    force_disable_wandb = False
    # Handle explicit None values (e.g., from command line wandb_resume_id=None)
    if wandb_run_id is not None and str(wandb_run_id).lower() == 'none':
        wandb_run_id = None
        force_disable_wandb = True
        log.info("Explicitly disabling WandB run resumption (wandb_resume_id=None)")
    wandb_resume_mode = None
    
    resume_checkpoint_path = config.get('resume')
    # If resuming from checkpoint and no explicit wandb ID provided, try to load from checkpoint
    if resume_checkpoint_path and not wandb_run_id:
        # Check if checkpoint file exists
        if not os.path.exists(resume_checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {resume_checkpoint_path}")
        # Peek into checkpoint to get wandb run ID if available
        checkpoint_data = torch.load(resume_checkpoint_path, map_location='cpu')
        wandb_run_id = checkpoint_data.get('wandb_run_id')
        if wandb_run_id and not force_disable_wandb:
            log.info(f"Found wandb run ID in checkpoint: {wandb_run_id}")
            wandb_resume_mode = "must"  # Force resume of the exact run
    
    # Initialize W&B logging (if enabled in config)
    wandb_init_kwargs = {
        'project': wandb_config.get('project', 'consistency-lens'),
        'name': run_name,  # Use our generated run name
        'config': config,
        'mode': wandb_config.get('mode', 'online'),
        'tags': []  # Initialize tags list
    }
    
    # Add command line invocation to config
    command_line_args = ' '.join(sys.argv)
    wandb_init_kwargs['config']['command_line'] = command_line_args
    
    # Add environment variable that submit_with_config.sh can set
    submit_script_command = os.environ.get('SUBMIT_SCRIPT_COMMAND', None)
    if submit_script_command:
        wandb_init_kwargs['config']['submit_script_command'] = submit_script_command
        log.info(f"Submit script command: {submit_script_command}")
    
    # Add SLURM environment info to config if available
    slurm_job_id = os.environ.get('SLURM_JOB_ID', None)
    if slurm_job_id:
        slurm_info = {
            'slurm_job_id': slurm_job_id,
            'slurm_job_name': os.environ.get('SLURM_JOB_NAME', 'unknown'),
            'slurm_nodelist': os.environ.get('SLURM_NODELIST', 'unknown'),
            'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID', None),
        }
        wandb_init_kwargs['config']['slurm_info'] = slurm_info
        wandb_init_kwargs['tags'].append(f"slurm-{slurm_job_id}")
        wandb_init_kwargs['tags'].append(f"node-{slurm_info['slurm_nodelist']}")
        log.info(f"Running under SLURM job ID: {slurm_job_id} on nodes: {slurm_info['slurm_nodelist']}")
    
    # Add dataset and model tags
    if dataset_info.get('dataset'):
        wandb_init_kwargs['tags'].append(f"dataset-{dataset_info['dataset']}")
    if dataset_info.get('model_name'):
        model_tag = dataset_info['model_name'].replace('/', '-')
        wandb_init_kwargs['tags'].append(f"model-{model_tag}")
    if dataset_info.get('layer') is not None:
        wandb_init_kwargs['tags'].append(f"layer-{dataset_info['layer']}")
    
    # Add config name tag
    if config_name and config_name != 'config':
        wandb_init_kwargs['tags'].append(f"config-{config_name}")
    
    # Add resume parameters if we have a run ID
    if wandb_run_id and not force_disable_wandb:
        wandb_init_kwargs['id'] = wandb_run_id
        wandb_init_kwargs['resume'] = wandb_resume_mode or "allow"
    
    # Initialize wandb and get the run ID
    current_wandb_run_id = log_init(**wandb_init_kwargs)
    
    log.info(f"Current CUDA devices: {torch.cuda.current_device()}")
    log.info(f"Current CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # Prepare DataLoaders by calling the helper function
    train_loader, val_loader, train_ds, val_ds = _prepare_dataloaders(
        config=config,
        activation_dir=activation_dir,
        effective_val_activation_dir=effective_val_activation_dir,
        max_train_samples_req=config.get('max_train_samples'),
        max_val_samples_req=config.get('max_val_samples'),
        log=log
    )

    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0: # Should not happen if dataset is not empty and batch_size > 0
        log.warning("DataLoader is empty or batch_size is too large for dataset; steps_per_epoch is 0.")
        # This implies an issue, but to prevent division by zero if max_steps > 0:
        num_epochs_total_approx = 0 if max_steps == 0 else 1 
    else:
        # Handle epoch-based training
        num_train_epochs = config.get('num_train_epochs', 0)
        
        if num_train_epochs > 0 and max_steps == 0:
            # Epoch-based training: calculate max_steps from num_train_epochs
            max_steps = steps_per_epoch * num_train_epochs
            num_epochs_total_approx = num_train_epochs
            log.info(f"Epoch-based training: {num_train_epochs} epochs × {steps_per_epoch} steps/epoch = {max_steps} total steps")
        elif max_steps > 0:
            # Step-based training: calculate approximate epochs from max_steps
            num_epochs_total_approx = (max_steps - 1) // steps_per_epoch + 1
        else:
            # Neither epochs nor steps specified
            raise ValueError("Either 'num_train_epochs' or 'max_train_steps' must be > 0 in config")
    
    # Parse intervals with flexible notation (now that we have steps_per_epoch)
    wandb_log_interval = _resolve_schedule_to_steps(config['wandb_log_interval'], steps_per_epoch, log, "wandb_log_interval", gradient_accumulation_steps)
    log_interval = _resolve_schedule_to_steps(config['log_interval'], steps_per_epoch, log, "log_interval", gradient_accumulation_steps)
    
    if log_interval <= 0:
        raise ValueError(f"log_interval must be positive, got {config['log_interval']}")
    if wandb_log_interval <= 0:
        raise ValueError(f"wandb_log_interval must be positive, got {config['wandb_log_interval']}")
    
    # val_interval is used by the training loop
    val_interval_str = config['val_interval']
    val_interval = _resolve_schedule_to_steps(val_interval_str, steps_per_epoch, log, "val_interval", gradient_accumulation_steps)
    
    # Initialize checkpoint manager with updated config (now that we have steps_per_epoch)
    checkpoint_manager = CheckpointManager(config, log, steps_per_epoch, gradient_accumulation_steps)

    log.info("Starting training run – Model: %s, Activations: %s", model_name, activation_dir)
    log.info(
        "Configuration: %d total steps, Batch Size: %d, Gradient Accumulation: %d, Train Dataset Size: %d, Val Dataset Size: %d samples",
        max_steps, config['batch_size'], gradient_accumulation_steps, len(train_ds), len(val_ds)
    )
    if gradient_accumulation_steps > 1:
        log.info(
            "Effective batch size: %d (batch_size=%d × gradient_accumulation_steps=%d)",
            config['batch_size'] * gradient_accumulation_steps, config['batch_size'], gradient_accumulation_steps
        )
    if steps_per_epoch > 0 :
        log.info(
            "Derived: %d steps/epoch, Approx. %d total epochs",
            steps_per_epoch, num_epochs_total_approx
        )

    if val_loader:
        log.info("Dataset split – train: %d | val: %d", len(train_ds), len(val_ds))

    # Initialize models using new config flags
    decoder_config = DecoderConfig(
        model_name=model_name,
        **decoder_train_cfg
    )
    dec_raw = Decoder(decoder_config)

    encoder_config = EncoderConfig(
        model_name=model_name,
        **encoder_train_cfg
    )
    enc_raw = Encoder(encoder_config)

    # ------------------------------------------------------------------
    # Tokenizer & vocab-size-based resizing (do this BEFORE optional torch.compile)
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    log.info(f"Tokenizer name: {tokenizer_name}")

    new_vocab_size = tokenizer.vocab_size

    from lens.utils.embedding_remap import remap_embeddings
    base_tok = AutoTokenizer.from_pretrained(model_name)

    if config["tokenizer_name"] != model_name:
        log.info(f"Remapping embeddings from {model_name} to {tokenizer_name}")
        remap_embeddings(dec_raw.base, base_tok, tokenizer)
        remap_embeddings(enc_raw.base, base_tok, tokenizer)
        log.info("Remapped Decoder & Encoder embedding matrices to new tokenizer")

    # Reinitialise prompt ids to ensure they are within new vocab
    dec_raw.set_prompt(config["decoder_prompt"], tokenizer)
    log.info("Prompt for decoder: %s",str([tokenizer.decode(d) for d in dec_raw.prompt_ids]))

    # Initialize encoder soft prompt from text if specified
    encoder_soft_prompt_text = encoder_train_cfg.get('soft_prompt_init_text')
    if encoder_soft_prompt_text:
        enc_raw.set_soft_prompt_from_text(encoder_soft_prompt_text, tokenizer)
        log.info("Initialized encoder soft prompt from text: %s", encoder_soft_prompt_text)

    # Ensure Decoder's standalone LM head matches new vocab
    if dec_raw.out.weight.size(0) != new_vocab_size:
        d_model = dec_raw.base.config.hidden_size
        dec_raw.out = nn.Linear(d_model, new_vocab_size, bias=False)
        with torch.no_grad():
            dec_raw.out.weight.copy_(dec_raw.base.get_output_embeddings().weight)
        log.info("Resized Decoder.out to new vocab size")

    # ... after model initialization but before compilation ...

    # Original model wrapper (remap after creation)
    orig = OrigWrapper(model_name, load_in_8bit=False)
    if config["tokenizer_name"] != model_name:
        remap_embeddings(orig.model, base_tok, tokenizer)
        log.info("Remapped Orig model embeddings to new tokenizer")
    orig.model.to(device)
    
    # Get trainable params and log parameter counts (do this BEFORE compilation)
    param_stats = log_parameter_counts(dec_raw, enc_raw, orig, decoder_config, encoder_config, log)
    trainable_params = param_stats['trainable_params_list']
    total_trainable_params_val = param_stats['total_trainable']
    


    # Now compile models if requested
    if config.get('compile_models', True):
        log.info("Compiling models")
        dec = torch.compile(dec_raw).to(device)
        enc = torch.compile(enc_raw).to(device)
    else:
        log.info("Not compiling models")
        dec = dec_raw.to(device)
        enc = enc_raw.to(device)

    # Handle freeze schedule
    freeze_schedule_config = config.get('freeze_schedule', {})
    freeze_schedule_enabled = freeze_schedule_config.get('enabled', False)
    

    if freeze_schedule_enabled:
        # Log initial freeze schedule configuration
        global_unfreeze_at = freeze_schedule_config.get('unfreeze_at')
        if global_unfreeze_at:
            log.info(f"Freeze schedule enabled: global unfreeze timing = {global_unfreeze_at}")
        
        # Log component-specific timing
        components_config = freeze_schedule_config.get('components', {})
        for component_name, component_cfg in components_config.items():
            for param_name, param_cfg in component_cfg.items():
                if isinstance(param_cfg, dict) and 'unfreeze_at' in param_cfg:
                    unfreeze_at = param_cfg['unfreeze_at']
                    if unfreeze_at:
                        log.info(f"Freeze schedule: {component_name}.{param_name} will unfreeze at {unfreeze_at}")
        
        # Legacy compatibility logging
        legacy_step = freeze_schedule_config.get('unfreeze_at_step')
        legacy_epoch = freeze_schedule_config.get('unfreeze_at_epoch')
        if legacy_step is not None:
            log.info(f"Legacy freeze schedule: will unfreeze at step {legacy_step}")
        elif legacy_epoch is not None:
            log.info(f"Legacy freeze schedule: will unfreeze at epoch {legacy_epoch}")
        
        # Initially freeze non-adapter parameters (base_model and output_head)
        # We'll freeze them regardless of the config settings, then unfreeze later
        for name, param in dec_raw.named_parameters():
            if 'base' in name or 'out' in name:
                param.requires_grad = False
        
        for name, param in enc_raw.named_parameters():
            if 'base' in name:
                param.requires_grad = False
        
        log.info("Froze non-adapter parameters (base models and output head) for initial training phase")
    else:
        unfreeze_at_step = -1  # Never unfreeze
        log.info("Freeze schedule disabled - using standard trainable component settings")

    # Get trainable params and log parameter counts
    trainable_params = param_stats['trainable_params_list']
    total_trainable_params_val = param_stats['total_trainable']
    
    log.info(f"Hyperparameters: lm_base_weight={lm_base_weight}, kl_base_weight={kl_base_weight}, entropy_weight={entropy_weight}")
    log.info(f"Learning rate: {learning_rate}, Projection LR Multiplier: {projection_lr_multiplier}, Embedding LR Multiplier: {embedding_lr_multiplier}, Prompt LR Multiplier: {prompt_lr_multiplier}")
    log.info(f"Prompt LR Multiplier: {prompt_lr_multiplier}")
    log.info(f"Stop-grad on A′: {config['stop_grad_aprime']}")
    log.info(f"Grad clip: {config['grad_clip']}")
    
    # Create optimizer groups with potentially different LRs
    optimizer_groups = param_groups([dec, enc], learning_rate, projection_lr_multiplier, embedding_lr_multiplier, prompt_lr_multiplier)

    # Verify that the number of parameters in optimizer groups matches the count from trainable_params list.
    num_params_in_optimizer_groups = sum(p.numel() for group in optimizer_groups for p in group['params'])
    if total_trainable_params_val != num_params_in_optimizer_groups:
        log_message = (
            f"Parameter count difference: initial total_trainable_params_val (before freeze schedule modifications) is {total_trainable_params_val:,}, "
            f"but optimizer groups (reflecting current requires_grad) sum to {num_params_in_optimizer_groups:,}."
        )
        if freeze_schedule_enabled:
            log.info(
                f"{log_message} This difference is expected when a freeze schedule is active, "
                "as some parameters are initially frozen and excluded from the optimizer."
            )
        else:
            log.warning(
                f"{log_message} Check requires_grad flags and param grouping logic, especially if no freeze schedule is active."
            )

    opt = torch.optim.AdamW(optimizer_groups)
    
    # Get mixed precision configuration
    mixed_precision_config = config.get('mixed_precision', {'enabled': True, 'dtype': 'auto'})
    
    # Log mixed precision settings
    if mixed_precision_config.get('enabled', True):
        dtype_str = mixed_precision_config.get('dtype', 'auto')
        if dtype_str == 'auto':
            actual_dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float32'
            log.info(f"Mixed precision enabled: auto mode will use {actual_dtype}")
        else:
            log.info(f"Mixed precision enabled: using {dtype_str}")
    else:
        log.info("Mixed precision disabled")
    
    # GradScaler setup based on mixed precision config
    # Only enable scaler for float16 or bfloat16 on CUDA
    scaler_enabled = (
        device.type == "cuda" and 
        mixed_precision_config.get('enabled', True) and
        mixed_precision_config.get('dtype', 'auto') != 'float32'
    )
    scaler = GradScaler(enabled=scaler_enabled)
    
    # Create learning rate scheduler  
    lr_scheduler_config = config.get('lr_scheduler', {'type': 'constant'})
    scheduler_last_epoch = -1  # Default for new training
    resume_epoch = 0  # Default for new training
    lr_scheduler = get_lr_scheduler(opt, lr_scheduler_config, max_steps, 
                                    last_epoch=scheduler_last_epoch,
                                    current_epoch=0, steps_per_epoch=steps_per_epoch)
    if lr_scheduler:
        log.info(f"Using LR scheduler: {lr_scheduler_config['type']}")
        if lr_scheduler_config.get('warmup_steps', 0) > 0:
            log.info(f"  with {lr_scheduler_config['warmup_steps']} warmup steps")

    start_step = 0
    # resume_checkpoint_path was defined earlier from config.get('resume')
    if resume_checkpoint_path:
        # Check if we should reset the step counter
        reset_steps = config.get('resume_reset_steps', False)
        
        # Checkpoint stores the last completed step
        try:
            rec = checkpoint_manager.load_checkpoint(
                resume_checkpoint_path, 
                models={"dec": dec, "enc": enc}, 
                optimizer=opt if not reset_steps else None,  # Don't load optimizer state if resetting
                map_location=device
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {resume_checkpoint_path}: {str(e)}") from e
        
        if reset_steps:
            start_step = 0
            resume_epoch = 0
            scheduler_last_epoch = -1
            log.info("Resetting training steps to 0 (keeping model weights only)")
        else:
            start_step = int(rec.get("step", -1)) + 1 # Resume from the next step
            resume_epoch = start_step // steps_per_epoch if steps_per_epoch > 0 else 0
            # For LR scheduler, last_epoch should be the step count (it's misnamed in PyTorch)
            scheduler_last_epoch = int(rec.get("step", -1))
            log.info(f"Resuming training from step {start_step}")
        
        # Load scheduler state if available (unless we're resetting steps)
        if not reset_steps:
            if lr_scheduler and 'scheduler' in rec:
                try:
                    lr_scheduler.load_state_dict(rec['scheduler'])
                    log.info("Loaded scheduler state from checkpoint")
                except Exception as e:
                    log.warning(f"Failed to load scheduler state: {e}. Creating new scheduler.")
                    lr_scheduler = get_lr_scheduler(opt, lr_scheduler_config, max_steps, 
                                                    last_epoch=scheduler_last_epoch,
                                                    current_epoch=resume_epoch, 
                                                    steps_per_epoch=steps_per_epoch)
        else:
            log.info("Skipping scheduler state load due to step reset")
        
        # Override learning rate if it's different from checkpoint
        # This ensures command-line learning rate overrides take effect
        new_base_lr = config.get('learning_rate')
        old_base_lr = opt.param_groups[0]['lr'] if opt.param_groups else new_base_lr
        
        if abs(new_base_lr - old_base_lr) > 1e-9:  # Check if LR changed
            log.info(f"Overriding learning rate from checkpoint: {old_base_lr:.2e} -> {new_base_lr:.2e}")
            
            # First, we need to recreate the optimizer with new base learning rates
            # This is necessary because schedulers cache the base_lrs at initialization
            
            # Get current optimizer state (momentum buffers, etc.)
            old_state_dict = opt.state_dict()
            
            # Extract learning rate multipliers from config
            custom_lr_multipliers = config.get('custom_lr_multipliers', {})
            projection_lr_multiplier = custom_lr_multipliers.get('projection_layers', 1.0)
            embedding_lr_multiplier = custom_lr_multipliers.get('embedding_layers', 1.0) 
            prompt_lr_multiplier = custom_lr_multipliers.get('prompt_layers', 1.0)
            base_model_lr_multiplier = custom_lr_multipliers.get('base_models', 1.0)
            
            # Recreate optimizer with new learning rate
            params = param_groups(
                [dec, enc], 
                new_base_lr,  # Use new base learning rate
                projection_lr_multiplier, 
                embedding_lr_multiplier, 
                prompt_lr_multiplier,
                base_model_lr_multiplier
            )
            opt = torch.optim.AdamW(params)
            
            # Restore optimizer state (momentum, etc.) but not the learning rates
            old_state = old_state_dict['state']
            if old_state:
                opt.load_state_dict({'state': old_state, 'param_groups': opt.state_dict()['param_groups']})
            
            log.info("Recreated optimizer with new base learning rate")
            log.info("Optimizer param groups after recreation:")
            for i, group in enumerate(opt.param_groups[:5]):
                log.info(f"  Group {i}: lr={group['lr']:.2e}")
            
            # Now recreate the scheduler with the new optimizer
            if lr_scheduler is not None:
                # For PyTorch schedulers, last_epoch actually means the last step count
                # We need to set it to start_step - 1 since it will be incremented on first step()
                scheduler_last_epoch = start_step - 1 if start_step > 0 else -1
                
                # Recreate the scheduler with the correct last_epoch
                lr_scheduler = get_lr_scheduler(opt, lr_scheduler_config, max_steps, 
                                                last_epoch=scheduler_last_epoch,
                                                current_epoch=resume_epoch, 
                                                steps_per_epoch=steps_per_epoch)
                
                current_lr_from_scheduler = lr_scheduler.get_last_lr()[0]
                log.info(f"Learning rate scheduler reinitialized at step {start_step}")
                log.info(f"Current LR from scheduler.get_last_lr(): {current_lr_from_scheduler:.6f}")
                
                # Check what the scheduler thinks the base LRs are
                if hasattr(lr_scheduler, 'base_lrs'):
                    log.info(f"Scheduler base_lrs: {[f'{lr:.6f}' for lr in lr_scheduler.base_lrs[:3]]}")

    epoch_decoded_tokens = []  # Initialize accumulator for decoded tokens per epoch
    step_iter = iter(train_loader)
    
    # Calculate start epoch from start step
    start_epoch = start_step // steps_per_epoch if steps_per_epoch > 0 else 0
    
    log.info(f"Start step: {start_step}, Start epoch: {start_epoch}, Max steps: {max_steps}")
    
    # Track validation loss for best checkpoint
    best_val_loss = float('inf')
    
    # Performance tracking
    step_times = deque(maxlen=100)  # Track last 100 step times
    start_time = time.time()
    last_log_time = start_time
    
    # Cache tokenized natural language prefix if specified
    cached_prefix_ids = None
    lm_loss_natural_prefix = config.get('lm_loss_natural_prefix')
    if lm_loss_natural_prefix:
        cached_prefix_ids = tokenizer(lm_loss_natural_prefix, add_special_tokens=False, return_tensors="pt").input_ids
        log.info(f"Cached natural language prefix: '{lm_loss_natural_prefix}' ({cached_prefix_ids.shape[1]} tokens)")
    
    # Track freeze schedule state
    start_epoch = start_step // steps_per_epoch if steps_per_epoch > 0 else 0
    non_adapters_frozen = freeze_schedule_enabled and not should_unfreeze_any_component(start_step, start_epoch, freeze_schedule_config, freeze_schedule_enabled)
    
    # Track warmup for newly unfrozen parameters
    unfreeze_warmup_duration = freeze_schedule_config.get('warmup_duration', "100s")
    unfreeze_warmup_steps = _resolve_schedule_to_steps(unfreeze_warmup_duration, steps_per_epoch, log, "warmup_duration", gradient_accumulation_steps)

    newly_unfrozen_params = set()
    unfreeze_transition_step = None
    
    # Create progress bar
    pbar = tqdm(range(start_step, max_steps),
                initial=start_step,
                total=max_steps,
                desc=f"Training",
                ncols=None,  # Fallback width if not TTY or if width detection fails
                dynamic_ncols=True,  # Adjust to terminal width changes; helps ensure description text fits well, supporting multi-line.
                mininterval=1.0,  # Update progress bar no more than once per second
                leave=True)

    enc.train()
    dec.train()
    orig.model.eval() # leave in validation mode?
    
    for step in pbar:
        # Calculate current epoch from step
        epoch = step // steps_per_epoch if steps_per_epoch > 0 else 0
        
        # Determine if this is the first step in an accumulation cycle
        is_accumulation_start = (step % gradient_accumulation_steps == 0)
        # Determine if we should perform optimizer step
        is_accumulation_end = ((step + 1) % gradient_accumulation_steps == 0) or (step == max_steps - 1)
        
        # Check if we should unfreeze non-adapter parameters
        if freeze_schedule_enabled and non_adapters_frozen and should_unfreeze_any_component(step, epoch, freeze_schedule_config, freeze_schedule_enabled):
            log.info(f"Unfreezing non-adapter parameters at step {step}")
            
            # Save current optimizer state
            opt_state = opt.state_dict()
            
            # Unfreeze and recreate optimizer
            opt, trainable_params, newly_unfrozen_params = unfreeze_non_adapters(
                dec_raw, enc_raw, config, learning_rate, projection_lr_multiplier, embedding_lr_multiplier, prompt_lr_multiplier, opt_state, step, epoch
            )
            
            # Recreate LR scheduler with new optimizer
            if lr_scheduler:
                lr_scheduler = get_lr_scheduler(opt, lr_scheduler_config, max_steps, 
                                                last_epoch=step-1,
                                                current_epoch=current_epoch,
                                                steps_per_epoch=steps_per_epoch)
            
            # Update frozen state
            non_adapters_frozen = False
            
            # Track the newly unfrozen parameters and transition step
            unfreeze_transition_step = step
            
            # Log the transition
            log_metrics({
                "freeze_schedule/transition_step": step,
                "freeze_schedule/non_adapters_frozen": 0,  # 0 = unfrozen
            }, step=step)
        
        step_start_time = time.time()
        current_epoch = step // steps_per_epoch if steps_per_epoch > 0 else 0  # 0-based epoch
        current_epoch_num = current_epoch + 1  # 1-based epoch for display
        epoch_just_finished = False
        
        try:
            batch = next(step_iter)
        except StopIteration:
            # Epoch finished
            epoch_just_finished = True
            
            # Log token stats
            if epoch_decoded_tokens:
                _log_epoch_token_statistics(epoch_decoded_tokens, tokenizer, current_epoch_num, step, log_interval, log)

            epoch_decoded_tokens = [] # Reset for the next epoch
            step_iter = iter(train_loader)# the sampler should shuffle, so we do not need to "set_epoch" like in distributed training
            batch = next(step_iter)

        # Move batch tensors to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Print first element of each batch item
        if step == 0:
            log.info("First batch contents:")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    if v.dim() > 0:
                        log.info(f"  {k}: {str(v[0])[:100]}")
                    else:
                        log.info(f"  {k}: {v} (scalar tensor)")
                else:
                    log.info(f"  {k}: {v}")


        # Only zero gradients at the start of accumulation cycles
        if is_accumulation_start:
            opt.zero_grad(set_to_none=True)
        
        ctx = get_autocast_context(device, mixed_precision_config)
        with ctx:
            current_tau = get_schedule_value(config['gumbel_tau_schedule'], step, max_steps,
                                            current_epoch, steps_per_epoch)
            current_alpha = get_schedule_value(config['alpha_schedule'], step, max_steps,
                                              current_epoch, steps_per_epoch)

            losses = train_step(
                batch,
                {"dec": dec, "enc": enc, "orig": orig},
                {
                    "tau": current_tau,
                    "t_text": t_text,
                    "alpha": current_alpha,
                    "lm_base_weight": lm_base_weight,
                    "kl_base_weight": kl_base_weight,
                    "entropy_weight": entropy_weight,
                    "mse_weight": config.get('mse_weight', 0.0),
                },
                lm_loss_natural_prefix=config.get('lm_loss_natural_prefix'),
                tokenizer=tokenizer,
                cached_prefix_ids=cached_prefix_ids,
                resample_ablation=config.get('resample_ablation')
            )
            loss = losses["total"]
            
            # Scale loss by gradient accumulation steps
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            
            # Accumulate decoded tokens from the current step
            if "decoded_tokens_batch" in losses:
                epoch_decoded_tokens.extend(losses["decoded_tokens_batch"].tolist())

        # Apply gradient scaling and backward pass (but not clipping yet)
        if device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Only perform gradient clipping and optimizer step at the end of accumulation
        if is_accumulation_end:
            # Unscale and clip gradients
            if device.type == "cuda":
                scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, config['grad_clip'])
            param_before = [p.detach().clone() for p in trainable_params]
        else:
            # Don't compute grad norm or param updates if not stepping
            grad_norm = None
            param_before = None
        
        # Only perform optimizer step and related operations at the end of accumulation
        if is_accumulation_end:
            # Apply warmup BEFORE optimizer step
            unfreeze_transition_step, should_clear_params = apply_unfreeze_warmup(
                opt, newly_unfrozen_params, unfreeze_transition_step, 
                unfreeze_warmup_steps, step, freeze_schedule_config, 
                log_interval, log
            )
            
            if should_clear_params:
                newly_unfrozen_params.clear()
            
            # Perform optimizer step
            optimizer_step(opt, scaler, device)

            with torch.no_grad():
                upd_sq = 0.0
                param_sq = 0.0
                for p, prev in zip(trainable_params, param_before):
                    diff = p.data - prev
                    upd_sq += diff.pow(2).sum().item()
                    param_sq += p.data.pow(2).sum().item()
                update_norm = math.sqrt(upd_sq)
                param_norm = math.sqrt(param_sq)
                update_ratio = update_norm / (param_norm + 1e-12)
            del param_before
            
            # Step the learning rate scheduler AFTER optimizer step
            if lr_scheduler:
                lr_scheduler.step()
        else:
            # Set dummy values for metrics when not stepping
            update_norm = 0.0
            param_norm = 0.0
            update_ratio = 0.0
        
        # Get the current learning rate
        lr_current = opt.param_groups[0]["lr"]
        
        # Track step time
        step_time = time.time() - step_start_time
        step_times.append(step_time)
        
        # Calculate performance metrics
        avg_step_time = sum(step_times) / len(step_times)
        steps_per_second = 1.0 / avg_step_time if avg_step_time > 0 else 0
        samples_per_second = steps_per_second * batch["A"].size(0)
        tokens_per_second = samples_per_second * config.get('t_text', 10)
        
        # Update progress bar with current metrics
        pbar.set_postfix({
            'loss': f'{losses["total"].item():.4f}',  # Show unscaled loss
            'lr': f'{lr_current:.1e}',
            'samples/s': f'{samples_per_second:.1f}',
            'eta': format_time((max_steps - step - 1) * avg_step_time),
            'tau': f'{current_tau:.2f}',
            'alpha': f'{current_alpha:.2f}',
            'acc': f'{(step % gradient_accumulation_steps) + 1}/{gradient_accumulation_steps}' if gradient_accumulation_steps > 1 else '',
        })

        # Log metrics at specified interval or at the last step
        if step % log_interval == 0 or step == max_steps - 1:
            # Only log to file/console at intervals to reduce clutter
            log_msg_parts = [
                f"Step {step}/{max_steps-1}",
                f"loss {losses['total'].item():.4f}",
                f"lr {lr_current:.1e}",
                f"{samples_per_second:.1f} samples/s",
            ]
            if gradient_accumulation_steps > 1:
                log_msg_parts.append(f"acc_step {(step % gradient_accumulation_steps) + 1}/{gradient_accumulation_steps}")
            log.info(" | ".join(log_msg_parts))
        if step % wandb_log_interval == 0 or step == max_steps - 1:
            # Get system metrics periodically (not every step to avoid overhead)
            sys_metrics = {}
            if step % (wandb_log_interval * 10) == 0:  # Every 10 wandb logs
                sys_metrics = get_system_metrics(device)
            
            metrics_to_log = {
                "loss/total": losses["total"].item(),  # Use unscaled loss for logging
                "loss/mse": losses["mse"].item(),
                "loss/lm": losses["lm"].item(),
                "loss/kl": losses["kl"].item(),
                "loss/entropy": losses["entropy"].item(),
                "params/tau": current_tau,
                "params/alpha": current_alpha,
                "params/lm_w": lm_base_weight,
                "params/kl_w": kl_base_weight,
                "params/entropy_w": entropy_weight,
                "optim/lr": lr_current,
                "grads/norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else (float(grad_norm) if grad_norm is not None else None),
                "updates/norm": update_norm,
                "updates/ratio": update_ratio,
                # Gradient accumulation
                "gradient_accumulation/is_update_step": int(is_accumulation_end),
                "gradient_accumulation/accumulation_step": step % gradient_accumulation_steps,
                # Performance metrics
                "performance/steps_per_second": steps_per_second,
                "performance/samples_per_second": samples_per_second,
                "performance/tokens_per_second": tokens_per_second,
                "performance/avg_step_time": avg_step_time,
            }
            
            # Add freeze schedule state
            if freeze_schedule_enabled:
                metrics_to_log["freeze_schedule/non_adapters_frozen"] = 1 if non_adapters_frozen else 0
            
            # Add system metrics if available
            if sys_metrics:
                metrics_to_log.update({
                    "system/cpu_percent": sys_metrics.get('cpu_percent', 0),
                    "system/memory_percent": sys_metrics.get('memory_percent', 0),
                })
                if 'gpu_utilization' in sys_metrics:
                    metrics_to_log.update({
                        "system/gpu_utilization": sys_metrics['gpu_utilization'],
                        "system/gpu_memory_percent": sys_metrics['gpu_memory_percent'],
                        "system/gpu_temperature": sys_metrics['gpu_temperature'],
                    })
            
            if steps_per_epoch > 0:
                metrics_to_log["epoch"] = current_epoch_num
            
            log_metrics(metrics_to_log, step=step)

        # ------------------------------------------------------------------
        # Checkpoint
        # ------------------------------------------------------------------
        # Check if we should save based on step interval
        if checkpoint_manager.should_save_step(step):
            current_metrics = {
                "loss/total": loss.item(),
                "loss/mse": losses["mse"].item(),
                "loss/lm": losses["lm"].item(),
                "loss/kl": losses["kl"].item(),
                "loss/entropy": losses["entropy"].item(),
                "params/tau": current_tau,
                "params/alpha": current_alpha,
                "optim/lr": lr_current,
                "grads/norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else (float(grad_norm) if grad_norm is not None else None),
            }
            
            checkpoint_manager.save_checkpoint(
                step=step,
                epoch=current_epoch_num,
                models={"dec": dec_raw if config.get('compile_models', True) else dec, 
                        "enc": enc_raw if config.get('compile_models', True) else enc},
                optimizer=opt,
                scheduler=lr_scheduler,
                metrics=current_metrics,
                config=config,
                tau=current_tau,
                alpha=current_alpha,
                wandb_run_id=current_wandb_run_id,
            )
        
        # Check if we should save based on epoch completion
        if epoch_just_finished and checkpoint_manager.should_save_epoch(current_epoch_num, epoch_just_finished):
            current_metrics = {
                "loss/total": loss.item(),
                "loss/mse": losses["mse"].item(),
                "loss/lm": losses["lm"].item(),
                "loss/kl": losses["kl"].item(),
                "loss/entropy": losses["entropy"].item(),
                "params/tau": current_tau,
                "params/alpha": current_alpha,
                "optim/lr": lr_current,
                "grads/norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else (float(grad_norm) if grad_norm is not None else None),
            }
            
            checkpoint_manager.save_checkpoint(
                step=step,
                epoch=current_epoch_num,
                models={"dec": dec_raw if config.get('compile_models', True) else dec, 
                        "enc": enc_raw if config.get('compile_models', True) else enc},
                optimizer=opt,
                scheduler=lr_scheduler,
                metrics=current_metrics,
                config=config,
                tau=current_tau,
                alpha=current_alpha,
                wandb_run_id=current_wandb_run_id,
            )

        # run validation at interval
        if val_loader and val_interval > 0 and step % val_interval == 0:
            validation_metrics = run_validation_step(
                dec=dec,
                enc=enc,
                orig=orig,
                val_loader=val_loader,
                config=config,
                tokenizer=tokenizer,
                cached_prefix_ids=cached_prefix_ids,
                device=device,
                current_step=step,
                current_epoch=current_epoch, # current_epoch is 0-based
                max_steps=max_steps,
                steps_per_epoch=steps_per_epoch,
                log=log
            )
            avg_val_loss = validation_metrics["avg_val_loss"]
            normalized_val_loss = validation_metrics["normalized_val_loss"]
            
            # Save checkpoint if validation loss tracking is enabled
            if checkpoint_manager.track_best_n > 0 and not math.isnan(avg_val_loss):
                val_metrics_for_ckpt = {
                    "loss/total": losses["total"].item(), # from training step
                    "loss/mse": losses["mse"].item(),     # from training step
                    "loss/lm": losses["lm"].item(),       # from training step
                    "loss/kl": losses["kl"].item(),       # from training step
                    "loss/entropy": losses["entropy"].item(), # from training step
                    "eval/loss/total": avg_val_loss,
                    "eval/loss/mse": validation_metrics["avg_val_mse"],
                    "eval/loss/lm": validation_metrics["avg_val_lm"],
                    "eval/loss/kl": validation_metrics["avg_val_kl"],
                    "params/tau": current_tau,
                    "params/alpha": current_alpha,
                    "optim/lr": lr_current,
                    "grads/norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else (float(grad_norm) if grad_norm is not None else None),
                }
                
                checkpoint_manager.save_checkpoint(
                    step=step,
                    epoch=current_epoch_num,
                    models={"dec": dec_raw if config.get('compile_models', True) else dec, 
                            "enc": enc_raw if config.get('compile_models', True) else enc},
                    optimizer=opt,
                    scheduler=lr_scheduler,
                    metrics=val_metrics_for_ckpt,
                    config=config,
                    val_loss=normalized_val_loss,  # Use normalized loss for best checkpoint tracking
                    raw_val_loss=avg_val_loss,     # Keep raw loss for logging
                    tau=current_tau,
                    alpha=current_alpha,
                    wandb_run_id=current_wandb_run_id,
                )
                
            # dec.train() and enc.train() are called at the end of run_validation_step

        # Verbose sample printing
        verbose_config = config.get('verbose_samples', {})
        if verbose_config.get('enabled', False):
            # In your evaluation loop, after getting a batch:
            if  step == 0:  # Test on first batch
                do_all_initial_validation(batch, orig, tokenizer, device, log, activation_dir)

            should_print = False
            
            # Check if we should print based on flexible interval notation
            verbose_interval_str = verbose_config.get('interval', "1000s")
            try:
                verbose_interval = _resolve_schedule_to_steps(verbose_interval_str, steps_per_epoch, log, "verbose_interval", gradient_accumulation_steps)
                if step % verbose_interval == 0:
                    should_print = True
            except Exception as e:
                log.warning(f"Failed to parse verbose_samples interval '{verbose_interval_str}': {e}")
                should_print = False
            
            # Also check for combined epoch+steps printing
            if verbose_config.get('print_every_epoch', False) and epoch_just_finished:
                should_print = True
            if verbose_config.get('print_every_n_steps', 0) > 0:
                if step > 0 and step % verbose_config['print_every_n_steps'] == 0:
                    should_print = True
            
            if should_print:
                log.info(f"Generating verbose samples at step {step}, epoch {current_epoch_num}")
                dec.eval()
                enc.eval()
                
                # Get schedule arguments
                sch_args = {
                    "tau": get_schedule_value(config['gumbel_tau_schedule'], step, max_steps,
                                             current_epoch, steps_per_epoch),
                    "t_text": t_text,
                    "alpha": get_schedule_value(config['alpha_schedule'], step, max_steps,
                                               current_epoch, steps_per_epoch),
                    "lm_base_weight": lm_base_weight,
                    "kl_base_weight": kl_base_weight,
                    "entropy_weight": entropy_weight,
                }
                
                # Process verbose samples
                result = process_and_print_verbose_batch_samples(
                    batch=batch,  # Use current batch
                    cfg=config,
                    models={"dec": dec, "enc": enc},
                    orig=orig,
                    tok=tokenizer,
                    sch_args=sch_args,
                    device=device,
                    num_samples=verbose_config.get('num_samples', 2),
                    top_n_analysis=verbose_config.get('top_n_predictions', 3),
                    printed_count_so_far=0,
                    generate_continuation=verbose_config.get('generate_continuation', True),
                    continuation_tokens=verbose_config.get('continuation_tokens', 30),
                    return_structured_data=False,
                    capture_output=True,
                    cached_prefix_ids=cached_prefix_ids,  # Pass the cached prefix for loss computation
                    resample_ablation=config.get('resample_ablation')

                )
                
                # Handle return value based on whether we got captured output
                num_printed, captured_text = result
                # Log to wandb if available
                if captured_text and current_wandb_run_id:
                    verbose_samples_logger.log_verbose_samples(captured_text, 
                        step=step,
                        table_name="training_verbose_samples",
                        limit_rows=verbose_config.get('wandb_table_limit', False)
                    )
                
                dec.train()
                enc.train()

    # Save final checkpoint if enabled
    if checkpoint_manager.save_at_end:
        log.info("Saving final checkpoint...")
        final_metrics = {
            "loss/total": loss.item() if 'loss' in locals() else None,
            "step": max_steps - 1,
            "epoch": num_epochs_total_approx,
        }
        
        checkpoint_manager.save_checkpoint(
            step=max_steps - 1,
            epoch=num_epochs_total_approx,
            models={"dec": dec_raw if config.get('compile_models', True) else dec, 
                    "enc": enc_raw if config.get('compile_models', True) else enc},
            optimizer=opt,
            scheduler=lr_scheduler,
            metrics=final_metrics,
            config=config,
            tau=current_tau,
            alpha=current_alpha,
            wandb_run_id=current_wandb_run_id,
        )

    # Close progress bar
    pbar.close()
    
    # Training summary
    log.info("=" * 60)
    log.info("TRAINING COMPLETE")
    log.info("=" * 60)
    log.info(f"Run Name: {run_name}")
    log.info(f"Total Steps: {max_steps}")
    log.info(f"Final Step: {step}")
    log.info(f"Checkpoint Directory: {run_checkpoint_dir}")
    
    # Find best checkpoint if tracking
    best_ckpt_path = checkpoint_manager.get_best_checkpoint_path()
    if best_ckpt_path:
        log.info(f"Best Checkpoint: {best_ckpt_path.name}")
    
    # Log final hyperparameters
    log.info(f"Final tau: {current_tau:.4f}")
    log.info(f"Final alpha: {current_alpha:.4f}")
    log.info(f"Final LR: {lr_current:.2e}")
    
    # If wandb is active, log the run URL
    if current_wandb_run_id:
        log.info(f"WandB Run ID: {current_wandb_run_id}")
    
    log.info("=" * 60)

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

if __name__ == "__main__":
    main()
