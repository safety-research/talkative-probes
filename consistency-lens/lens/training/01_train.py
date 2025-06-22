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
                "T_text": t_text,
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

if __name__ == "__main__":
    main()
