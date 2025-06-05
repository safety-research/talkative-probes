#!/usr/bin/env python3
"""Distributed training script for Consistency Lens with multi-GPU support and proper gradient accumulation."""

import logging
import math
import os
import time
import warnings
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

# Enable TF32 for better performance on Ampere GPUs (A100, H100)
torch.set_float32_matmul_precision('high')

from lens.data.collate import collate
from lens.data.dataset import ActivationDataset
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.orig import OrigWrapper
from lens.training.loop import train_step as original_train_step
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
from lens.training.distributed import (
    init_distributed,
    cleanup_distributed,
    setup_for_distributed,
    is_main,
    get_rank,
    get_world_size,
    get_local_rank,
    reduce_dict,
)
from lens.utils.checkpoint_manager import CheckpointManager
from lens.training.distributed import set_seed
from lens.evaluation.wandb_logger import verbose_samples_logger
from lens.training.fast_distributed_sampler import FastDistributedSampler
from lens.training.test import diagnose_activation_mismatch, diagnose_activation_save_load, check_dataset_activation_format, test_autocast_difference, check_layer_indexing, test_decoder_generation

# Import all the utility functions from the original training script
sys.path.insert(0, str(Path(__file__).parent))
import importlib
train_module = importlib.import_module("01_train")
extract_dataset_info = train_module.extract_dataset_info
resolve_path = train_module.resolve_path
generate_run_name = train_module.generate_run_name  
prepare_dataset_and_loaders = train_module._prepare_dataloaders
_resolve_schedule_to_steps = train_module._resolve_schedule_to_steps
process_and_print_verbose_batch_samples = train_module.process_and_print_verbose_batch_samples
_get_hydra_config_name = train_module._get_hydra_config_name
get_system_metrics = train_module.get_system_metrics
do_all_initial_validation = train_module.do_all_initial_validation


def get_initial_model_state(model: torch.nn.Module) -> dict:
    """Capture initial trainable parameters (always stored on CPU)."""
    state = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            state[name] = p.detach().cpu().clone()   #  <-- keep on CPU
    return state

def log_parameter_drift(
    model: torch.nn.Module,
    initial_state: dict,
    model_prefix: str,
    step: int,
    logger_fn,  # e.g., log_metrics from lens.utils.logging (for W&B)
    log: logging.Logger, # Python logger
    is_main_process: bool
):
    """Calculates and logs parameter drift from their initial states."""
    metrics_to_log = {}
    param_group_drifts = {}  # To store sum of drifts and sum of initial norms per group

    for name, param in model.named_parameters():
        if param.requires_grad and name in initial_state:
            initial_p = initial_state[name].to(param.device)
            current_p = param.data
            
            abs_drift = torch.norm(current_p - initial_p)
            norm_initial_p = torch.norm(initial_p)
            # rel_drift is calculated per parameter but not directly logged per parameter to avoid too many metrics.
            # We will calculate an overall relative drift for the group.

            group = "other_trainable"
            # Determine group for parameter based on its name
            if name == "soft_prompt_embeddings":  # Encoder specific
                group = "prompts"
            elif name == "prompt_left_emb" or name == "prompt_right_emb":  # Decoder specific
                group = "prompts"
            elif name.startswith("proj."):
                group = "projection"
            elif name.startswith("out."):  # Decoder specific output head
                group = "output_head"
            elif name.startswith("base."):
                if any(emb_keyword in name for emb_keyword in [".wte.", ".wpe.", ".embed_tokens.", ".word_embeddings.", ".position_embeddings."]):
                    group = "base_model_input_embeddings"
                elif any(layer_keyword in name for layer_keyword in [".h.", ".layers.", ".layer.", ".block.", ".attention.", ".mlp."]):
                    group = "base_model_transformer_layers"
                else:
                    group = "base_model_other"
            
            if group not in param_group_drifts:
                param_group_drifts[group] = {'sum_abs_drift': 0.0, 'sum_norm_initial': 0.0, 'count': 0}
            
            param_group_drifts[group]['sum_abs_drift'] += abs_drift.item()
            param_group_drifts[group]['sum_norm_initial'] += norm_initial_p.item()
            param_group_drifts[group]['count'] += 1
        elif param.requires_grad and name not in initial_state and is_main_process:
            log.warning(f"Trainable parameter {name} in {model_prefix} not found in initial_state for drift calculation.")

    for group, data in param_group_drifts.items():
        if data['count'] > 0:
            avg_abs_drift = data['sum_abs_drift'] / data['count']
            # Overall relative drift for the group: Sum(||curr-init||) / Sum(||init||)
            overall_rel_drift_group = data['sum_abs_drift'] / (data['sum_norm_initial'] + 1e-9)
                            
            metrics_to_log[f"drift/{model_prefix}/{group}/avg_abs_drift"] = avg_abs_drift
            metrics_to_log[f"drift/{model_prefix}/{group}/overall_rel_drift"] = overall_rel_drift_group
        elif is_main_process:
            log.warning(f"Parameter group {group} for {model_prefix} had 0 parameters for drift calculation. This is unexpected.")

    if metrics_to_log and is_main_process:
        logger_fn(metrics_to_log, step=step)


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


def distributed_train_step(
    decoder,
    encoder, 
    orig_model,
    batch,
    optimizer,
    scaler,
    config,
    step,
    device,
    tokenizer,
    cached_prefix_ids,
    gradient_accumulation_steps=1,
    is_distributed=False,
    lr_scheduler=None,
    steps_per_epoch=None,
):
    """Training step with proper gradient accumulation for distributed training.
    
    Key optimization: Only synchronize gradients at accumulation boundaries
    to minimize communication overhead.
    """
    # Determine accumulation boundaries
    is_accumulation_start = (step % gradient_accumulation_steps == 0)
    is_accumulation_end = ((step + 1) % gradient_accumulation_steps == 0) or (step == config['max_train_steps'] - 1)
    
    # Get base models if using DDP
    decoder_base = decoder.module if hasattr(decoder, 'module') else decoder
    encoder_base = encoder.module if hasattr(encoder, 'no_sync') else encoder
    
    # Prepare models dict for train_step
    models = {
        "dec": decoder_base,
        "enc": encoder_base,
        "orig": orig_model
    }
    
    # Loss function parameters
    loss_fns = {
        "T_text": config.get('t_text', 8),
        "tau": get_schedule_value(config['gumbel_tau_schedule'], step, config['max_train_steps'], 
                                current_epoch=step // steps_per_epoch if steps_per_epoch > 0 else 0, 
                                steps_per_epoch=steps_per_epoch, 
                                grad_accum_steps=gradient_accumulation_steps),
        "alpha": get_schedule_value(config['alpha_schedule'], step, config['max_train_steps'],
                                  current_epoch=step // steps_per_epoch if steps_per_epoch > 0 else 0, 
                                  steps_per_epoch=steps_per_epoch, 
                                  grad_accum_steps=gradient_accumulation_steps),
        "kl_base_weight": config.get('kl_base_weight', 1.0),
        "entropy_weight": config.get('entropy_weight', 0.0),
        "mse_weight": config.get('mse_weight', 0.0),
        "lm_base_weight": config.get('lm_base_weight'),
    }
    
    decoder_no_sync_cm = nullcontext()
    encoder_no_sync_cm = nullcontext()

    if is_distributed and not is_accumulation_end:
        # If decoder is DDP, decoder.no_sync() is the correct context manager
        if hasattr(decoder, 'no_sync'):
            decoder_no_sync_cm = decoder.no_sync()
        # If encoder is DDP, encoder.no_sync() is the correct context manager
        if hasattr(encoder, 'no_sync'): 
            encoder_no_sync_cm = encoder.no_sync()
    
    # The 'with' statement handles __enter__ and __exit__ for both contexts.
    # The original try...finally for manual exit calls is no longer needed here.
    with decoder_no_sync_cm, encoder_no_sync_cm:
        # Get autocast context based on mixed precision config
        mixed_precision_config = config.get('mixed_precision', {'enabled': True, 'dtype': 'auto'})
        autocast_context = get_autocast_context(device, mixed_precision_config)
        
        # Forward pass with autocast
        with autocast_context:
            losses = original_train_step(
                batch=batch,
                models=models,
                _loss_fns=loss_fns,
                lm_loss_natural_prefix=config.get('lm_loss_natural_prefix'),
                tokenizer=tokenizer,
                cached_prefix_ids=cached_prefix_ids,
                resample_ablation=config.get('resample_ablation', True)
            )
            
            # Scale loss by accumulation steps
            loss = losses['total'] / gradient_accumulation_steps
        
        # Backward pass
        if device.type == "cuda" and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
    
    # Gradient clipping and optimizer step only at accumulation boundaries
    if is_accumulation_end:
        if device.type == "cuda" and scaler is not None:
            scaler.unscale_(optimizer)
        
        # Get all parameters for gradient clipping
        all_params = list(decoder_base.parameters()) + list(encoder_base.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(all_params, config['grad_clip'])
        
        # Capture parameters before update for update norm calculation (if enabled)
        compute_update_norm = config.get('compute_update_norm', True)
        if compute_update_norm:
            trainable_params = [p for p in all_params if p.requires_grad]
            param_before = [p.detach().clone() for p in trainable_params]
        
        # Optimizer step
        if device.type == "cuda" and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Calculate update norm and ratio (if enabled)
        if compute_update_norm:
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
        else:
            update_norm = 0.0
            param_norm = 0.0
            update_ratio = 0.0

        # Update learning rate scheduler AFTER optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        # Zero gradients for next iteration
        optimizer.zero_grad(set_to_none=True)
    else:
        grad_norm = None
        update_norm = 0.0
        param_norm = 0.0
        update_ratio = 0.0
    
    # Return metrics
    metrics = {
        'loss': losses['total'].item(),
        'loss_mse': losses['mse'].item(),
        'loss_lm': losses['lm'].item(),
        'loss_kl': losses['kl'].item(),
        'loss_entropy': losses.get('entropy', 0.0).item() if 'entropy' in losses else 0.0,
        'grad_norm': grad_norm.item() if grad_norm is not None else 0.0,
        'update_norm': update_norm,
        'param_norm': param_norm,
        'update_ratio': update_ratio,
    }
    
    return metrics


def setup_distributed_models(decoder, encoder, orig_model, device, rank, world_size):
    """Wrap models with DistributedDataParallel.
    
    Args:
        decoder: Decoder model
        encoder: Encoder model
        orig_model: Original model wrapper
        device: Device to use
        rank: Current process rank
        world_size: Total number of processes
        
    Returns:
        Tuple of (decoder, encoder, orig_model) wrapped with DDP if needed
    """
    if world_size > 1:
        # Move models to device before wrapping with DDP
        decoder = decoder.to(device)
        encoder = encoder.to(device)
        orig_model = orig_model.to(device)
        
        # Wrap with DDP
        # Note: We don't wrap orig_model as it's frozen and doesn't need gradients
        decoder = DDP(
            decoder, 
            device_ids=[rank], 
            output_device=rank,
            find_unused_parameters=False,  # Set to True if you have unused parameters
            gradient_as_bucket_view=True,  # Memory optimization
        )
        encoder = DDP(
            encoder, 
            device_ids=[rank], 
            output_device=rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        
    else:
        # Single GPU - just move to device
        decoder = decoder.to(device)
        encoder = encoder.to(device) 
        orig_model = orig_model.to(device)
        
    return decoder, encoder, orig_model


def get_dataloader_for_distributed(dataset, batch_size, world_size, rank, shuffle=True, **kwargs):
    """Create a DataLoader with DistributedSampler if needed.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size per GPU
        world_size: Total number of processes
        rank: Current process rank
        shuffle: Whether to shuffle data
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        DataLoader configured for distributed training
    """
    if world_size > 1:
        sampler = FastDistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
        # Remove shuffle from kwargs since we're using a sampler
        kwargs.pop('shuffle', None)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, **kwargs)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    
    return dataloader


def sync_metrics(metrics_dict, world_size):
    """Synchronize metrics across all processes.
    
    Args:
        metrics_dict: Dictionary of metrics to synchronize
        world_size: Number of processes
        
    Returns:
        Dictionary of synchronized metrics
    """
    if world_size <= 1:
        return metrics_dict
    
    # Convert all metrics to tensors
    tensor_dict = {}
    for key, value in metrics_dict.items():
        if isinstance(value, torch.Tensor):
            tensor_dict[key] = value.clone()
        else:
            tensor_dict[key] = torch.tensor(value, dtype=torch.float32, device='cuda')
    
    # Reduce across all processes
    synced_dict = reduce_dict(tensor_dict, average=True)
    
    # Convert back to regular Python types
    result_dict = {}
    for key, value in synced_dict.items():
        result_dict[key] = value.item()
    
    return result_dict


def validate_distributed(
    decoder,
    encoder,
    orig_model,
    val_loader,
    config,
    step,
    device,
    tokenizer,
    cached_prefix_ids,
    world_size,
    log,
    is_main_process,
    wandb_run_id=None,
    steps_per_epoch=None
):
    """Distributed validation function using train_step without gradients.
    
    Tracks activation statistics and computes validation metrics.
    """
    # Get base models (unwrap DDP if needed)
    decoder_base = decoder.module if hasattr(decoder, 'module') else decoder
    encoder_base = encoder.module if hasattr(encoder, 'module') else encoder
    
    # Put models in eval mode
    decoder_base.eval()
    encoder_base.eval()
    orig_model.model.eval()
    
    # Prepare models dict for train_step
    models = {
        "dec": decoder_base,
        "enc": encoder_base,
        "orig": orig_model
    }
    
    # Loss function parameters (use same as training)
    loss_fns = {
        "T_text": config.get('t_text', 8),
        "tau": get_schedule_value(config['gumbel_tau_schedule'], step, config['max_train_steps'], 
                                current_epoch=step // steps_per_epoch if steps_per_epoch and steps_per_epoch > 0 else 0, 
                                steps_per_epoch=steps_per_epoch, 
                                grad_accum_steps=config['gradient_accumulation_steps']),
        "alpha": get_schedule_value(config['alpha_schedule'], step, config['max_train_steps'],
                                  current_epoch=step // steps_per_epoch if steps_per_epoch and steps_per_epoch > 0 else 0, 
                                  steps_per_epoch=steps_per_epoch, 
                                  grad_accum_steps=config['gradient_accumulation_steps']),
        "kl_base_weight": config.get('kl_base_weight', 1.0),
        "entropy_weight": config.get('entropy_weight', 0.0),
        "mse_weight": config.get('mse_weight', 0.0),
        "lm_base_weight": config.get('lm_base_weight'),
    }
    
    # Metrics accumulators
    total_loss = 0.0
    total_mse = 0.0
    total_lm = 0.0
    total_kl = 0.0
    total_entropy = 0.0
    num_batches = 0
    
    # Activation statistics accumulators
    all_activations = []
    all_reconstructions = []
    
    # Limit validation batches for efficiency
    max_val_batches = config.get('max_val_batches', 50)
    
    # No gradients needed for validation
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_val_batches:
                break
                
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass using train_step (but with no_grad context)
            losses = original_train_step(
                batch=batch,
                models=models,
                _loss_fns=loss_fns,
                lm_loss_natural_prefix=config.get('lm_loss_natural_prefix'),
                tokenizer=tokenizer,
                cached_prefix_ids=cached_prefix_ids,
                resample_ablation=config.get('resample_ablation', True)
            )
            
            # Accumulate losses
            total_loss += losses['total'].item()
            total_mse += losses['mse'].item()
            total_lm += losses['lm'].item()
            total_kl += losses['kl'].item()
            total_entropy += losses.get('entropy', 0.0).item()
            num_batches += 1
            
            # Collect activation statistics
            if batch_idx < 10:  # Only collect for first 10 batches to avoid memory issues
                activations = batch['A'].float()
                all_activations.append(activations.cpu())
                
                # Get reconstructions through the encoder-decoder pipeline
                gen_result = decoder_base.generate_soft(
                    activations, 
                    max_length=config.get('t_text', 8), 
                    gumbel_tau=loss_fns['tau']
                )
                reconstructions = encoder_base(gen_result.generated_text_embeddings)
                all_reconstructions.append(reconstructions.cpu())
    
    # Compute average losses across all processes
    if world_size > 1:
        # Create tensor dict for reduction
        metrics_tensor = torch.tensor([
            total_loss,
            total_mse,
            total_lm,
            total_kl,
            total_entropy,
            float(num_batches)
        ], device=device)
        
        # All-reduce
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        # Extract reduced values
        total_loss = metrics_tensor[0].item()
        total_mse = metrics_tensor[1].item()
        total_lm = metrics_tensor[2].item()
        total_kl = metrics_tensor[3].item()
        total_entropy = metrics_tensor[4].item()
        num_batches = int(metrics_tensor[5].item())
    
    # Compute averages
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_mse = total_mse / num_batches if num_batches > 0 else 0.0
    avg_lm = total_lm / num_batches if num_batches > 0 else 0.0
    avg_kl = total_kl / num_batches if num_batches > 0 else 0.0
    avg_entropy = total_entropy / num_batches if num_batches > 0 else 0.0
    
    # Compute activation statistics (only on main process)
    if is_main_process and all_activations:
        all_activations = torch.cat(all_activations, dim=0)
        all_reconstructions = torch.cat(all_reconstructions, dim=0)
        
        # Original activation statistics
        act_mean = all_activations.mean().item()
        act_std = all_activations.std().item()
        act_min = all_activations.min().item()
        act_max = all_activations.max().item()
        
        # Reconstruction statistics
        recon_mean = all_reconstructions.mean().item()
        recon_std = all_reconstructions.std().item()
        recon_min = all_reconstructions.min().item()
        recon_max = all_reconstructions.max().item()
        
        # Baseline comparisons
        zero_mse = (all_activations ** 2).mean().item()
        mean_mse = ((all_activations - all_activations.mean()) ** 2).mean().item()
        
        # Reconstruction error (should match avg_mse)
        reconstruction_mse = ((all_activations - all_reconstructions) ** 2).mean().item()
        
        # Additional statistics
        # Correlation between original and reconstructed
        act_flat = all_activations.flatten()
        recon_flat = all_reconstructions.flatten()
        if len(act_flat) > 1:
            correlation = torch.corrcoef(torch.stack([act_flat, recon_flat]))[0, 1].item()
        else:
            correlation = 0.0
        
        # Relative error
        relative_error = (torch.abs(all_activations - all_reconstructions) / (torch.abs(all_activations) + 1e-8)).mean().item()
        
        log.info(f"\n{'='*60}")
        log.info(f"Validation Results at Step {step}")
        log.info(f"{'='*60}")
        log.info(f"Average Losses:")
        log.info(f"  Total Loss: {avg_loss:.4f}")
        log.info(f"  MSE Loss: {avg_mse:.4f}")
        log.info(f"  LM Loss: {avg_lm:.4f}")
        log.info(f"  KL Loss: {avg_kl:.4f}")
        log.info(f"  Entropy: {avg_entropy:.4f}")
        log.info(f"\nActivation Statistics:")
        log.info(f"  Original - Mean: {act_mean:.4f}, Std: {act_std:.4f}")
        log.info(f"  Original - Min: {act_min:.4f}, Max: {act_max:.4f}")
        log.info(f"  Reconstructed - Mean: {recon_mean:.4f}, Std: {recon_std:.4f}")
        log.info(f"  Reconstructed - Min: {recon_min:.4f}, Max: {recon_max:.4f}")
        log.info(f"\nBaseline Comparisons:")
        log.info(f"  Zero MSE (predicting zeros): {zero_mse:.4f}")
        log.info(f"  Mean MSE (predicting mean): {mean_mse:.4f}")
        log.info(f"  Our Reconstruction MSE: {reconstruction_mse:.4f}")
        log.info(f"  Improvement over zero baseline: {(zero_mse - reconstruction_mse) / zero_mse * 100:.1f}%")
        log.info(f"  Improvement over mean baseline: {(mean_mse - reconstruction_mse) / mean_mse * 100:.1f}%")
        log.info(f"\nAdditional Metrics:")
        log.info(f"  Correlation (original vs reconstructed): {correlation:.4f}")
        log.info(f"  Mean Relative Error: {relative_error:.4f}")
        log.info(f"{'='*60}\n")
        
        # Log to wandb
        if wandb_run_id:
            val_metrics = {
                'val/loss': avg_loss,
                'val/loss_mse': avg_mse,
                'val/loss_lm': avg_lm,
                'val/loss_kl': avg_kl,
                'val/loss_entropy': avg_entropy,
                'val/activation_mean': act_mean,
                'val/activation_std': act_std,
                'val/activation_min': act_min,
                'val/activation_max': act_max,
                'val/reconstruction_mean': recon_mean,
                'val/reconstruction_std': recon_std,
                'val/reconstruction_min': recon_min,
                'val/reconstruction_max': recon_max,
                'val/baseline_zero_mse': zero_mse,
                'val/baseline_mean_mse': mean_mse,
                'val/reconstruction_mse': reconstruction_mse,
                'val/improvement_over_zero': (zero_mse - reconstruction_mse) / zero_mse * 100,
                'val/improvement_over_mean': (mean_mse - reconstruction_mse) / mean_mse * 100,
                'val/correlation': correlation,
                'val/relative_error': relative_error,
            }
            log_metrics(val_metrics, step=step)
    
    # Put models back in train mode
    decoder_base.train()
    encoder_base.train()
    # orig_model typically stays in eval mode



@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Distributed training entry point with optimized gradient accumulation."""
    
    overall_start_time = time.time() # For overall script timing

    # Initialize distributed training
    rank, world_size, local_rank = init_distributed()
    
    # Setup for distributed (disable printing on non-master processes)
    setup_for_distributed(is_main())
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    
    # Convert Hydra config to a plain Python dict for most operations,
    # but CheckpointManager will use the original cfg.
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Parse flexible schedule notations
    config = parse_schedule_config(config)
    
    # Logging setup (only on main process)
    if is_main():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        logging.basicConfig(level=logging.WARNING)
    
    log = logging.getLogger(__name__)
    
    # --- Timer for Initial Setup ---
    with Timer("Initial Setup (Paths, Tokenizer, Run Name)", log, main_process=is_main()):
        # Load tokenizer (all processes will have it, but primarily used by main for logging unless train_step needs it)
        tokenizer_name = config.get("tokenizer_name", config['model_name'])
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if is_main():
            log.info(f"Tokenizer loaded: {tokenizer_name}")
        
        # Cache tokenized natural language prefix if specified (all processes do this, small overhead)
        cached_prefix_ids = None
        lm_loss_natural_prefix_text = config.get('lm_loss_natural_prefix') # Assuming this key holds the text
        if lm_loss_natural_prefix_text and isinstance(lm_loss_natural_prefix_text, str) : # Check if it's a string
            cached_prefix_ids = tokenizer(lm_loss_natural_prefix_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            if is_main():
                log.info(f"Cached natural language prefix: '{lm_loss_natural_prefix_text}' ({cached_prefix_ids.shape[1]} tokens)")
        elif config.get('lm_loss_natural_prefix') is True: # Handle boolean true case if it implies a default prefix or other logic
            if is_main():
                log.warning("lm_loss_natural_prefix is True but not a string. Cannot cache prefix IDs without prefix text.")
        
        # Extract configuration values
        model_name = config['model_name']
        layer_l = config.get('layer_l', 5)
        out_layer = config['trainable_components']['encoder']['output_layer']
        
        # Setup paths (same as original)
        cli_activation_dir = config.get('activation_dir')
        base_activation_dir_str = cli_activation_dir if cli_activation_dir is not None else config['activation_dumper']['output_dir']
        base_activation_path = resolve_path(base_activation_dir_str)
        model_name_clean = config['model_name'].replace("/", "_")
        activation_dir = str(base_activation_path.parent / model_name_clean / f"layer_{layer_l}" / base_activation_path.name)
        
        # Validation activation directory
        base_val_activation_dir_str = config.get('val_activation_dir')
        effective_val_activation_dir = None
        if base_val_activation_dir_str:
            base_val_path = resolve_path(base_val_activation_dir_str)
            effective_val_activation_dir = str(base_val_path.parent / model_name_clean / f"layer_{layer_l}" / base_val_path.name)
        
        # Training parameters
        max_steps = config['max_train_steps']
        learning_rate = config['learning_rate']
        batch_size = config['batch_size']
        gradient_accumulation_steps = config['gradient_accumulation_steps']
        
        # Adjust batch size for distributed training
        effective_batch_size = batch_size * gradient_accumulation_steps * world_size
        
        if is_main():
            log.info(f"Distributed training with {world_size} GPUs")
            log.info(f"Per-GPU batch size: {batch_size}")
            log.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
            log.info(f"Effective batch size: {effective_batch_size}")
            log.info(f"Gradient sync: Every {gradient_accumulation_steps} steps")
        
        # Set random seed for reproducibility
        seed = config.get('seed', 42)
        set_seed(seed + rank)  # Different seed per rank
        
        # Dataset info
        dataset_info = extract_dataset_info(activation_dir)
        
        # Run name generation (only on main process)
        if is_main():
            config_name = _get_hydra_config_name()
            
            run_name_override = config.get('run_name')
            if run_name_override:
                run_name = run_name_override
            else:
                run_name = generate_run_name(config, dataset_info, config.get('resume'), config_name, config.get('run_suffix'))
                run_name = f"{run_name}_dist{world_size}"
            log.info(f"Run name: {run_name}")
        else:
            run_name = None # Will be broadcasted
        
        # Synchronize run name across processes
        if world_size > 1:
            run_name_list = [run_name] if is_main() else [None]
            dist.broadcast_object_list(run_name_list, src=0)
            if not is_main():
                run_name = run_name_list[0]
        
        # Setup checkpoint directory
        checkpoint_config = config.get('checkpoint', {})
        base_checkpoint_dir = resolve_path(checkpoint_config.get('output_dir', 'outputs'))
        run_checkpoint_dir = base_checkpoint_dir / run_name
        
        if is_main():
            run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"Checkpoints will be saved to: {run_checkpoint_dir}")
        
        if 'checkpoint' not in cfg: # For original DictConfig
            cfg.checkpoint = OmegaConf.create()
        cfg.checkpoint.output_dir = str(run_checkpoint_dir)
        
        # Update the config dict to include the run-specific checkpoint directory
        if 'checkpoint' not in config:
            config['checkpoint'] = {}
        config['checkpoint']['output_dir'] = str(run_checkpoint_dir)

    # --- Timer for Model Setup ---
    with Timer("Model Setup (Init, DDP)", log, main_process=is_main()):
        # Extract trainable components configuration
        trainable_components_config = config['trainable_components']
        decoder_train_cfg = trainable_components_config['decoder']
        encoder_train_cfg = trainable_components_config['encoder']
        
        # Initialize models using the same pattern as the regular training script
        decoder_config_obj = DecoderConfig(
            model_name=model_name,
            **decoder_train_cfg
        )
        decoder = Decoder(decoder_config_obj)
        
        encoder_config_obj = EncoderConfig(
            model_name=model_name,
            **encoder_train_cfg
        )
        encoder = Encoder(encoder_config_obj)
        
        orig_model = OrigWrapper(model_name, load_in_8bit=False)
        
        # Initialize Decoder prompt
        decoder_base = decoder.module if hasattr(decoder, 'module') else decoder
        if 'decoder_prompt' in config and config['decoder_prompt']:
            if is_main():
                log.info(f"Setting decoder prompt: \"{config['decoder_prompt']}\"")
            decoder_base.set_prompt(config['decoder_prompt'], tokenizer)
        elif is_main():
            log.warning("Decoder prompt ('decoder_prompt') not found in config or is empty. Decoder soft prompts will not be initialized from text.")
        
        # Decoder generation testing will happen after models are moved to device

        # Initialize Encoder soft prompt
        # Check if soft_prompt_init_text is configured for the encoder
        if encoder_config_obj.soft_prompt_init_text:
            if is_main():
                log.info(f"Setting encoder soft prompt from text: \"{encoder_config_obj.soft_prompt_init_text}\"")
            encoder.set_soft_prompt_from_text(encoder_config_obj.soft_prompt_init_text, tokenizer)
        elif encoder_config_obj.soft_prompt_length > 0:
            if is_main():
                log.info(f"Encoder using randomly initialized soft prompt of length {encoder_config_obj.soft_prompt_length}.")
        elif is_main(): # No text and length is 0 (default)
            log.warning("Encoder soft prompt not configured (neither 'soft_prompt_init_text' nor 'soft_prompt_length > 0'). Encoder soft prompts will be empty.")
        
        decoder, encoder, orig_model = setup_distributed_models(
            decoder, encoder, orig_model, device, rank, world_size
        )
        
        if is_main():
            decoder_base_timer = decoder.module if hasattr(decoder, 'module') else decoder
            encoder_base_timer = encoder.module if hasattr(encoder, 'module') else encoder
            num_params_decoder = sum(p.numel() for p in decoder_base_timer.parameters())
            num_params_encoder = sum(p.numel() for p in encoder_base_timer.parameters())
            log.info(f"Decoder parameters: {num_params_decoder:,}")
            log.info(f"Encoder parameters: {num_params_encoder:,}")
        
        # Test decoder generation now that models are on the correct device
        if is_main():
            original_prompt = config.get('decoder_prompt', '')
            test_decoder_generation(decoder, encoder, tokenizer, device, log, is_main(), original_prompt)
            

    # --- Timer for Dataset and DataLoader Setup ---
    with Timer("Dataset and DataLoader Setup", log, main_process=is_main()):
        train_loader, val_loader, train_ds, val_ds = prepare_dataset_and_loaders(
            config=config,
            activation_dir=activation_dir,
            effective_val_activation_dir=effective_val_activation_dir,
            max_train_samples_req=config.get('max_train_samples'),
            max_val_samples_req=config.get('max_val_samples'),
            log=log
        )
        
        # Determine num_workers based on CPU count and world size
        num_dataloader_workers = 0 # Default to 0
        if world_size > 0 and os.cpu_count() is not None:
            workers_per_gpu = os.cpu_count() // world_size
            num_dataloader_workers = max(1, 2)#workers_per_gpu // 4) # Heuristic: half of available CPUs per GPU, at least 1. Adjust as needed.
            log.warning(f"Setting num_workers for DataLoaders to: {num_dataloader_workers} (hardcoded right now - to be resolved) (os.cpu_count()={os.cpu_count()}, world_size={world_size})") #TODO remove hardcoding, but not sure if the //4 is correct. It might be crashing the servers.
            if is_main():
                log.info(f"Setting num_workers for DataLoaders to: {num_dataloader_workers} (os.cpu_count()={os.cpu_count()}, world_size={world_size})")
        elif is_main():
            log.warning(f"Could not determine optimal num_workers. Defaulting to 0. os.cpu_count()={os.cpu_count()}, world_size={world_size}")

        train_loader = get_dataloader_for_distributed(
            train_ds, batch_size=batch_size, world_size=world_size, rank=rank, shuffle=True,
            collate_fn=collate, num_workers=num_dataloader_workers, pin_memory=True,
            persistent_workers=True if num_dataloader_workers > 0 else False,
        )
        
        if val_ds is not None:
            val_loader = get_dataloader_for_distributed(
                val_ds, batch_size=batch_size, world_size=world_size, rank=rank, shuffle=False,
                collate_fn=collate, num_workers=num_dataloader_workers, pin_memory=True,
                persistent_workers=True if num_dataloader_workers > 0 else False,
            )
        else:
            val_loader = None
        
        steps_per_epoch = len(train_loader) if train_loader else 0
        # Determine max_steps, similar to 01_train.py
        max_steps = config['max_train_steps']  # Read from config first
        num_train_epochs = config.get('num_train_epochs', 0)
        num_epochs_total_approx = 0 # For logging

        if steps_per_epoch == 0:
            if is_main():
                log.warning("Train loader is empty or batch_size is too large for dataset; steps_per_epoch is 0.")
            # If loader is empty, max_steps from config (or 0 if not set for epoch training) will be used.
            # If max_steps is 0, training loop won't run.
            num_epochs_total_approx = 0 if max_steps == 0 else 1 # Basic approximation for logging
            if num_train_epochs > 0 and max_steps == 0: # Epoch training specified but loader empty
                 if is_main():
                    log.warning(f"Epoch-based training ({num_train_epochs} epochs) requested, but train loader is empty. Effective max_steps will be 0.")
                 max_steps = 0 # Ensure no training
        else: # steps_per_epoch > 0
            if num_train_epochs > 0 and max_steps == 0:
                # Epoch-based training: calculate max_steps from num_train_epochs
                max_steps = steps_per_epoch * num_train_epochs
                num_epochs_total_approx = num_train_epochs
                if is_main():
                    log.info(f"Epoch-based training: {num_train_epochs} epochs × {steps_per_epoch} steps/epoch = {max_steps} total steps")
            elif max_steps > 0:
                # Step-based training: calculate approximate epochs from max_steps
                num_epochs_total_approx = (max_steps - 1) // steps_per_epoch + 1
            else:
                # Neither epochs nor steps specified, and loader is not empty. This is an error.
                if is_main():
                    # This error should ideally stop all ranks.
                    # For now, log error and set max_steps to 0 to prevent training.
                    log.error("Config Error: If train_loader is not empty, either 'num_train_epochs' or 'max_train_steps' must be > 0.")
                max_steps = 0 # Prevent training loop

        config['max_train_steps'] = max_steps # Update config dict with calculated max_steps for consistency if other parts rely on it

        # Handle early termination if configured
        early_termination_config = config.get('early_termination', {})
        if early_termination_config.get('enabled', False) and early_termination_config.get('steps'):
            early_term_str = early_termination_config['steps']
            early_term_steps = _resolve_schedule_to_steps(early_term_str, steps_per_epoch, log, "early_termination.steps", gradient_accumulation_steps)
            
            if early_term_steps > 0 and early_term_steps < max_steps:
                if is_main():
                    log.info(f"Early termination enabled: will stop after {early_term_steps} steps (original max_steps: {max_steps})")
                max_steps = early_term_steps
                config['max_train_steps'] = max_steps  # Update config with early termination limit
                
                # Recalculate approximate epochs with early termination
                if steps_per_epoch > 0:
                    num_epochs_total_approx = (max_steps - 1) // steps_per_epoch + 1

        # Calculate the number of optimizer steps
        max_optimizer_steps = max_steps // gradient_accumulation_steps
        if max_steps % gradient_accumulation_steps != 0: # Account for any remaining steps
            max_optimizer_steps +=1
        if is_main():
            log.info(f"Total micro-steps (fwd/bwd passes): {max_steps}")
            log.info(f"Total optimizer steps (scheduler steps): {max_optimizer_steps}")

        # Parse flexible interval settings (log / wandb / val) now that steps_per_epoch is known
        # These intervals are based on micro-steps (main loop steps)
        log_interval = _resolve_schedule_to_steps(config['log_interval'], steps_per_epoch, log, "log_interval", gradient_accumulation_steps)
        wandb_log_interval = _resolve_schedule_to_steps(config['wandb_log_interval'], steps_per_epoch, log, "wandb_log_interval", gradient_accumulation_steps)
        val_interval_str = config.get('val_interval', "500s")
        val_interval = _resolve_schedule_to_steps(val_interval_str, steps_per_epoch, log, "val_interval", gradient_accumulation_steps)
        
        if is_main():
            log.info(f"Validation setup: val_loader={'exists' if val_loader else 'None'}, interval={val_interval} steps")

        # -------- Drift-logging configuration --------
        drift_cfg = config.get('parameter_drift', {})
        drift_enabled = drift_cfg.get('enabled', True)
        drift_log_interval_str = drift_cfg.get('interval', "1000s")
        drift_log_interval = _resolve_schedule_to_steps(
            drift_log_interval_str, steps_per_epoch, log, "parameter_drift.interval",
            gradient_accumulation_steps
        ) if drift_enabled else -1
        if drift_enabled and drift_log_interval <= 0:
            drift_log_interval = max(steps_per_epoch, 1000)
            log.warning(f"parameter_drift.interval <=0 – resetting to {drift_log_interval} steps")
        if is_main():
            log.info(
                f"Parameter-drift logging "
                f"{'enabled' if drift_enabled else 'disabled'}"
                + (f" every {drift_log_interval} micro-steps." if drift_enabled else "")
            )

        if log_interval <= 0:
            raise ValueError(f"log_interval must be positive, got {config['log_interval']}")
        if wandb_log_interval <= 0:
            raise ValueError(f"wandb_log_interval must be positive, got {config['wandb_log_interval']}")

    # --- Timer for Optimizer and Scheduler Setup ---
    with Timer("Optimizer and Scheduler Setup", log, main_process=is_main()):
        trainable_components_config = config.get('trainable_components')
        decoder_train_cfg = trainable_components_config.get('decoder')
        encoder_train_cfg = trainable_components_config.get('encoder')
        custom_lr_multipliers = config.get('custom_lr_multipliers')
        
        # Get base models for parameter groups
        decoder_base = decoder.module if hasattr(decoder, 'module') else decoder
        encoder_base = encoder.module if hasattr(encoder, 'module') else encoder
        
        # Extract learning rate multipliers from config
        projection_lr_multiplier = custom_lr_multipliers.get('projection_layers')
        embedding_lr_multiplier = custom_lr_multipliers.get('embedding_layers')
        prompt_lr_multiplier = custom_lr_multipliers.get('prompt_layers')
        base_model_lr_multiplier = custom_lr_multipliers.get('base_models')
        
        params = param_groups(
            [decoder_base, encoder_base], 
            learning_rate, 
            projection_lr_multiplier, 
            embedding_lr_multiplier, 
            prompt_lr_multiplier,
            base_model_lr_multiplier
        )
        optimizer = torch.optim.AdamW(params)
        
        # Learning rate scheduler
        lr_scheduler_cfg = config.get('lr_scheduler', {})
        # Initialize scheduler with max_optimizer_steps
        lr_scheduler = get_lr_scheduler(optimizer, lr_scheduler_cfg, max_optimizer_steps, grad_accum_steps=gradient_accumulation_steps) 
        
        # Get mixed precision configuration
        mixed_precision_config = config.get('mixed_precision', {'enabled': True, 'dtype': 'auto'})
        
        # Log mixed precision settings (only on main process)
        if is_main():
            if mixed_precision_config.get('enabled', True):
                dtype_str = mixed_precision_config.get('dtype', 'auto')
                if dtype_str == 'auto':
                    actual_dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float32'
                    log.info(f"Mixed precision enabled: auto mode will use {actual_dtype}")
                else:
                    log.info(f"Mixed precision enabled: using {dtype_str}")
            else:
                log.info("Mixed precision disabled")
        
        # Initialize gradient scaler for mixed precision
        # Only enable scaler for float16 or bfloat16 on CUDA
        scaler_enabled = (
            device.type == "cuda" and 
            mixed_precision_config.get('enabled', True) and
            mixed_precision_config.get('dtype', 'auto') != 'float32'
        )
        scaler = torch.amp.GradScaler('cuda') if scaler_enabled else None

    # Initialize CheckpointManager (after steps_per_epoch is known)
    # It uses the updated config dict with the run-specific checkpoint directory.
    checkpoint_manager = CheckpointManager(config, log, steps_per_epoch, gradient_accumulation_steps)
    
    # Initialize W&B (only on main process)
    if is_main():
        wandb_config = config.get('wandb', {})
        
        # Handle wandb resume
        wandb_run_id = config.get('wandb_resume_id')
        force_disable_wandb = False
        # Handle explicit None values (e.g., from command line wandb_resume_id=None)
        if wandb_run_id is not None and str(wandb_run_id).lower() == 'none':
            wandb_run_id = None
            force_disable_wandb = True
            log.info("Explicitly disabling WandB run resumption (wandb_resume_id=None)")
        wandb_resume_mode = None
        
        # Build wandb_init_kwargs like in the regular training script
        wandb_init_kwargs = {
            'project': wandb_config.get('project', 'consistency-lens'),
            'name': run_name,  # Use our generated run name
            'config': config,
            'mode': wandb_config.get('mode', 'online'),
            'tags': []  # Initialize tags list
        }
        
        # Add dataset and model tags
        if dataset_info.get('dataset'):
            wandb_init_kwargs['tags'].append(f"dataset-{dataset_info['dataset']}")
        if dataset_info.get('model_name'):
            model_tag = dataset_info['model_name'].replace('/', '-')
            wandb_init_kwargs['tags'].append(f"model-{model_tag}")
        if dataset_info.get('layer') is not None:
            wandb_init_kwargs['tags'].append(f"layer-{dataset_info['layer']}")
        
        # Add distributed training tag
        wandb_init_kwargs['tags'].append(f"distributed-{world_size}gpu")
        
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
        
        # Add resume parameters if we have a run ID
        if wandb_run_id and not force_disable_wandb:
            wandb_init_kwargs['id'] = wandb_run_id
            wandb_init_kwargs['resume'] = wandb_resume_mode or "allow"
            log.info(f"Resuming WandB run: {wandb_run_id}")
        
        current_wandb_run_id = log_init(**wandb_init_kwargs)
        
        # Save config
        config_save_path = run_checkpoint_dir / "config.yaml"
        with open(config_save_path, "w") as f:
            OmegaConf.save(cfg, f)
        log.info(f"Config saved to: {config_save_path}")
    else:
        current_wandb_run_id = None
    
    # Resume from checkpoint if specified
    start_step = 0
    if config.get('resume'):
        checkpoint_path_str = config['resume']
        if Path(checkpoint_path_str).exists():
            if is_main():
                log.info(f"Resuming from checkpoint: {checkpoint_path_str}")
            
            models_to_load = {"decoder": decoder_base, "encoder": encoder_base}
            
            # Check if we should reset the step counter
            reset_steps = config.get('resume_reset_steps', False)
            
            try:
                rec = checkpoint_manager.load_checkpoint(
                    checkpoint_path_str,
                    models=models_to_load,
                    optimizer=optimizer if not reset_steps else None,  # Don't load optimizer state if resetting
                    map_location=device
                )
            except Exception as e:
                error_msg = f"Failed to load checkpoint from {checkpoint_path_str}: {str(e)}"
                if is_main():
                    log.error(error_msg)
                if world_size > 1:
                    dist.barrier()  # Ensure all processes reach this point
                raise RuntimeError(error_msg) from e
            
            if reset_steps:
                start_step = 0
                if is_main():
                    log.info("Resetting training steps to 0 (keeping model weights only)")
            else:
                start_step = int(rec.get("step", -1)) + 1
            
            # Load scheduler state if available (unless we're resetting steps)
            if not reset_steps:
                if lr_scheduler and "scheduler" in rec:
                    lr_scheduler.load_state_dict(rec["scheduler"])
                    if is_main():
                        log.info("Scheduler state successfully loaded from checkpoint.")
                elif lr_scheduler and is_main():
                    log.warning("Scheduler state not found in checkpoint.")
            else:
                if is_main():
                    log.info("Skipping scheduler state load due to step reset")
            
            if is_main():
                log.info(f"Resumed from micro-step {start_step}")
            
            # Override learning rate if it's different from checkpoint OR if we're resetting steps
            # This ensures command-line learning rate overrides take effect
            new_base_lr = learning_rate
            old_base_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else new_base_lr
            
            # Force LR reset if we're resetting steps
            if reset_steps or abs(new_base_lr - old_base_lr) > 1e-9:  # Check if LR changed or steps reset
                if is_main():
                    if reset_steps:
                        log.info(f"Resetting optimizer and scheduler (step reset). New LR: {new_base_lr:.2e}")
                    else:
                        log.info(f"Overriding learning rate from checkpoint: {old_base_lr:.2e} -> {new_base_lr:.2e}")
                
                # First, we need to recreate the optimizer with new base learning rates
                # This is necessary because schedulers cache the base_lrs at initialization
                
                # Get current optimizer state (momentum buffers, etc.)
                old_state_dict = optimizer.state_dict()
                
                # Recreate optimizer with new learning rate
                params = param_groups(
                    [decoder_base, encoder_base], 
                    new_base_lr,  # Use new base learning rate
                    projection_lr_multiplier, 
                    embedding_lr_multiplier, 
                    prompt_lr_multiplier,
                    base_model_lr_multiplier
                )
                optimizer = torch.optim.AdamW(params)
                
                # Restore optimizer state (momentum, etc.) but not the learning rates
                old_state = old_state_dict['state']
                if old_state:
                    optimizer.load_state_dict({'state': old_state, 'param_groups': optimizer.state_dict()['param_groups']})
                
                if is_main():
                    log.info("Recreated optimizer with new base learning rate")
                    log.info("Optimizer param groups after recreation:")
                    for i, group in enumerate(optimizer.param_groups[:5]):
                        log.info(f"  Group {i}: lr={group['lr']:.2e}")
                
                # Now recreate the scheduler with the new optimizer
                if lr_scheduler is not None:
                    # Calculate the current optimizer step based on micro-steps
                    current_optimizer_step = start_step // gradient_accumulation_steps
                    
                    # For PyTorch schedulers, last_epoch actually means the last step count
                    # We need to set it to current_optimizer_step - 1 since it will be incremented on first step()
                    # If we're resetting steps, always start from -1
                    if reset_steps:
                        scheduler_last_epoch = -1
                        current_optimizer_step = 0
                    else:
                        scheduler_last_epoch = current_optimizer_step - 1 if current_optimizer_step > 0 else -1
                    
                    # Recreate the scheduler with the correct last_epoch
                    lr_scheduler = get_lr_scheduler(optimizer, lr_scheduler_cfg, max_optimizer_steps, 
                                                    last_epoch=scheduler_last_epoch,
                                                    grad_accum_steps=gradient_accumulation_steps)
                    
                    if is_main():
                        current_lr_from_scheduler = lr_scheduler.get_last_lr()[0]
                        log.info(f"Learning rate scheduler reinitialized at optimizer step {current_optimizer_step}")
                        log.info(f"Current LR from scheduler.get_last_lr(): {current_lr_from_scheduler:.6f}")
                        
                        # Check what the scheduler thinks the base LRs are
                        if hasattr(lr_scheduler, 'base_lrs'):
                            log.info(f"Scheduler base_lrs: {[f'{lr:.6f}' for lr in lr_scheduler.base_lrs[:3]]}")
                        
                        # Verify it's working correctly
                        warmup_steps = parse_schedule_to_steps(lr_scheduler_cfg.get('warmup_steps', 0), steps_per_epoch, gradient_accumulation_steps)
                        if current_optimizer_step >= warmup_steps:
                            progress = (current_optimizer_step - warmup_steps) / max(1, max_optimizer_steps - warmup_steps)
                            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
                            expected_lr = new_base_lr * cosine_factor
                            log.info(f"Expected LR: base={new_base_lr:.6f} * cosine_factor={cosine_factor:.4f} = {expected_lr:.6f}")
                            log.info(f"Actual LR: {current_lr_from_scheduler:.6f} (should match expected)")
        else:
            error_msg = f"Resume checkpoint path not found: {checkpoint_path_str}"
            if is_main():
                log.error(error_msg)
            if world_size > 1:
                dist.barrier()  # Ensure all processes reach this point
            raise FileNotFoundError(error_msg)
    
    # Capture initial model states for drift calculation after model setup and potential checkpoint loading
    decoder_base_for_drift = decoder.module if hasattr(decoder, 'module') else decoder
    encoder_base_for_drift = encoder.module if hasattr(encoder, 'module') else encoder

    initial_decoder_state = None
    initial_encoder_state = None
    if drift_enabled and is_main():
        initial_decoder_state = get_initial_model_state(decoder_base_for_drift)
        initial_encoder_state = get_initial_model_state(encoder_base_for_drift)
        log.info(
            "Captured initial model states for drift calculation "
            f"(resume step={start_step})."
        )

    # Training loop
    if is_main():
        log.info("Starting training...")
        # CheckpointManager is already initialized
    
    # Initialize metrics
    running_losses = deque(maxlen=100)
    step_times = deque(maxlen=100)
    
    # Store last computed gradient/update norms for consistent logging
    last_grad_norm = 0.0
    last_update_norm = 0.0
    last_update_ratio = 0.0
    
    # Zero gradients at start
    optimizer.zero_grad(set_to_none=True)
    iter_loader = iter(train_loader)

    decoder.train()
    encoder.train()
    orig_model.model.eval() # leave in validation mode?

    # Main training loop
    for step in range(start_step, max_steps):
        step_start_time = time.time()
        
        current_epoch = step // steps_per_epoch if steps_per_epoch > 0 else 0
        
        ## Set epoch for DistributedSampler
        #if world_size > 1 and hasattr(train_loader.sampler, 'set_epoch'):
        #    epoch = step // len(train_loader)
        #    train_loader.sampler.set_epoch(epoch)
        
        # Training step with optimized gradient accumulation
        # Advance iterator – restart when exhausted
        try:
            batch = next(iter_loader)
        except StopIteration:
            if world_size > 1 and hasattr(train_loader.sampler, "set_epoch"):
                current_epoch += 1 # Ensure current_epoch is advanced if sampler epoch is set
                train_loader.sampler.set_epoch(current_epoch)
            else:
                current_epoch += 1 # Ensure current_epoch is advanced if sampler epoch is set
                if not hasattr(train_loader.sampler, "set_epoch"):
                    log.warning("train_loader.sampler does not have set_epoch method. Distributed samplers should have this.")
            iter_loader = iter(train_loader)
            batch = next(iter_loader)

        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        # ...

    
        #batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)} # Move batch to device
        if step == 0 and is_main():
            do_all_initial_validation(batch, orig_model, tokenizer, device, log, activation_dir)

        metrics = distributed_train_step(
            decoder,
            encoder,
            orig_model,
            batch,
            optimizer,
            scaler,
            config,
            step,
            device,
            tokenizer,
            cached_prefix_ids,
            gradient_accumulation_steps=gradient_accumulation_steps,
            is_distributed=(world_size > 1),
            lr_scheduler=lr_scheduler,
            steps_per_epoch=steps_per_epoch
        )
        
        # Synchronize metrics across processes
        metrics = sync_metrics(metrics, world_size)
        
        # Update running metrics
        running_losses.append(metrics['loss'])
        step_times.append(time.time() - step_start_time)
        
        # Update last computed gradient/update norms if they were calculated this step
        if metrics['grad_norm'] > 0:
            last_grad_norm = metrics['grad_norm']
            last_update_norm = metrics['update_norm']
            last_update_ratio = metrics['update_ratio']
        
        # Average metrics for logging
        avg_loss = sum(running_losses) / len(running_losses)
        avg_step_time = sum(step_times) / len(step_times) # Average time per micro-step
        
        # Calculate samples_per_sec: the rate of samples processed by the model.
        # effective_batch_size = (per_device_batch_size * num_devices * grad_accum_steps)
        # True sample throughput is (per_device_batch_size * num_devices) / avg_step_time.
        # This is equivalent to effective_batch_size / (avg_step_time * grad_accum_steps).
        if avg_step_time > 0:
            # gradient_accumulation_steps is guaranteed to be >= 1.
            samples_per_sec = effective_batch_size / (avg_step_time * gradient_accumulation_steps)
        else:
            samples_per_sec = 0.0

        current_lr = optimizer.param_groups[0]['lr']
        accumulation_step = (step % gradient_accumulation_steps) + 1
        
        # Performance metrics (calculated similarly to 01_train.py)
        # steps_per_second refers to micro-steps per second
        steps_per_second = 1.0 / avg_step_time if avg_step_time > 0 else 0
        # tokens_per_second is based on the corrected samples_per_sec
        tokens_per_second = samples_per_sec * config.get('t_text', 10) # t_text is tokens per sample

        # Console logging (only on main process)
        if is_main() and (step % log_interval == 0 or step == max_steps - 1):
            log.info(
                f"Step {step}/{max_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Samples/sec: {samples_per_sec:.1f} | " # Display corrected samples_per_sec
                f"Acc: {accumulation_step}/{gradient_accumulation_steps}"
            )

        # W&B logging (only on main process)
        if is_main() and (step % wandb_log_interval == 0 or step == max_steps - 1):
            # Get schedule values for logging (consistent with 01_train.py)
            # Current optimizer step for schedule value calculation
            current_optimizer_step_for_sched = step // gradient_accumulation_steps
            
            current_tau_log = get_schedule_value(config['gumbel_tau_schedule'], step, max_steps,
                                                         current_epoch=current_epoch, 
                                                         steps_per_epoch=steps_per_epoch, 
                                                         grad_accum_steps=gradient_accumulation_steps)
            current_alpha_log = get_schedule_value(config['alpha_schedule'], step, max_steps,
                                                           current_epoch=current_epoch,
                                                           steps_per_epoch=steps_per_epoch,
                                                           grad_accum_steps=gradient_accumulation_steps)
            
            wandb_metrics = {
                'train/loss': avg_loss, # This is the running average loss
                'loss/total': metrics['loss'], # This is the current step's synced loss
                'loss/mse': metrics['loss_mse'],
                'loss/lm': metrics['loss_lm'],
                'loss/kl': metrics['loss_kl'],
                'loss/entropy': metrics['loss_entropy'],
                'params/tau': current_tau_log,
                'params/alpha': current_alpha_log,
                'params/lm_w': config.get('lm_base_weight'),
                'params/kl_w': config.get('kl_base_weight', 1.0),
                'params/entropy_w': config.get('entropy_weight', 0.0),
                'optim/lr': current_lr,
                'gradient_accumulation/is_update_step': 1 if ((step + 1) % gradient_accumulation_steps == 0) or (step == max_steps - 1) else 0,
                'gradient_accumulation/accumulation_step': accumulation_step,
                'performance/steps_per_second': steps_per_second, # This is micro-steps per second
                'performance/samples_per_second': samples_per_sec, # Now: effective_batch_size / (avg_step_time * gradient_accumulation_steps)
                'performance/tokens_per_second': tokens_per_second,
                'performance/avg_step_time': avg_step_time, # This is avg micro-step time
                'train/gpu_memory_allocated': torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                'train/gradient_accumulation_step': accumulation_step,
                # Add explicit step counters for different x-axis options in WandB
                'progress/optimizer_step': current_optimizer_step_for_sched,
                'progress/micro_step': step,
                'progress/samples_seen': step * batch_size * world_size,  # Total samples processed across all GPUs
                'progress/tokens_seen': step * batch_size * world_size * config.get('t_text', 10),  # Total tokens processed
                'progress/optimizer_progress': current_optimizer_step_for_sched / max_optimizer_steps if max_optimizer_steps > 0 else 0,  # Fraction of training complete
                'progress/epoch_fraction': (step / steps_per_epoch) if steps_per_epoch > 0 else 0,  # Fractional epoch (e.g., 1.5 = halfway through 2nd epoch)
                'progress/epoch': current_epoch,  # Integer epoch number
            }
            
            # Always log gradient and update metrics using last computed values
            # These are only updated at accumulation boundaries but we want consistent logging
            wandb_metrics['grads/norm'] = last_grad_norm
            wandb_metrics['updates/norm'] = last_update_norm
            wandb_metrics['updates/ratio'] = last_update_ratio
            wandb_metrics['updates/lr_actual'] = lr_scheduler.get_last_lr()[0]
            
            # Add epoch if applicable
            # Epoch calculation should be based on optimizer steps or effective data passes
            effective_steps_per_epoch = steps_per_epoch // gradient_accumulation_steps if steps_per_epoch > 0 and gradient_accumulation_steps > 0 else 0
            if effective_steps_per_epoch > 0:
                current_effective_epoch = current_optimizer_step_for_sched // effective_steps_per_epoch
                wandb_metrics["epoch"] = current_effective_epoch + 1 # 1-based epoch for logging
            elif steps_per_epoch > 0 : # Fallback if effective_steps_per_epoch is 0 but steps_per_epoch > 0 (e.g. GAS=1)
                 wandb_metrics["epoch"] = (step // steps_per_epoch) + 1

            # System metrics (less frequently, similar to 01_train.py)
            if step % (wandb_log_interval * 10) == 0: # Log every 10 wandb log intervals
                sys_metrics_dict = get_system_metrics(device) # Assuming get_system_metrics is available
                
                # Ensure sys_metrics are prefixed correctly for wandb
                wandb_sys_metrics = {f"system/{k.replace('_', '_')}": v for k,v in sys_metrics_dict.items()}
                wandb_metrics.update(wandb_sys_metrics)

            if current_wandb_run_id:
                log_metrics(wandb_metrics, step)
        
        # Log parameter drift
        if drift_enabled and is_main():
            log_now = (
                step == start_step
                or (step == max_steps - 1 and max_steps > 0)
                or (step > start_step and drift_log_interval > 0 and step % drift_log_interval == 0)
            )
            if log_now and initial_decoder_state and initial_encoder_state:
                log.info(f"Logging parameter drift at step {step} …")
                log_parameter_drift(decoder_base_for_drift, initial_decoder_state,
                                    "decoder", step, log_metrics, log, True)
                log_parameter_drift(encoder_base_for_drift, initial_encoder_state,
                                    "encoder", step, log_metrics, log, True)

        # Checkpointing (only on main process)
        if is_main() and checkpoint_manager.should_save_step(step):
            # Get current tau and alpha for checkpointing, similar to 01_train.py
            # Checkpointing metadata should reflect the state at the current *micro-step*
            # but schedules are evaluated based on optimizer steps.
            current_optimizer_step_for_ckpt = step // gradient_accumulation_steps
            current_tau_ckpt = get_schedule_value(config['gumbel_tau_schedule'], step, max_steps, 
                                           current_epoch=current_epoch, 
                                           steps_per_epoch=steps_per_epoch, 
                                           grad_accum_steps=gradient_accumulation_steps)
            current_alpha_ckpt = get_schedule_value(config['alpha_schedule'], step, max_steps,
                                             current_epoch=current_epoch, 
                                             steps_per_epoch=steps_per_epoch, 
                                             grad_accum_steps=gradient_accumulation_steps)
            # current_wandb_run_id is already defined
            # Epoch for checkpoint filename/metadata: based on micro-steps or optimizer steps?
            # 01_train.py uses current_epoch_num = (micro_step // steps_per_epoch) + 1. Let's stick to that for filename.
            current_epoch_num_for_ckpt_filename = (step // steps_per_epoch) + 1 if steps_per_epoch > 0 else 1

            saved_path = checkpoint_manager.save_checkpoint(
                step=step, # Save with micro-step
                epoch=current_epoch_num_for_ckpt_filename, # Use 1-based epoch based on micro-steps
                models={'decoder': decoder_base, 'encoder': encoder_base},
                optimizer=optimizer,
                scheduler=lr_scheduler, # Scheduler state is based on optimizer steps
                metrics=metrics, 
                config=config, 
                tau=current_tau_ckpt,
                alpha=current_alpha_ckpt,
                wandb_run_id=current_wandb_run_id,
                additional_name=""
            )
            if saved_path:
                log.info(f"Checkpoint saved: {saved_path}")
            else:
                log.info(f"Checkpoint not saved at step {step} (e.g., max_checkpoints reached or interval not met).")
        
        # Validation
        if val_loader and val_interval > 0 and (step % val_interval == 0):
            if is_main():
                log.info(f"Running validation at step {step}")
            with Timer("Validation", log, main_process=is_main(), log_wandb=True, wandb_step=step):
                validate_distributed(
                    decoder=decoder,
                    encoder=encoder,
                    orig_model=orig_model,
                    val_loader=val_loader,
                    config=config,
                    step=step,
                    device=device,
                    tokenizer=tokenizer,
                    cached_prefix_ids=cached_prefix_ids,
                    world_size=world_size,
                    log=log,
                    is_main_process=is_main(),
                    wandb_run_id=current_wandb_run_id,
                    steps_per_epoch=steps_per_epoch
                )

        # Verbose sample printing (adapting logic from 01_train.py)
        if is_main():
            verbose_config = config.get('verbose_samples', {})
            if verbose_config.get('enabled', False):
                should_print_verbose = False
                # Condition for epoch end (approximation for distributed setup)
                epoch_just_finished = (steps_per_epoch > 0 and (step + 1) % steps_per_epoch == 0)

                # Check interval based on flexible schedule string
                verbose_interval_str = verbose_config.get('interval', "1000s") # Default from 01_train.py
                try:
                    # Assuming train_module contains _resolve_schedule_to_steps
                    verbose_interval = _resolve_schedule_to_steps(
                        verbose_interval_str, steps_per_epoch, log, "verbose_interval", gradient_accumulation_steps
                    )
                    if step % verbose_interval == 0:
                        should_print_verbose = True
                except Exception as e:
                    log.warning(f"Failed to parse verbose_samples interval '{verbose_interval_str}': {e}. Using step-based fallback if configured.")
                
                # Check for legacy epoch/step printing conditions
                if verbose_config.get('print_every_epoch', False) and epoch_just_finished:
                    should_print_verbose = True
                
                print_every_n_steps_val = verbose_config.get('print_every_n_steps', 0)
                if print_every_n_steps_val > 0 and step > 0 and step % print_every_n_steps_val == 0:
                    should_print_verbose = True

                if should_print_verbose:
                    with Timer("Verbose Sample Generation", log, main_process=is_main(), log_wandb=True, wandb_step=step):
                        log.info(f"Generating verbose samples at step {step}, epoch {current_epoch}")
                        decoder.eval()
                        encoder.eval()
                        orig_model.model.eval() # Ensure orig model is also in eval

                        # Get schedule arguments for verbose samples - these use optimizer steps
                        current_optimizer_step_for_verbose = step // gradient_accumulation_steps
                        sch_args_verbose = {
                            "tau": get_schedule_value(config['gumbel_tau_schedule'], step, max_steps,
                                                     current_epoch=current_epoch, 
                                                     steps_per_epoch=steps_per_epoch, 
                                                     grad_accum_steps=gradient_accumulation_steps),
                            "T_text": config.get('t_text', 8),
                            "alpha": get_schedule_value(config['alpha_schedule'], step, max_steps,
                                                       current_epoch=current_epoch, 
                                                       steps_per_epoch=steps_per_epoch, 
                                                       grad_accum_steps=gradient_accumulation_steps),
                            "lm_base_weight": config.get('lm_base_weight'), 
                            "kl_base_weight": config.get('kl_base_weight', 1.0),
                            "entropy_weight": config.get('entropy_weight', 0.0),
                            "mse_weight": config.get('mse_weight', 0.0), # Added for completeness
                        }

                        # Ensure batch is on the correct device (it should be already, but good practice)
                        verbose_batch = {k: v.to(device) for k, v in batch.items()}

                        # Assuming train_module contains process_and_print_verbose_batch_samples
                        num_printed, captured_text = process_and_print_verbose_batch_samples(
                            batch=verbose_batch,
                            cfg=config, # Pass the resolved dictionary config directly
                            models={"dec": decoder_base, "enc": encoder_base}, # Pass base models
                            orig=orig_model,
                            tok=tokenizer,
                            sch_args=sch_args_verbose,
                            device=device,
                            num_samples=verbose_config.get('num_samples', 2),
                            top_n_analysis=verbose_config.get('top_n_predictions', 3),
                            printed_count_so_far=0, # Reset for each call or manage globally if needed
                            generate_continuation=verbose_config.get('generate_continuation', True),
                            continuation_tokens=verbose_config.get('continuation_tokens', 30),
                            return_structured_data=False, # As per 01_train.py default for this path
                            capture_output=True,
                            cached_prefix_ids=cached_prefix_ids,  # Pass the cached prefix for loss computation
                            resample_ablation=config.get('resample_ablation', True)
                        )
                        
                        # current_wandb_run_id is already defined
                        if captured_text and current_wandb_run_id:
                            verbose_samples_logger.log_verbose_samples(
                                captured_text,
                                step=step,
                                table_name="training_verbose_samples_dist", # Differentiate if needed
                                limit_rows=verbose_config.get('wandb_table_limit', False)
                            )

                        decoder.train()
                        encoder.train()
                        orig_model.model.eval() # Set orig model back to train if it has trainable parts (usually not)

    
    # Final checkpoint
    if is_main() and checkpoint_manager.save_at_end:
        current_epoch_for_ckpt = max_steps // steps_per_epoch if steps_per_epoch > 0 else 0
        current_epoch_num_for_ckpt = current_epoch_for_ckpt + 1 if max_steps > 0 else 1 # Ensure 1-based epoch, or 1 if no steps
        step_for_ckpt = max_steps - 1 if max_steps > 0 else -1 # Consistent with saving at step -1 if max_steps is 0

        # current_wandb_run_id is already defined
        
        # Determine parameters for final schedule value calculation
        if max_steps == 0:
            # If no training steps, get initial value of schedules (optimizer_step 0 of a hypothetical 1-optimizer_step schedule)
            calc_optimizer_step_for_final_sched = 0
            # calc_total_optimizer_steps_for_final_sched = 1 
        else:
            # Get schedule value at the last completed micro_step, translating to optimizer_step
            calc_optimizer_step_for_final_sched = (max_steps - 1) // gradient_accumulation_steps
            # calc_total_optimizer_steps_for_final_sched = max_optimizer_steps

        # Get final tau and alpha using calculated optimizer step parameters
        final_micro_step_for_sched = max_steps -1 if max_steps > 0 else 0

        # Calculate final epoch for schedule calculations
        final_epoch_for_sched = final_micro_step_for_sched // steps_per_epoch if steps_per_epoch > 0 else 0
        
        final_tau = get_schedule_value(
            config['gumbel_tau_schedule'], 
            final_micro_step_for_sched, # current micro_step
            max_steps,                   # total micro_steps
            current_epoch=final_epoch_for_sched, 
            steps_per_epoch=steps_per_epoch,
            grad_accum_steps=gradient_accumulation_steps
        )
        final_alpha = get_schedule_value(
            config['alpha_schedule'], 
            final_micro_step_for_sched,
            max_steps,
            current_epoch=final_epoch_for_sched,
            steps_per_epoch=steps_per_epoch,
            grad_accum_steps=gradient_accumulation_steps
        )

        # 'metrics' here would be from the last training step, or undefined if loop didn't run
        final_metrics_to_save = metrics if 'metrics' in locals() else {}


        final_checkpoint_path = checkpoint_manager.save_checkpoint(
            step=step_for_ckpt, 
            epoch=current_epoch_num_for_ckpt, # Use 1-based epoch
            models={'decoder': decoder_base, 'encoder': encoder_base},
            optimizer=optimizer,
            scheduler=lr_scheduler,
            metrics=final_metrics_to_save,
            config=config, # Pass the resolved dictionary config
            tau=final_tau,
            alpha=final_alpha,
            wandb_run_id=current_wandb_run_id,
            additional_name="final"
        )
        if final_checkpoint_path:
            log.info(f"Final checkpoint saved: {final_checkpoint_path}")
        else:
            log.info(f"Final checkpoint not saved (e.g. disabled).")
    
    if is_main():
        log.info("Training completed!")
    
    # Cleanup distributed training
    cleanup_distributed()


if __name__ == "__main__":
    main()