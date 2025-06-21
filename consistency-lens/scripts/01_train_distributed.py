#!/usr/bin/env python3
"""Distributed training script for Consistency Lens with multi-GPU support and proper gradient accumulation."""

import logging
import math
import os
import sys
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path

try:
    import GPUtil
except ImportError:
    GPUtil = None

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM

# Enable TF32 for better performance on Ampere GPUs (A100, H100)
torch.set_float32_matmul_precision('high')
from typing import TYPE_CHECKING, Optional

from torch.utils.data import DataLoader
from tqdm import tqdm

from lens.data.collate import collate
from lens.evaluation.wandb_logger import verbose_samples_logger
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.orig import OrigWrapper
from lens.models.tuned_lens import initialize_consistency_lens_projection, load_full_tuned_lens
from lens.training.distributed import (
    cleanup_distributed,
    init_distributed,
    is_main,
    reduce_dict,
    set_seed,
    setup_for_distributed,
)
from lens.training.fast_distributed_sampler import FastDistributedSampler
from lens.training.loop import train_step as original_train_step
from lens.training.optim import param_groups
from lens.training.schedules import (
    SmoothTransitionScheduler,
    get_autocast_context,
    get_lr_scheduler,
    get_schedule_value,
    parse_schedule_config,
    unfreeze_non_adapters,
)
from lens.training.test import (
    test_decoder_generation,
)
from lens.utils.checkpoint_manager import CheckpointManager
from lens.utils.logging import init as log_init
from lens.utils.logging import log as log_metrics
from lens.utils.logging import summary_log as summary_log_metrics
from lens.utils.param_utils import log_parameter_counts, log_parameter_drift

if TYPE_CHECKING:
    from tuned_lens import TunedLens

# Import all the utility functions from the original training script
sys.path.insert(0, str(Path(__file__).parent))
import importlib
import dotenv
dotenv.load_dotenv()

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


# Track if we've logged the first conversion
_first_conversion_logged = False

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
    rank,
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
    #is_accumulation_start = (step % gradient_accumulation_steps == 0)
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
    # Handle entropy schedule with backward compatibility
    entropy_schedule = config.get('entropy_schedule', None)
    if entropy_schedule:
        entropy_weight = get_schedule_value(entropy_schedule, step, config['max_train_steps'],
                                          current_epoch=step // steps_per_epoch if steps_per_epoch > 0 else 0,
                                          steps_per_epoch=steps_per_epoch,
                                          grad_accum_steps=gradient_accumulation_steps)
    else:
        # Fall back to static entropy_weight for backward compatibility
        entropy_weight = config.get('entropy_weight', 0.0)
    
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
        "entropy_weight": entropy_weight,
        "mse_weight": config.get('mse_weight', 0.0),
        "lm_base_weight": config.get('lm_base_weight'),
        "GRPO_weight": config['GRPO_weight'],
        "GRPO_beta": config['GRPO_beta'],
        "GRPO_entropy_weight": config['GRPO_entropy_weight'],
        "group_n": config['group_n'],
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
        
        # Handle batch data dtype conversion if requested
        convert_batch_dtype(batch, config, device)
        
        # Forward pass with autocast
        with autocast_context:
            if config.get('detect_anomaly', False):
                ctxt_anomaly = torch.autograd.detect_anomaly(check_nan=True)
            else:
                ctxt_anomaly = nullcontext()
            with ctxt_anomaly:
               
                losses = original_train_step(
                    batch=batch,
                    models=models,
                    _loss_fns=loss_fns,
                    lm_loss_natural_prefix=config.get('lm_loss_natural_prefix'),
                    tokenizer=tokenizer,
                    cached_prefix_ids=cached_prefix_ids,
                    resample_ablation=config.get('resample_ablation', True),
                    do_kl_computation=config.get('do_kl_computation'),
                    do_lm_computation=config.get('do_lm_computation')
                )

            # Scale loss by accumulation steps
            loss = losses['total'] / gradient_accumulation_steps
            
            # Debug: Check loss dtype before backward
            if step == 0 and rank == 0:
                print(f"DEBUG: loss dtype before backward: {loss.dtype}")
                print(f"DEBUG: autocast enabled: {autocast_context != nullcontext()}")
                # Check a model parameter dtype
                param = next(decoder_base.parameters())
                print(f"DEBUG: decoder param dtype: {param.dtype}")
        
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
        'loss_entropy': losses['entropy'].item(),
        'loss_GRPO': losses['loss_GRPO'].item(),
        'KL_GRPO': losses['KL_GRPO'].item(),
        'advantages_mean': losses['advantages_mean'].item(),
        'advantages_std': losses['advantages_std'].item(),
        'mean_reward': losses['mean_reward'].item(),
        'mean_reward_std': losses['mean_reward_std'].item(),
        'fraction_variance_explained': losses['fraction_variance_explained'].item(),
        'grad_norm': grad_norm.item() if grad_norm is not None else 0.0,
        'update_norm': update_norm,
        'param_norm': param_norm,
        'update_ratio': update_ratio,
    }
    
    return metrics


def setup_distributed_models(decoder, encoder, orig_model, device, rank, world_size, decoder_has_trainable_params=True, encoder_has_trainable_params=True, compile_models=True, log=None):
    """Wrap models with DistributedDataParallel.
    
    Args:
        decoder: Decoder model
        encoder: Encoder model
        orig_model: Original model wrapper
        device: Device to use
        rank: Current process rank
        world_size: Total number of processes
        decoder_has_trainable_params: Whether the decoder has trainable parameters
        encoder_has_trainable_params: Whether the encoder has trainable parameters
        
    Returns:
        Tuple of (decoder, encoder, orig_model) wrapped with DDP if needed
    """
    if world_size > 1:
        # Move models to device before wrapping with DDP
        decoder = decoder.to(device)
        encoder = encoder.to(device)
        orig_model = orig_model.to(device) # orig_model is always moved
        
        # Wrap with DDP only if the model has trainable parameters
        if decoder_has_trainable_params:
            decoder = DDP(
                decoder, 
                device_ids=[rank], 
                output_device=rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
        
        if encoder_has_trainable_params:
            encoder = DDP(
                encoder, 
                device_ids=[rank], 
                output_device=rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
        # Note: orig_model is not wrapped as it's typically frozen.
        
    else:
        log.info("Not using DDP, moving models to device")
        # Single GPU - just move to device
        decoder = decoder.to(device)
        encoder = encoder.to(device) 
        orig_model = orig_model.to(device)
        

    if compile_models:
        decoder = torch.compile(decoder)
        encoder = torch.compile(encoder)
        log.info("Compiled models (after wrapping with DDP), not compiling orig_model")
    else:
        log.info("Not compiling models.")
        
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

def generate_verbose_samples_from_batch(
    decoder_base,
    encoder_base, 
    orig_model,
    batch,
    config,
    step,
    current_epoch,
    max_steps,
    steps_per_epoch,
    gradient_accumulation_steps,
    tokenizer,
    cached_prefix_ids,
    device,
    current_wandb_run_id,
    log,
    data_source="validation",
    comparison_tuned_lens: Optional["TunedLens"] = None, # New argument
):
    """Generate verbose samples from a given batch."""
    verbose_config = config.get('verbose_samples', {})
    
    # # Get schedule arguments for verbose samples
    # sch_args_verbose = {
    #     "tau": get_schedule_value(config['gumbel_tau_schedule'], step, max_steps,
    #                              current_epoch=current_epoch, 
    #                              steps_per_epoch=steps_per_epoch, 
    #                              grad_accum_steps=gradient_accumulation_steps),
    #     "T_text": config.get('t_text', 8),
    #     "alpha": get_schedule_value(config['alpha_schedule'], step, max_steps,
    #                                current_epoch=current_epoch, 
    #                                steps_per_epoch=steps_per_epoch, 
    #                                grad_accum_steps=gradient_accumulation_steps),
    #     "lm_base_weight": config.get('lm_base_weight'), 
    #     "kl_base_weight": config.get('kl_base_weight', 1.0),
    #     "entropy_weight": config.get('entropy_weight', 0.0),
    #     "mse_weight": config.get('mse_weight', 0.0),
    #     "GRPO_weight": config.get('GRPO_weight', 0.0),
    #     "GRPO_beta": config.get('GRPO_beta', 0.0),
    #     "GRPO_entropy_weight": config.get('GRPO_entropy_weight', 0.0),
    # }
      # Loss function parameters
    sch_args_verbose = {
        "t_text": config['t_text'],
        "tau": get_schedule_value(config['gumbel_tau_schedule'], step, config['max_train_steps'], 
                                current_epoch=step // steps_per_epoch if steps_per_epoch > 0 else 0, 
                                steps_per_epoch=steps_per_epoch, 
                                grad_accum_steps=gradient_accumulation_steps),
        "alpha": get_schedule_value(config['alpha_schedule'], step, config['max_train_steps'],
                                  current_epoch=step // steps_per_epoch if steps_per_epoch > 0 else 0, 
                                  steps_per_epoch=steps_per_epoch, 
                                  grad_accum_steps=gradient_accumulation_steps),
        "kl_base_weight": config['kl_base_weight'],
        "entropy_weight": config['entropy_weight'],
        "mse_weight": config['mse_weight'],
        "lm_base_weight": config['lm_base_weight'],
        "GRPO_weight": config['GRPO_weight'],
        "GRPO_beta": config['GRPO_beta'],
        "GRPO_entropy_weight": config['GRPO_entropy_weight'],
        "group_n": config['group_n'],
    }

    # Set models to eval mode
    decoder_base.eval()
    encoder_base.eval()
    orig_model.model.eval()

    try:
        # Ensure we're in no_grad mode to prevent gradient accumulation
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                profile_memory=True,
                record_shapes=True
            ) as prof:
            with torch.no_grad():
                # Generate verbose samples
                num_printed, captured_text = process_and_print_verbose_batch_samples(
                    batch=batch,
                    cfg=config,
                    models={"dec": decoder_base, "enc": encoder_base},
                    orig=orig_model,
                    tok=tokenizer,
                    sch_args=sch_args_verbose,
                    device=device,
                    num_samples=verbose_config.get('num_samples', 2),
                    top_n_analysis=verbose_config.get('top_n_predictions', 3),
                    printed_count_so_far=0,
                    generate_continuation=verbose_config.get('generate_continuation', True),
                    continuation_tokens=verbose_config.get('continuation_tokens', 30),
                    return_structured_data=False,
                    capture_output=True,
                    cached_prefix_ids=cached_prefix_ids,
                    resample_ablation=config.get('resample_ablation', True),
                    comparison_tuned_lens=comparison_tuned_lens # Pass it down
                )
        
            # Log to wandb
            if captured_text and current_wandb_run_id:
                table_name = f"{data_source}_verbose_samples_dist"
                verbose_samples_logger.log_verbose_samples(
                    captured_text,
                    step=step,
                    table_name=table_name,
                    limit_rows=verbose_config.get('wandb_table_limit', False)
                )
        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
            
    finally:
        # Always restore training mode
        decoder_base.train()
        encoder_base.train()
        # orig_model.model stays in eval mode
        orig_model.model.eval()
        
        # Clear any potential internal model caches
        # Some models (like Llama, Gemma) might have internal KV caches
        if hasattr(orig_model.model, 'past_key_values'):
            log.error("Clearing orig_model.model.past_key_values")  
            orig_model.model.past_key_values = None
        
        # Clear decoder's base model cache if it exists
        if hasattr(decoder_base, 'base') and hasattr(decoder_base.base, 'past_key_values'):
            log.error("Clearing decoder_base.base.past_key_values")
            decoder_base.base.past_key_values = None
            
        # Clear encoder's base model cache if it exists  
        if hasattr(encoder_base, 'base') and hasattr(encoder_base.base, 'past_key_values'):
            log.error("Clearing encoder_base.base.past_key_values")
            encoder_base.base.past_key_values = None
        
        # Force cleanup of any GPU memory that might have been allocated
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def _check_and_generate_verbose_samples(
    decoder_base,
    encoder_base,
    orig_model,
    val_loader,
    config,
    step,
    current_epoch,
    max_steps,
    steps_per_epoch,
    gradient_accumulation_steps,
    val_interval,
    tokenizer,
    cached_prefix_ids,
    device,
    wandb_run_id,
    log,
    comparison_tuned_lens: Optional["TunedLens"] = None, # New argument
):
    """Check if verbose samples should be generated and generate them if needed."""
    verbose_config = config.get('verbose_samples', {})
    if not verbose_config.get('enabled', False):
        return
    
    # Check if we should generate verbose samples at this step
    should_generate_verbose = False
    
    # Check interval-based conditions
    verbose_interval_str = verbose_config.get('interval', "1000s")
    try:
        verbose_interval = _resolve_schedule_to_steps(
            verbose_interval_str, steps_per_epoch, log, "verbose_interval", gradient_accumulation_steps
        )
        # Only generate verbose samples every verbose interval.
        # if this step is the first val_interval step after a verbose_interval step.
        if (step % verbose_interval) < val_interval:
            should_generate_verbose = True
    except Exception as e:
        log.warning(f"Failed to parse verbose_samples interval '{verbose_interval_str}': {e}")
    
    # Check legacy conditions
    epoch_just_finished = (steps_per_epoch > 0 and (step + 1) % steps_per_epoch == 0)
    if verbose_config.get('print_every_epoch', False) and epoch_just_finished:
        should_generate_verbose = True
    
    print_every_n_steps_val = verbose_config.get('print_every_n_steps', 0)
    if print_every_n_steps_val > 0 and step > 0 and step % print_every_n_steps_val == 0:
        should_generate_verbose = True
    
    if should_generate_verbose:
        with Timer("Verbose Sample Generation", log, main_process=True, log_wandb=True, wandb_step=step):
            log.info(f"Generating verbose samples from validation data at step {step}")
            
            # Get a fresh validation batch for verbose samples
            try:
                temp_val_iter = iter(val_loader)
                val_batch = next(temp_val_iter)
                val_batch = {k: v.to(device) for k, v in val_batch.items() if isinstance(v, torch.Tensor)}
                
                generate_verbose_samples_from_batch(
                    decoder_base=decoder_base,
                    encoder_base=encoder_base,
                    orig_model=orig_model,
                    batch=val_batch,
                    config=config,
                    step=step,
                    current_epoch=current_epoch,
                    max_steps=max_steps,
                    steps_per_epoch=steps_per_epoch,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    tokenizer=tokenizer,
                    cached_prefix_ids=cached_prefix_ids,
                    device=device,
                    current_wandb_run_id=wandb_run_id,
                    log=log,
                    data_source="validation",
                    comparison_tuned_lens=comparison_tuned_lens # Pass it down
                )
            except Exception as e:
                log.error(f"Failed to generate verbose samples from validation data: {e}")
                raise


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
    steps_per_epoch=None,
    current_epoch=None,
    max_steps=None,
    gradient_accumulation_steps=None,
    val_interval=None,
    comparison_tuned_lens: Optional["TunedLens"] = None, # New argument
    should_print_val = False,
    shared_base_model = None,
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
    
    # # Loss function parameters (use same as training)
    # loss_fns = {
    #     "T_text": config.get('t_text', 8),
    #     "tau": get_schedule_value(config['gumbel_tau_schedule'], step, config['max_train_steps'], 
    #                             current_epoch=step // steps_per_epoch if steps_per_epoch and steps_per_epoch > 0 else 0, 
    #                             steps_per_epoch=steps_per_epoch, 
    #                             grad_accum_steps=config['gradient_accumulation_steps']),
    #     "alpha": get_schedule_value(config['alpha_schedule'], step, config['max_train_steps'],
    #                               current_epoch=step // steps_per_epoch if steps_per_epoch and steps_per_epoch > 0 else 0, 
    #                               steps_per_epoch=steps_per_epoch, 
    #                               grad_accum_steps=config['gradient_accumulation_steps']),
    #     "kl_base_weight": config.get('kl_base_weight', 1.0),
    #     "entropy_weight": config.get('entropy_weight', 0.0),
    #     "mse_weight": config.get('mse_weight', 0.0),
    #     "lm_base_weight": config.get('lm_base_weight'),
    #     "GRPO_weight": config['GRPO_weight'],
    #     "GRPO_beta": config['GRPO_beta'],
    #     "group_n": config['group_n'],
    # }
      # Loss function parameters
    # Handle entropy schedule with backward compatibility
    entropy_schedule = config.get('entropy_schedule', None)
    if entropy_schedule:
        entropy_weight = get_schedule_value(entropy_schedule, step, config['max_train_steps'],
                                          current_epoch=step // steps_per_epoch if steps_per_epoch > 0 else 0,
                                          steps_per_epoch=steps_per_epoch,
                                          grad_accum_steps=gradient_accumulation_steps)
    else:
        # Fall back to static entropy_weight for backward compatibility
        entropy_weight = config.get('entropy_weight', 0.0)
    
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
        "entropy_weight": entropy_weight,
        "mse_weight": config.get('mse_weight', 0.0),
        "lm_base_weight": config.get('lm_base_weight'),
        "GRPO_weight": config['GRPO_weight'],
        "GRPO_beta": config['GRPO_beta'],
        "GRPO_entropy_weight": config['GRPO_entropy_weight'],
        "group_n": config['group_n'],
    }
    
    # Metrics accumulators
    total_loss = 0.0
    total_mse = 0.0
    total_lm = 0.0
    total_kl = 0.0
    total_entropy = 0.0
    total_fraction_variance_explained = 0.0
    total_GRPO = 0.0
    total_KL_GRPO = 0.0
    total_advantages_mean = 0.0
    total_advantages_std = 0.0
    total_mean_reward = 0.0
    total_mean_reward_std = 0.0
    num_batches = 0
    
    # Activation statistics accumulators
    all_activations = []
    all_reconstructions = []
    
    # Intervention metrics accumulators
    intervention_metrics = {
        'mse_baseline': 0.0,
        'mse_decoder': 0.0,
        'mse_shuffle': 0.0,
        'mse_shuffle_all': 0.0,
        'kl_baseline': 0.0,
        'kl_decoder': 0.0,
        'kl_shuffle': 0.0,
        'kl_shuffle_all': 0.0,
        'fraction_variance_explained': 0.0,
    }
    intervention_batches = 0
    
    # Limit validation batches for efficiency
    max_val_batches = config.get('max_val_batches', 50)
    # Run interventions on a subset of validation batches
    max_intervention_batches = min(10, max_val_batches)  # Limit interventions to first 10 batches
    
    if config['GRPO_beta'] == 0 and config['lm_base_weight'] == 0 and shared_base_model is None:
        orig_model.to(device)
    # No gradients needed for validation
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_val_batches:
                break
                
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # No group_n in valdiation batch
            
            # Handle batch data dtype conversion if requested
            convert_batch_dtype(batch, config, device)
            
            # Forward pass using train_step (but with no_grad context)
            # Run with eval_mode=True for first few batches to get intervention metrics
            run_interventions = batch_idx < max_intervention_batches
            
            losses = original_train_step(
                batch=batch,
                models=models,
                _loss_fns=loss_fns,
                lm_loss_natural_prefix=config.get('lm_loss_natural_prefix'),
                tokenizer=tokenizer,
                cached_prefix_ids=cached_prefix_ids,
                resample_ablation=config.get('resample_ablation', True),
                eval_mode=run_interventions,  # Enable eval mode for interventions
                verbose_eval=False,  # We don't need verbose data for now
                do_kl_computation=True, 
                do_lm_computation=True,
                GRPO_validate_mode=True
            )
            
            # Accumulate losses
            total_loss += losses['total'].item()
            total_mse += losses['mse'].item()
            total_lm += losses['lm'].item()
            total_kl += losses['kl'].item()
            total_GRPO += losses['loss_GRPO'].item()
            total_KL_GRPO += losses['KL_GRPO'].item()
            total_advantages_mean += losses['advantages_mean'].item()
            total_advantages_std += losses['advantages_std'].item()
            total_fraction_variance_explained += losses['fraction_variance_explained'].item()
            total_entropy += losses['entropy'].item()
            num_batches += 1
            
            # Accumulate intervention metrics if available
            if run_interventions:
                for key in intervention_metrics.keys():
                    metric_key = f"intervention_{key}"
                    if metric_key in losses:
                        intervention_metrics[key] += losses[metric_key].item()
                intervention_batches += 1
            
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
            total_fraction_variance_explained,
            total_GRPO,
            total_KL_GRPO,
            total_advantages_mean,
            total_advantages_std,
            total_mean_reward,
            total_mean_reward_std,
            float(num_batches),
            # Add intervention metrics
            intervention_metrics['mse_baseline'],
            intervention_metrics['mse_decoder'],
            intervention_metrics['mse_shuffle'],
            intervention_metrics['mse_shuffle_all'],
            intervention_metrics['kl_baseline'],
            intervention_metrics['kl_decoder'],
            intervention_metrics['kl_shuffle'],
            intervention_metrics['kl_shuffle_all'],
            float(intervention_batches)
        ], device=device)
        
        # All-reduce
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        # Extract reduced values
        total_loss = metrics_tensor[0].item()
        total_mse = metrics_tensor[1].item()
        total_lm = metrics_tensor[2].item()
        total_kl = metrics_tensor[3].item()
        total_entropy = metrics_tensor[4].item()
        total_fraction_variance_explained = metrics_tensor[5].item()
        total_GRPO = metrics_tensor[6].item()
        total_KL_GRPO = metrics_tensor[7].item()
        total_advantages_mean = metrics_tensor[8].item()
        total_advantages_std = metrics_tensor[9].item()
        total_mean_reward = metrics_tensor[10].item()
        total_mean_reward_std = metrics_tensor[11].item()
        num_batches = int(metrics_tensor[12].item())
        # Extract intervention metrics
        intervention_metrics['mse_baseline'] = metrics_tensor[13].item()
        intervention_metrics['mse_decoder'] = metrics_tensor[14].item()
        intervention_metrics['mse_shuffle'] = metrics_tensor[15].item()
        intervention_metrics['mse_shuffle_all'] = metrics_tensor[16].item()
        intervention_metrics['kl_baseline'] = metrics_tensor[17].item()
        intervention_metrics['kl_decoder'] = metrics_tensor[18].item()
        intervention_metrics['kl_shuffle'] = metrics_tensor[19].item()
        intervention_metrics['kl_shuffle_all'] = metrics_tensor[20].item()
        intervention_batches = int(metrics_tensor[21].item())
    
    # Compute averages
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_mse = total_mse / num_batches if num_batches > 0 else 0.0
    avg_lm = total_lm / num_batches if num_batches > 0 else 0.0
    avg_kl = total_kl / num_batches if num_batches > 0 else 0.0
    avg_entropy = total_entropy / num_batches if num_batches > 0 else 0.0
    avg_fraction_variance_explained = total_fraction_variance_explained / num_batches if num_batches > 0 else 0.0
    avg_GRPO = total_GRPO / num_batches if num_batches > 0 else 0.0
    avg_KL_GRPO = total_KL_GRPO / num_batches if num_batches > 0 else 0.0
    avg_advantages_mean = total_advantages_mean / num_batches if num_batches > 0 else 0.0
    avg_advantages_std = total_advantages_std / num_batches if num_batches > 0 else 0.0
    avg_mean_reward = total_mean_reward / num_batches if num_batches > 0 else 0.0
    avg_mean_reward_std = total_mean_reward_std / num_batches if num_batches > 0 else 0.0
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
        
        # Compute average intervention metrics
        avg_intervention_metrics = {}
        if intervention_batches > 0:
            for key, value in intervention_metrics.items():
                avg_intervention_metrics[key] = value / intervention_batches
        if should_print_val: 
            log.info(f"\n{'='*60}")
            log.info(f"Validation Results at Step {step}")
            log.info(f"{'='*60}")
            log.info("Average Losses:")
            log.info(f"  Total Loss: {avg_loss:.4f}")
            log.info(f"  MSE Loss: {avg_mse:.4f}")
            log.info(f"  LM Loss: {avg_lm:.4f}")
            log.info(f"  KL Loss: {avg_kl:.4f}")
            log.info(f"  Entropy: {avg_entropy:.4f}")
            log.info(f"  Fraction Variance Explained: {avg_fraction_variance_explained:.4f}")
            log.info(f"  GRPO Loss: {avg_GRPO:.4f}")
            log.info(f"  KL GRPO: {avg_KL_GRPO:.4f}")
            log.info(f"  Advantages Mean: {avg_advantages_mean:.4f}")
            log.info(f"  Advantages Std: {avg_advantages_std:.4f}")
            log.info(f"  Mean Reward: {avg_mean_reward:.4f}")
            log.info(f"  Mean Reward Std: {avg_mean_reward_std:.4f}")
            log.info("\nActivation Statistics:")
            log.info(f"  Original - Mean: {act_mean:.4f}, Std: {act_std:.4f}")
            log.info(f"  Original - Min: {act_min:.4f}, Max: {act_max:.4f}")
            log.info(f"  Reconstructed - Mean: {recon_mean:.4f}, Std: {recon_std:.4f}")
            log.info(f"  Reconstructed - Min: {recon_min:.4f}, Max: {recon_max:.4f}")
            log.info("\nBaseline Comparisons:")
            log.info(f"  Zero MSE (predicting zeros): {zero_mse:.4f}")
            log.info(f"  Mean MSE (predicting mean): {mean_mse:.4f}")
            log.info(f"  Our Reconstruction MSE: {reconstruction_mse:.4f}")
            log.info(f"  Improvement over zero baseline: {(zero_mse - reconstruction_mse) / zero_mse * 100:.1f}%")
            log.info(f"  Improvement over mean baseline: {(mean_mse - reconstruction_mse) / mean_mse * 100:.1f}%")
            log.info("\nAdditional Metrics:")
            log.info(f"  Correlation (original vs reconstructed): {correlation:.4f}")
            log.info(f"  Mean Relative Error: {relative_error:.4f}")
        
        # Log intervention metrics if available
        if intervention_batches > 0 and should_print_val:
            log.info(f"\nIntervention Analysis (on {intervention_batches} batches):")
            log.info(f"  MSE Baseline (original tokens): {avg_intervention_metrics['mse_baseline']:.4f}")
            log.info(f"  MSE Decoder (generated explanation): {avg_intervention_metrics['mse_decoder']:.4f}")
            log.info(f"  MSE Shuffle (first n-3 tokens): {avg_intervention_metrics['mse_shuffle']:.4f}")
            log.info(f"  MSE Shuffle (ALL tokens): {avg_intervention_metrics['mse_shuffle_all']:.4f}")
            log.info(f"  KL Baseline: {avg_intervention_metrics['kl_baseline']:.4f}")
            log.info(f"  KL Decoder: {avg_intervention_metrics['kl_decoder']:.4f}")
            log.info(f"  KL Shuffle (first n-3): {avg_intervention_metrics['kl_shuffle']:.4f}")
            log.info(f"  KL Shuffle (ALL tokens): {avg_intervention_metrics['kl_shuffle_all']:.4f}")
        
        log.info(f"{'='*60}\n")
        
        # Log to wandb
        if wandb_run_id:
            val_metrics = {
                'val/loss': avg_loss,
                'val/loss_mse': avg_mse,
                'val/loss_lm': avg_lm,
                'val/loss_kl': avg_kl,
                'val/loss_entropy': avg_entropy,
                'val/fraction_variance_explained': avg_fraction_variance_explained,
                'val/GRPO_loss': avg_GRPO,
                'val/KL_GRPO': avg_KL_GRPO,
                'val/advantages_mean': avg_advantages_mean, 
                'val/advantages_std': avg_advantages_std,
                'val/mean_reward': avg_mean_reward,
                'val/mean_reward_std': avg_mean_reward_std,
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
            
            # Add intervention metrics to wandb if available
            if intervention_batches > 0:
                for key, value in avg_intervention_metrics.items():
                    val_metrics[f'val/intervention_{key}'] = value
                val_metrics['val/intervention_batches'] = intervention_batches
            
            log_metrics(val_metrics, step)
    

    # Generate verbose samples if on main process and conditions are met
    if is_main_process and all(v is not None for v in [current_epoch, max_steps, gradient_accumulation_steps, val_interval]):
        # Move comparison_tuned_lens to device if it exists
        active_comparison_tuned_lens = None
        if comparison_tuned_lens:
            try:
                active_comparison_tuned_lens = comparison_tuned_lens.to(device)
            except Exception as e:
                log.error(f"Failed to move comparison_tuned_lens to device {device}: {e}")
                active_comparison_tuned_lens = None

        try:
            _check_and_generate_verbose_samples(
                decoder_base=decoder_base,
                encoder_base=encoder_base,
                orig_model=orig_model,
                val_loader=val_loader,
                config=config,
                step=step,
                current_epoch=current_epoch,
                max_steps=max_steps,
                steps_per_epoch=steps_per_epoch,
                gradient_accumulation_steps=gradient_accumulation_steps,
                val_interval=val_interval,
                tokenizer=tokenizer,
                cached_prefix_ids=cached_prefix_ids,
                device=device,
                wandb_run_id=wandb_run_id,
                log=log,
                comparison_tuned_lens=active_comparison_tuned_lens
            )
        finally:
            # Clean up GPU memory
            if active_comparison_tuned_lens is not None:
                del active_comparison_tuned_lens
                torch.cuda.empty_cache()  # Force GPU memory cleanup
                log.info("Cleared active_comparison_tuned_lens and emptied cache")
    
    # Put models back in train mode
    decoder_base.train()
    encoder_base.train()
    orig_model.model.eval()
    # orig_model.model stays in eval mode
    if shared_base_model is None:
        orig_model.to('cpu')
        log.info("Moved orig_model to cpu")
    
    # Return the average validation loss
    return avg_mse

# Load checkpoint BEFORE moving models to device if resuming
def maybe_resume_from_checkpoint(
    config, decoder, encoder, log, is_main, decoder_train_cfg, encoder_train_cfg, gradient_accumulation_steps
):
    start_step = 0
    checkpoint_data = None

    if not config.get('resume'):
        return start_step, checkpoint_data

    checkpoint_path_str = config['resume']
    checkpoint_path = Path(checkpoint_path_str)

    # If the resume path is a directory, find the latest checkpoint file within it.
    if checkpoint_path.is_dir():
        if is_main():
            log.info(f"Resume path is a directory. Searching for the latest checkpoint in {checkpoint_path}...")
        checkpoint_files = [f for f in checkpoint_path.glob('*.pt') if f.is_file()]
        if not checkpoint_files:
            if is_main():
                log.warning(f"No checkpoint files (.pt) found in directory: {checkpoint_path}. Cannot resume.")
            checkpoint_path_str = None
        else:
            latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            checkpoint_path_str = str(latest_checkpoint)
            if is_main():
                log.info(f"Found latest checkpoint to resume from: {checkpoint_path_str}")

    if checkpoint_path_str and Path(checkpoint_path_str).is_file():
        if is_main():
            log.info(f"Loading checkpoint before device setup: {checkpoint_path_str}")

        models_to_load = {"decoder": decoder, "encoder": encoder}
        reset_steps = config.get('resume_reset_steps', False)

        try:
            temp_ckpt_mgr = CheckpointManager({'checkpoint': {'enabled': True, 'strict_load': config['checkpoint'].get('strict_load', True)}}, log, None, gradient_accumulation_steps)

            if is_main():
                log.info(f"Loading checkpoint from exact path: {checkpoint_path_str}")
                log.info(f"File exists: {Path(checkpoint_path_str).exists()}")
                log.info(f"File size: {Path(checkpoint_path_str).stat().st_size / 1024 / 1024:.1f} MB")

                raw_ckpt = torch.load(checkpoint_path_str, map_location='cpu', weights_only=False)
                if 'models' in raw_ckpt and 'decoder' in raw_ckpt['models']:
                    decoder_state = raw_ckpt['models']['decoder']
                    if 'prompt_left_emb' in decoder_state:
                        emb = decoder_state['prompt_left_emb']
                        log.info(f"RAW checkpoint prompt_left_emb: shape={emb.shape}, norm={emb.norm().item():.6f}, mean={emb.mean().item():.6f}")
                    if 'prompt_right_emb' in decoder_state:
                        emb = decoder_state['prompt_right_emb']
                        log.info(f"RAW checkpoint prompt_right_emb: shape={emb.shape}, norm={emb.norm().item():.6f}, mean={emb.mean().item():.6f}")

            dec_left_raw_norm = decoder.prompt_left_emb.norm().item()
            dec_right_raw_norm = decoder.prompt_right_emb.norm().item()
            enc_raw_norm = encoder.soft_prompt_embeddings.norm().item()
            log.info(f"RAW checkpoint decoder left prompt norm: {dec_left_raw_norm:.6f}, decoder right prompt norm: {dec_right_raw_norm:.6f}, encoder soft prompt norm: {enc_raw_norm:.6f}")

            rec = temp_ckpt_mgr.load_checkpoint(
                checkpoint_path_str,
                models=models_to_load,
                optimizer=None,
                map_location="cpu"
            )

            if is_main():
                if hasattr(decoder, 'prompt_left_emb') and decoder.prompt_left_emb is not None:
                    log.info(f"After checkpoint load - Decoder left prompt norm: {decoder.prompt_left_emb.norm().item():.6f}")
                if hasattr(decoder, 'prompt_right_emb') and decoder.prompt_right_emb is not None:
                    log.info(f"After checkpoint load - Decoder right prompt norm: {decoder.prompt_right_emb.norm().item():.6f}")

            if reset_steps:
                start_step = 0
                if is_main():
                    log.info("Resetting training steps to 0 (keeping model weights only)")
            else:
                start_step = int(rec.get("step", -1)) + 1

            if is_main():
                log.info(f"Checkpoint loaded successfully. Will resume from step {start_step}")
                if 'tau' in rec:
                    log.info(f"Checkpoint tau: {rec['tau']}")
                if 'alpha' in rec:
                    log.info(f"Checkpoint alpha: {rec['alpha']}")
                if 'metrics' in rec and rec['metrics']:
                    log.info(f"Checkpoint metrics: {rec['metrics']}")

            checkpoint_data = rec

            if is_main():
                log.info("Verifying checkpoint compatibility...")

                checkpoint_models = checkpoint_data.get('models', {})
                checkpoint_decoder_state = checkpoint_models.get('decoder', {})

                ckpt_has_per_layer = 'proj_weight' in checkpoint_decoder_state and 'proj_bias' in checkpoint_decoder_state
                ckpt_has_single = 'proj.weight' in checkpoint_decoder_state and 'proj.bias' in checkpoint_decoder_state
                model_has_per_layer = decoder_train_cfg.get('per_layer_projections', False)

                if ckpt_has_per_layer and not model_has_per_layer:
                    error_msg = (
                        "Checkpoint was saved with per_layer_projections=True but config has per_layer_projections=False!\n"
                        "Add 'per_layer_projections: true' to trainable_components.decoder in your config."
                    )
                    log.error(error_msg)
                    raise RuntimeError(error_msg)
                elif ckpt_has_single and model_has_per_layer:
                    error_msg = (
                        "Checkpoint was saved with per_layer_projections=False but config has per_layer_projections=True!\n"
                        "Set 'per_layer_projections: false' in trainable_components.decoder in your config."
                    )
                    log.error(error_msg)
                    raise RuntimeError(error_msg)

                decoder_has_left = hasattr(decoder, 'prompt_left_emb') and decoder.prompt_left_emb is not None
                decoder_has_right = hasattr(decoder, 'prompt_right_emb') and decoder.prompt_right_emb is not None

                ckpt_has_left = 'prompt_left_emb' in checkpoint_decoder_state
                ckpt_has_right = 'prompt_right_emb' in checkpoint_decoder_state

                if ckpt_has_left or ckpt_has_right:
                    log.info("Checkpoint contains trained decoder prompt embeddings")
                    if decoder_has_left and ckpt_has_left:
                        loaded_norm = decoder.prompt_left_emb.norm().item()
                        ckpt_norm = checkpoint_decoder_state['prompt_left_emb'].norm().item()
                        log.info(f"Decoder prompt_left_emb - loaded norm: {loaded_norm:.4f}, Checkpoint norm: {ckpt_norm:.4f}")
                        if abs(loaded_norm - ckpt_norm) > 1e-6:
                            log.warning("WARNING: Decoder left prompt embeddings might not have been loaded correctly!")
                        if abs(loaded_norm - dec_left_raw_norm) > 1e-6:
                            log.warning("WARNING: Decoder prompt norm did not change after loading checkpoint!")
                    if decoder_has_right and ckpt_has_right:
                        loaded_norm = decoder.prompt_right_emb.norm().item()
                        ckpt_norm = checkpoint_decoder_state['prompt_right_emb'].norm().item()
                        log.info(f"Decoder prompt_right_emb - loaded norm: {loaded_norm:.4f}, Checkpoint norm: {ckpt_norm:.4f}")
                        if abs(loaded_norm - ckpt_norm) > 1e-6:
                            log.warning("WARNING: Decoder right prompt embeddings might not have been loaded correctly!")
                        if abs(loaded_norm - dec_right_raw_norm) > 1e-6:
                            log.warning("WARNING: Decoder prompt norm did not change after loading checkpoint!")

                if (ckpt_has_left and not decoder_has_left) or (ckpt_has_right and not decoder_has_right):
                    error_msg = (
                        "Checkpoint contains decoder prompt embeddings but model doesn't have them!\n"
                        f"Checkpoint has: left={ckpt_has_left}, right={ckpt_has_right}\n"
                        f"Model has: left={decoder_has_left}, right={decoder_has_right}\n"
                        "This likely means set_prompt() was not called before loading."
                    )
                    log.error(error_msg)
                    raise RuntimeError(error_msg)

                encoder_has_soft = hasattr(encoder, 'soft_prompt_embeddings')
                checkpoint_encoder_state = checkpoint_models.get('encoder', {})
                ckpt_has_soft = 'soft_prompt_embeddings' in checkpoint_encoder_state

                if ckpt_has_soft and not encoder_has_soft:
                    soft_prompt_shape = checkpoint_encoder_state['soft_prompt_embeddings'].shape
                    error_msg = (
                        f"Checkpoint contains encoder soft_prompt_embeddings (shape {soft_prompt_shape}) "
                        f"but model has soft_prompt_length={encoder_train_cfg.get('soft_prompt_length', 0)}!\n"
                        f"Update encoder config to match checkpoint: soft_prompt_length={soft_prompt_shape[0]}"
                    )
                    log.error(error_msg)
                    raise RuntimeError(error_msg)

                log.info("Checkpoint compatibility verification passed!")

                if ckpt_has_left and decoder_has_left:
                    rec['_expected_left_norm'] = checkpoint_decoder_state['prompt_left_emb'].norm().item()
                    rec['_expected_right_norm'] = checkpoint_decoder_state['prompt_right_emb'].norm().item() if ckpt_has_right else None

        except Exception as e:
            error_msg = f"Failed to load checkpoint from {checkpoint_path_str}: {str(e)}"
            if is_main():
                log.error(error_msg)
            raise RuntimeError(error_msg) from e
    else:
        error_msg = f"Resume checkpoint path not found: {checkpoint_path_str}"
        if is_main():
            log.error(error_msg)
        raise FileNotFoundError(error_msg)

    return start_step, checkpoint_data

def setup_wandb_and_save_config(
    config, 
    run_name, 
    dataset_info, 
    world_size, 
    run_checkpoint_dir, 
    log, 
    cfg
):
    wandb_config = config.get('wandb', {})
    wandb_run_id = config.get('wandb_resume_id')
    force_disable_wandb = False
    if wandb_run_id is not None and str(wandb_run_id).lower() == 'none':
        wandb_run_id = None
        force_disable_wandb = True
        log.info("Explicitly disabling WandB run resumption (wandb_resume_id=None)")
    wandb_resume_mode = None

    wandb_init_kwargs = {
        'project': wandb_config.get('project', 'consistency-lens'),
        'name': run_name,
        'config': config,
        'mode': wandb_config.get('mode', 'online'),
        'tags': []
    }

    if dataset_info.get('dataset'):
        wandb_init_kwargs['tags'].append(f"dataset-{dataset_info['dataset']}")
    if dataset_info.get('model_name'):
        model_tag = dataset_info['model_name'].replace('/', '-')
        wandb_init_kwargs['tags'].append(f"model-{model_tag}")
    if dataset_info.get('layer') is not None:
        wandb_init_kwargs['tags'].append(f"layer-{dataset_info['layer']}")
    wandb_init_kwargs['tags'].append(f"distributed-{world_size}gpu")

    command_line_args = ' '.join(sys.argv)
    wandb_init_kwargs['config']['command_line'] = command_line_args

    submit_script_command = os.environ.get('SUBMIT_SCRIPT_COMMAND', None)
    if submit_script_command:
        wandb_init_kwargs['config']['submit_script_command'] = submit_script_command
        log.info(f"Submit script command: {submit_script_command}")

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

    if wandb_run_id and not force_disable_wandb:
        wandb_init_kwargs['id'] = wandb_run_id
        wandb_init_kwargs['resume'] = wandb_resume_mode or "allow"
        log.info(f"Resuming WandB run: {wandb_run_id}")

    current_wandb_run_id = log_init(**wandb_init_kwargs)

    config_save_path = run_checkpoint_dir / "config.yaml"
    with open(config_save_path, "w") as f:
        OmegaConf.save(cfg, f)
    log.info(f"Config saved to: {config_save_path}")

    return current_wandb_run_id

def maybe_restore_optimizer_and_scheduler_from_checkpoint(
    checkpoint_data,
    config,
    optimizer,
    lr_scheduler,
    decoder,
    encoder,
    decoder_base,
    encoder_base,
    param_groups_fn, # Renamed from param_groups to avoid conflict with the variable
    projection_lr_multiplier,
    embedding_lr_multiplier,
    prompt_lr_multiplier,
    base_model_lr_multiplier,
    overall_encoder_lr_multiplier,
    weight_decay,
    learning_rate, # This is the new base LR from current config
    steps_per_epoch,
    gradient_accumulation_steps,
    max_optimizer_steps,
    lr_scheduler_cfg,
    start_step,
    log,
    is_main,
    SmoothTransitionScheduler,
    get_lr_scheduler,
    _resolve_schedule_to_steps,
):
    if checkpoint_data is None:
        return optimizer, lr_scheduler

    reset_steps = config.get('resume_reset_steps', False)
    strict_opt_load = config.get('strict_optimizer_load', True)

    old_base_lr_from_checkpoint_optim = None

    if not reset_steps and optimizer is not None and "optim" in checkpoint_data:
        if is_main():
            log.info("Attempting to load optimizer state from checkpoint and reconfigure with current settings.")
            param_to_name_map = {}
            current_decoder_model_for_params = decoder.module if hasattr(decoder, 'module') else decoder
            current_encoder_model_for_params = encoder.module if hasattr(encoder, 'module') else encoder

            for name, p in current_decoder_model_for_params.named_parameters():
                param_to_name_map[id(p)] = f"decoder.{name}"
            for name, p in current_encoder_model_for_params.named_parameters():
                param_to_name_map[id(p)] = f"encoder.{name}"
            
            log.info("Original optimizer structure (before potential rebuild from checkpoint):")
            for i, group in enumerate(optimizer.param_groups):
                group_param_names = [param_to_name_map.get(id(p), f"UNKNOWN_PARAM_ID_{id(p)}") for p in group['params']]
                log.info(
                    f"  Orig. group {i}: LR={group['lr']:.3e}, WD={group.get('weight_decay', 0.0):.3e}, "
                    f"#Pms={len(group['params'])} Param names: {group_param_names[:5]}{'...' if len(group_param_names) > 5 else ''}"
                )

        try:
            checkpoint_optimizer_dict = checkpoint_data["optim"]
            checkpoint_optim_internal_state = checkpoint_optimizer_dict.get("state")

            if is_main() and checkpoint_optimizer_dict.get("param_groups") and checkpoint_optimizer_dict["param_groups"]:
                old_base_lr_from_checkpoint_optim = checkpoint_optimizer_dict["param_groups"][0]['lr']
                log.info(f"Base LR from checkpoint's optimizer for potential smooth transition: {old_base_lr_from_checkpoint_optim:.2e}")
            elif is_main():
                log.warning("Could not determine base LR from checkpoint's optimizer param_groups (or param_groups was empty).")

            if is_main():
                log.info(f"Rebuilding optimizer parameter groups with current config LR: {learning_rate:.2e} and current multipliers/WD.")

            new_optimizer_param_groups = param_groups_fn(
                [decoder_base, encoder_base],
                learning_rate, 
                projection_lr_multiplier,
                embedding_lr_multiplier,
                prompt_lr_multiplier,
                base_model_lr_multiplier,
                overall_encoder_lr_multiplier,
                weight_decay,
            )
            
            optimizer = torch.optim.AdamW(new_optimizer_param_groups)

            if checkpoint_optim_internal_state:
                if strict_opt_load:
                    optimizer.load_state_dict({'state': checkpoint_optim_internal_state, 'param_groups': optimizer.state_dict()['param_groups']})
                    if is_main():
                        log.info("Optimizer internal state (e.g., momentum) strictly loaded from checkpoint into reconfigured optimizer.")
                else: # Non-strict, not recommended for user's preference but kept for completeness of logic
                    log.warning("Non-strict optimizer state load enabled. Merging state. This is less safe.")
                    current_optim_state_for_merge = optimizer.state_dict().get('state', {})
                    merged_state = {**current_optim_state_for_merge, **checkpoint_optim_internal_state} # Checkpoint takes precedence
                    optimizer.load_state_dict({'state': merged_state, 'param_groups': optimizer.state_dict()['param_groups']})
                    if is_main():
                        log.info("Optimizer internal state (e.g., momentum) non-strictly loaded/merged.")
            else: 
                if is_main():
                    log.warning("No optimizer internal state ('state') found in checkpoint. Optimizer uses fresh state with new config.")
            
            for group in optimizer.param_groups:
                group['initial_lr'] = group['lr']

            if is_main():
                log.info("Optimizer reconfigured with current settings. New structure:")
                for i, group in enumerate(optimizer.param_groups):
                    group_param_names_new = [param_to_name_map.get(id(p), f"UNKNOWN_PARAM_ID_{id(p)}") for p in group['params']]
                    log.info(
                        f"  New group {i}: LR={group['lr']:.3e}, Initial_LR={group['initial_lr']:.3e}, "
                        f"WD={group.get('weight_decay', 0.0):.3e}, #Pms={len(group['params'])} Param names: {group_param_names_new[:5]}{'...' if len(group_param_names_new) > 5 else ''}"
                    )

        except Exception as e: 
            if is_main():
                log.error(f"CRITICAL: Failed to load optimizer state or reconfigure optimizer: {e}. This is a hard error as per preference.")
            raise RuntimeError(f"Failed to load and reconfigure optimizer state from checkpoint. Error: {e}") from e


    # --- Learning Rate Scheduler Handling ---
    current_optimizer_step_for_scheduler = start_step // gradient_accumulation_steps
    scheduler_last_epoch = current_optimizer_step_for_scheduler - 1 if current_optimizer_step_for_scheduler > 0 else -1

    if not reset_steps:
        smooth_lr_config = config.get('smooth_lr_transition', {})
        smooth_lr_enabled = smooth_lr_config.get('enabled', False)
        
        use_smooth_transition = (
            smooth_lr_enabled and 
            old_base_lr_from_checkpoint_optim is not None and 
            abs(learning_rate - old_base_lr_from_checkpoint_optim) > 1e-9
        )

        if use_smooth_transition:
            transition_steps_raw = smooth_lr_config.get('transition_steps', '1000s')
            transition_steps = _resolve_schedule_to_steps(
                transition_steps_raw, steps_per_epoch, log, "smooth_lr_transition.transition_steps", gradient_accumulation_steps
            )
            if is_main():
                log.info(f"Enabling smooth LR transition from checkpoint LR {old_base_lr_from_checkpoint_optim:.2e} to current config LR {learning_rate:.2e} over {transition_steps} optimizer steps.")
            
            base_scheduler_for_smooth_transition = get_lr_scheduler(
                optimizer, lr_scheduler_cfg, max_optimizer_steps,
                last_epoch=-1, # Base scheduler for smooth transition starts fresh
                grad_accum_steps=gradient_accumulation_steps
            )
            
            lr_scheduler = SmoothTransitionScheduler(
                optimizer=optimizer,
                base_scheduler=base_scheduler_for_smooth_transition,
                transition_steps=transition_steps // gradient_accumulation_steps, 
                start_lr=old_base_lr_from_checkpoint_optim, 
                last_epoch=scheduler_last_epoch # Smooth scheduler is stepped to current progress
            )
            if is_main():
                log.info(f"Smooth transition scheduler created and initialized to optimizer step {current_optimizer_step_for_scheduler}.")
        
        else: # No smooth transition, or LRs are too similar for it. Recreate scheduler from current config.
            if is_main():
                log.info("No smooth LR transition. Recreating scheduler from current config, advanced to current step.")
                if smooth_lr_enabled and old_base_lr_from_checkpoint_optim is not None and not (abs(learning_rate - old_base_lr_from_checkpoint_optim) > 1e-9) :
                    log.info(f"Smooth transition was enabled, but checkpoint LR ({old_base_lr_from_checkpoint_optim:.2e}) and current LR ({learning_rate:.2e}) are too similar.")
                elif smooth_lr_enabled and old_base_lr_from_checkpoint_optim is None:
                    log.info("Smooth transition was enabled, but could not determine checkpoint LR. Recreating scheduler.")


            if lr_scheduler is not None: # If a scheduler type is configured
                lr_scheduler = get_lr_scheduler(optimizer, lr_scheduler_cfg, max_optimizer_steps,
                                                last_epoch=scheduler_last_epoch,
                                                grad_accum_steps=gradient_accumulation_steps)
                if is_main():
                    log.info(f"Scheduler recreated from current config and advanced to optimizer step {current_optimizer_step_for_scheduler}.")
                
                # OPTIONAL: Cautious loading of scheduler state if explicitly configured (and exists)
                # For now, per user preference, we are NOT loading scheduler state here by default to avoid issues.
                # If loading is desired, it would be added here with a config flag and error handling.
                # e.g., if config.get('resume_load_scheduler_state_if_not_transitioning', False) and "scheduler" in checkpoint_data:
                #    try:
                #        lr_scheduler.load_state_dict(checkpoint_data["scheduler"])
                #        log.info("Loaded scheduler state from checkpoint into the re-created scheduler.")
                #    except Exception as e:
                #        log.warning(f"Failed to load scheduler state into re-created scheduler: {e}. Using fresh state for this scheduler.")
            else: # lr_scheduler is None (e.g. constant LR, no scheduler configured)
                 if is_main():
                    log.info("No LR scheduler configured (`lr_scheduler` is None). Optimizer will use its fixed LRs.")
    
    else: # reset_steps is True
        if is_main():
            log.info("Optimizer and scheduler are being reset (reset_steps=True). Using fresh state with current config.")
        if lr_scheduler is not None:
            lr_scheduler = get_lr_scheduler(optimizer, lr_scheduler_cfg, max_optimizer_steps,
                                            last_epoch=-1, # Start from beginning
                                            grad_accum_steps=gradient_accumulation_steps)
            if is_main():
                log.info("LR scheduler reinitialized for step 0 due to reset_steps.")


    if is_main():
        log.info(f"Resumed from micro-step {start_step}. Optimizer and scheduler configured.")
        if lr_scheduler:
            try:
                current_lrs_sched = lr_scheduler.get_last_lr()
                log.info(f"Current scheduler LRs (first few): {[f'{lr:.2e}' for lr in current_lrs_sched[:5]]}")
            except Exception as e:
                log.warning(f"Could not get LRs from scheduler for logging: {e}")
        elif optimizer and optimizer.param_groups:
            log.info(f"Current optimizer first group LR (no scheduler): {optimizer.param_groups[0]['lr']:.2e}")
        else:
            log.info("No active LR scheduler and optimizer has no param groups to log LR from.")


    return optimizer, lr_scheduler



@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Distributed training entry point with optimized gradient accumulation."""

    
     # Take memory snapshots at different points
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
    if world_size == 1:
        torch.cuda.memory._record_memory_history()
        # Analyze using https://pytorch.org/memory_viz
    
    
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

    log.warning("L2 loss grows quadratically with the norm which increases with layer depth. Change coefficients accordingly.") 
    log.info("L2 loss grows quadratically with the norm which increases with layer depth. Change coefficients accordingly.") 
    print("L2 loss grows quadratically with the norm which increases with layer depth. Change coefficients accordingly.") 

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
        group_n = config.get("group_n", 1)
        batch_size = config['batch_size']
        gradient_accumulation_steps = config['gradient_accumulation_steps']
        # Adjust batch size for distributed training
        effective_batch_size = batch_size * gradient_accumulation_steps * world_size * group_n
        
        if is_main():
            log.info(f"Distributed training with {world_size} GPUs")
            log.info(f"Per-GPU batch size (without group_n): {batch_size}")
            log.info(f"Per-GPU batch size (with group_n): {batch_size*group_n}")
            log.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
            log.info(f"Effective batch size: {effective_batch_size}")
            log.info(f"Gradient sync: Every {gradient_accumulation_steps} steps")
            log.info(f"Group n: {group_n}")
        
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
                
                # Add SLURM job ID to run name if running under SLURM
                slurm_job_id = os.environ.get('SLURM_JOB_ID', None)
                if slurm_job_id:
                    run_name = f"{run_name}_slurm{slurm_job_id}"
                    
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
            summary_log_metrics({"checkpoint_dir": str(run_checkpoint_dir)})
        
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
        
        # Check if we should share the base model for memory efficiency
        share_base_model = (
            not decoder_train_cfg.get('base_model', False) and  # Decoder not training base
            not (encoder_train_cfg.get('base_model', True) and encoder_train_cfg.get('use_base_model', False))  # Encoder not training base
        )

        # Load base model once if sharing
        shared_base_model = None
        if share_base_model:
            if is_main():
                log.info(f"Loading shared base model '{model_name}' for memory efficiency (base models are frozen)")
            shared_base_model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=False)
            shared_base_model.eval()
            for p in shared_base_model.parameters():
                p.requires_grad = False

        # Initialize models with optional shared base
        decoder_config_obj = DecoderConfig(
            model_name=model_name,
            **decoder_train_cfg
        )
        decoder = Decoder(decoder_config_obj, base_to_use=shared_base_model)
        
        encoder_config_obj = EncoderConfig(
            model_name=model_name,
            **encoder_train_cfg
        )
        encoder = Encoder(encoder_config_obj, base_to_use=shared_base_model)
        
        orig_model = OrigWrapper(model_name, load_in_8bit=False, base_to_use=shared_base_model)
        
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
        if is_main():
            log.info(f"Before init: traces of each proj factor: {[p.trace().item() for p in decoder.proj_weight]}")
            log.info(f"Before init: sum of abs of biases: {[p.abs().sum().item() for p in decoder.proj_bias]}")
        if is_main(): # Perform initialization only on the main process
            log.info("Attempting tuned lens initialization for decoder")
            # Initialize Decoder's projection layer  
            initialize_consistency_lens_projection(
                    model_component=decoder,
                    component_config=config['trainable_components']['decoder'],  # Changed from config_dict
                    component_name="Decoder", 
                    main_run_config=config,  # Changed from config_dict
                    log=log,
                    is_main_process=True,
                    resolve_path_fn=resolve_path  # Added this parameter
                )
            # Initialize Encoder's projection layer (if applicable)
            initialize_consistency_lens_projection(
                    model_component=encoder,
                    component_config=config['trainable_components']['encoder'],  # Changed from config_dict
                    component_name="Encoder",
                    main_run_config=config,  # Changed from config_dict
                    log=log,
                    is_main_process=True,
                    resolve_path_fn=resolve_path  # Added this parameter
                )
        if is_main():
            log.info(f"After init: traces of each proj factor: {[p.trace().item() for p in decoder.proj_weight]}")
            log.info(f"After init: sum of abs of biases: {[p.abs().sum().item() for p in decoder.proj_bias]}")
        

        start_step, checkpoint_data = maybe_resume_from_checkpoint(
            config, decoder, encoder, log, is_main, decoder_train_cfg, encoder_train_cfg, gradient_accumulation_steps
        )

        # Determine if models have trainable parameters BEFORE DDP setup
        decoder_has_trainable_params = any(p.requires_grad for p in decoder.parameters())
        encoder_has_trainable_params = any(p.requires_grad for p in encoder.parameters())

        if is_main():
            log.info(f"Decoder has trainable parameters: {decoder_has_trainable_params}")
            log.info(f"Encoder has trainable parameters: {encoder_has_trainable_params}")

        
        log.info('At start time, param counts:')
        log_parameter_counts(decoder, encoder, orig_model, decoder_config_obj, encoder_config_obj, log)
        
        # NOW move models to device and set up DDP
        decoder, encoder, orig_model = setup_distributed_models(
            decoder, encoder, orig_model, device, rank, world_size,
            decoder_has_trainable_params=decoder_has_trainable_params,
            encoder_has_trainable_params=encoder_has_trainable_params,
            compile_models=config['compile_models'],
            log=log
        )
        
        if is_main():
            decoder_base_timer = decoder.module if hasattr(decoder, 'module') else decoder
            encoder_base_timer = encoder.module if hasattr(encoder, 'module') else encoder
            num_params_decoder = sum(p.numel() for p in decoder_base_timer.parameters())
            num_params_encoder = sum(p.numel() for p in encoder_base_timer.parameters())
            log.info(f"Decoder parameters: {num_params_decoder:,}")
            log.info(f"Encoder parameters: {num_params_encoder:,}")
            
            # Final verification of prompt embeddings after DDP setup
            if checkpoint_data and '_expected_left_norm' in checkpoint_data:
                decoder_base = decoder.module if hasattr(decoder, 'module') else decoder
                actual_left_norm = decoder_base.prompt_left_emb.norm().item()
                expected_left_norm = checkpoint_data['_expected_left_norm']
                
                if abs(actual_left_norm - expected_left_norm) > 1e-4:
                    log.error("CRITICAL: Decoder prompt embeddings lost after DDP setup!")
                    log.error(f"Expected left norm: {expected_left_norm:.6f}, Actual: {actual_left_norm:.6f}")
                    log.error("This is causing the KL loss jump on resume!")
                    raise RuntimeError("Decoder prompt embeddings were corrupted during model setup")
                else:
                    log.info(f"✓ Decoder prompt embeddings preserved after DDP (norm: {actual_left_norm:.6f})")
            
            if checkpoint_data and '_expected_right_norm' in checkpoint_data:
                decoder_base = decoder.module if hasattr(decoder, 'module') else decoder
                actual_right_norm = decoder_base.prompt_right_emb.norm().item()
                expected_right_norm = checkpoint_data['_expected_right_norm']
                
                if abs(actual_right_norm - expected_right_norm) > 1e-4:
                    log.error("CRITICAL: Decoder prompt embeddings lost after DDP setup!")
                    log.error(f"Expected right norm: {expected_right_norm:.6f}, Actual: {actual_right_norm:.6f}")
                    raise RuntimeError("Decoder prompt embeddings were corrupted during model setup")
        
        # Test decoder generation now that models are on the correct device
        # IMPORTANT: Skip this test when resuming from checkpoint as it calls set_prompt() 
        # which overwrites the loaded prompt embeddings with fresh initialization values
        if is_main() and not config['resume']:
            log.info("Running decoder generation tests (new training run)")
            original_prompt = config.get('decoder_prompt', '')
            test_decoder_generation(decoder, encoder, tokenizer, device, log, is_main(), original_prompt)
        elif is_main() and config['resume']:
            log.info("Skipping decoder generation tests when resuming from checkpoint (preserves loaded prompt embeddings)")
            

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
        val_interval = _resolve_schedule_to_steps(config['val_interval'], steps_per_epoch, log, "val_interval", gradient_accumulation_steps)
        
        if is_main():
            log.info(f"Validation setup: val_loader={'exists' if val_loader else 'None'}, interval={val_interval} steps")

        # -------- Drift-logging configuration --------
        drift_cfg = config.get('parameter_drift', {})
        drift_enabled = drift_cfg.get('enabled', True)
        drift_log_interval = _resolve_schedule_to_steps(
            drift_cfg['interval'], steps_per_epoch, log, "parameter_drift.interval",
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
        
        # Extract learning rate multipliers from config (moved up to be available for unfreeze_non_adapters)
        projection_lr_multiplier = custom_lr_multipliers.get('projection_layers')
        embedding_lr_multiplier = custom_lr_multipliers.get('embedding_layers')
        prompt_lr_multiplier = custom_lr_multipliers.get('prompt_layers')
        base_model_lr_multiplier = custom_lr_multipliers.get('base_models')
        overall_encoder_lr_multiplier = custom_lr_multipliers.get('overall_encoder')
        weight_decay = config.get('weight_decay')
        
        # Get base models for parameter groups
        decoder_base = decoder.module if hasattr(decoder, 'module') else decoder
        encoder_base = encoder.module if hasattr(encoder, 'module') else encoder
        
        # CRITICAL FIX: Restore requires_grad state based on freeze schedule when resuming
        # This must happen BEFORE creating the optimizer
        if checkpoint_data and start_step > 0:
            freeze_schedule = config.get('freeze_schedule', {})
            if freeze_schedule.get('enabled', False):
                unfreeze_at_parsed = freeze_schedule.get('unfreeze_at_parsed', {})
                unfreeze_step = unfreeze_at_parsed.get('value', 0)
                
                if is_main():
                    log.info(f"Restoring requires_grad state for resumed training at step {start_step}")
                    log.info(f"Freeze schedule: unfreeze_at={unfreeze_step}, current_step={start_step}")
                
                # Check if we should be unfrozen at this step
                if start_step >= unfreeze_step:
                    # We're past the unfreeze point - need to unfreeze base models
                    current_epoch = start_step // steps_per_epoch if steps_per_epoch > 0 else 0
                    
                    # Apply unfreezing using the same logic as during training
                    # Note: unfreeze_non_adapters returns (optimizer, trainable_params, newly_unfrozen_params) tuple
                    # but we don't want the optimizer it creates - we'll create our own
                    _, _, newly_unfrozen_params = unfreeze_non_adapters(
                        decoder_base, 
                        encoder_base, 
                        config, 
                        learning_rate,
                        projection_lr_multiplier,
                        embedding_lr_multiplier,
                        prompt_lr_multiplier,
                        base_model_lr_multiplier,
                        overall_encoder_lr_multiplier,
                        opt_state_dict=None,  # We'll handle optimizer state later
                        current_step=start_step,
                        current_epoch=current_epoch,
                        grad_accum_steps=gradient_accumulation_steps
                    )
                    
                    if is_main():
                        # Count and log the number of trainable parameters after unfreezing
                        num_trainable_dec = sum(p.numel() for p in decoder_base.parameters() if p.requires_grad)
                        num_trainable_enc = sum(p.numel() for p in encoder_base.parameters() if p.requires_grad)
                        log.info("After restoring freeze state:")
                        log.info(f"  Decoder trainable parameters: {num_trainable_dec:,}")
                        log.info(f"  Encoder trainable parameters: {num_trainable_enc:,}")
                        
                        # Log which parameter groups are trainable
                        dec_base_trainable = any(p.requires_grad for n, p in decoder_base.named_parameters() if n.startswith('base.'))
                        enc_base_trainable = any(p.requires_grad for n, p in encoder_base.named_parameters() if n.startswith('base.'))
                        log.info(f"  Decoder base model trainable: {dec_base_trainable}")
                        log.info(f"  Encoder base model trainable: {enc_base_trainable}")
                else:
                    if is_main():
                        log.info(f"Not yet at unfreeze point ({start_step} < {unfreeze_step}), keeping original frozen state")
        
        # Now create optimizer with correctly set requires_grad states
        params = param_groups(
            [decoder_base, encoder_base], 
            learning_rate, 
            projection_lr_multiplier, 
            embedding_lr_multiplier, 
            prompt_lr_multiplier,
            base_model_lr_multiplier,
            overall_encoder_lr_multiplier,
            weight_decay
        )
        optimizer = torch.optim.AdamW(params)
        
        # Learning rate scheduler
        # Initialize scheduler with max_optimizer_steps
        lr_scheduler = get_lr_scheduler(optimizer, config['lr_scheduler'], max_optimizer_steps, grad_accum_steps=gradient_accumulation_steps) 
        
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
        # IMPORTANT: Only enable scaler for float16, NOT for bfloat16
        # bfloat16 doesn't need gradient scaling
        dtype_str = mixed_precision_config.get('dtype', 'auto')
        if dtype_str == 'auto':
            actual_dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
        else:
            actual_dtype = dtype_str
            
        scaler_enabled = (
            device.type == "cuda" and 
            mixed_precision_config.get('enabled', True) and
            actual_dtype == 'float16'  # Only for float16, not bfloat16!
        )
        scaler = torch.amp.GradScaler('cuda') if scaler_enabled else None
        
        if is_main():
            if scaler_enabled:
                log.info("Gradient scaler enabled for float16 mixed precision")
            elif mixed_precision_config.get('enabled', True) and actual_dtype == 'bfloat16':
                log.info("Using bfloat16 mixed precision (no gradient scaler needed)")
            
            # Log force_data_conversion setting
            if config.get('force_data_conversion', False):
                log.info("Force data conversion enabled - batch data will be converted to match training dtype")
            else:
                log.info("Force data conversion disabled - batch data will use original dtype")

    # Initialize CheckpointManager (after steps_per_epoch is known)
    # It uses the updated config dict with the run-specific checkpoint directory.
    checkpoint_manager = CheckpointManager(config, log, steps_per_epoch, gradient_accumulation_steps)
    
    # --- Load Comparison TunedLens for Verbose Samples (Main Process Only for Loading) ---
    comparison_tuned_lens_cpu = None
    verbose_sample_config = config.get('verbose_samples', {})
    if verbose_sample_config.get('enabled', False) and verbose_sample_config.get('compare_with_tuned_lens', False):
        if is_main():
            log.info("Loading comparison TunedLens for verbose samples...")
            try:
                tl_model_name_for_load = config['model_name']
                tl_checkpoint = verbose_sample_config.get('tuned_lens_checkpoint_path_or_name', None)

                comparison_tuned_lens_cpu = load_full_tuned_lens(
                    model_or_model_name=tl_model_name_for_load,
                    checkpoint_path_or_name=tl_checkpoint,
                    device="cpu",
                    log=log,
                    is_main_process=is_main()
                )
                if comparison_tuned_lens_cpu:
                    log.info(f"Successfully loaded comparison TunedLens from '{tl_checkpoint if tl_checkpoint else 'default HF location for ' + tl_model_name_for_load}' to CPU.")
                else:
                    log.warning(f"Failed to load comparison TunedLens from '{tl_checkpoint if tl_checkpoint else 'default HF location for ' + tl_model_name_for_load}'. Comparison will be skipped.")
            except Exception as e:
                log.error(f"Error loading comparison TunedLens: {e}. Comparison will be skipped.", exc_info=True)
                comparison_tuned_lens_cpu = None
    
    if is_main():
        current_wandb_run_id = setup_wandb_and_save_config(
            config, run_name, dataset_info, world_size, run_checkpoint_dir, log, cfg
        )
    else:
        current_wandb_run_id = None

    optimizer, lr_scheduler = maybe_restore_optimizer_and_scheduler_from_checkpoint(
        checkpoint_data=checkpoint_data,
        config=config,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        decoder=decoder,
        encoder=encoder,
        decoder_base=decoder_base,
        encoder_base=encoder_base,
        param_groups_fn=param_groups,
        projection_lr_multiplier=projection_lr_multiplier,
        embedding_lr_multiplier=embedding_lr_multiplier,
        prompt_lr_multiplier=prompt_lr_multiplier,
        base_model_lr_multiplier=base_model_lr_multiplier,
        overall_encoder_lr_multiplier=overall_encoder_lr_multiplier,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        steps_per_epoch=steps_per_epoch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_optimizer_steps=max_optimizer_steps,
        lr_scheduler_cfg=config['lr_scheduler'],
        start_step=start_step,
        log=log,
        is_main=is_main,
        SmoothTransitionScheduler=SmoothTransitionScheduler,
        get_lr_scheduler=get_lr_scheduler,
        _resolve_schedule_to_steps=_resolve_schedule_to_steps,
    )
    
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
    
    # Handle resuming from checkpoint
    reset_steps = config.get('resume_reset_steps', False)
    skip_batches_on_resume = config.get('skip_batches_on_resume', False)  # Make it optional
    
    if start_step > 0 and not reset_steps:
        resume_epoch = start_step // steps_per_epoch if steps_per_epoch > 0 else 0
        
        # Always set the correct epoch for the sampler when resuming
        if world_size > 1 and hasattr(train_loader.sampler, 'set_epoch') and resume_epoch > 0:
            train_loader.sampler.set_epoch(resume_epoch)
            if is_main():
                log.info(f"Set DistributedSampler epoch to {resume_epoch} for resuming")
        
        # Optionally skip batches within the epoch
        if skip_batches_on_resume:
            batches_to_skip = start_step % steps_per_epoch if steps_per_epoch > 0 else start_step
            iter_loader = iter(train_loader)
            
            if batches_to_skip > 0:
                if is_main():
                    log.info(f"Skipping {batches_to_skip} batches to resume from step {start_step}")
                for _ in range(batches_to_skip):
                    try:
                        _ = next(iter_loader)
                    except StopIteration:
                        if is_main():
                            log.warning(f"Dataset ended while skipping batches. Expected to skip {batches_to_skip} batches.")
                        break
        else:
            # Just start from beginning of epoch
            iter_loader = iter(train_loader)
            if is_main():
                log.info(f"Starting from beginning of epoch {resume_epoch} (step {start_step})")
    else:
        # Starting fresh or reset_steps=True
        iter_loader = iter(train_loader)

    decoder.train()
    encoder.train()
    orig_model.model.eval() # leave in validation mode?
    max_consecutive_nan_losses = 5
    consecutive_nan_losses = 0
    val_loss = None

    # Main training loop
    # Create tqdm progress bar (only on main process)
    if is_main():
        pbar = tqdm(range(start_step, max_steps), 
                    desc="Training", 
                    initial=start_step, 
                    total=max_steps, miniters=log_interval)
    else:
        pbar = range(start_step, max_steps)

    if config['GRPO_beta'] == 0 and config['lm_base_weight'] == 0 and shared_base_model is None:
        orig_model.to('cpu')
        log.info("Moved orig_model to cpu")
    try: 
        for step in pbar:
            if step ==gradient_accumulation_steps and world_size==1:
                # Stop recording
                # Dump memory snapshot
                torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

                # Stop recording
                torch.cuda.memory._record_memory_history(enabled=False)
    
                exit()
                raise Exception("Stop here")
            step_start_time = time.time()

            current_epoch = step // steps_per_epoch if steps_per_epoch > 0 else 0

            # Training step with optimized gradient accumulation
            # Advance iterator – restart when exhausted
            try:
                raw_batch = next(iter_loader)
            except StopIteration:
                if world_size > 1 and hasattr(train_loader.sampler, "set_epoch"):
                    current_epoch += 1
                    train_loader.sampler.set_epoch(current_epoch)
                else:
                    current_epoch += 1
                    if not hasattr(train_loader.sampler, "set_epoch"):
                        log.warning("train_loader.sampler does not have set_epoch method. Distributed samplers should have this.")
                iter_loader = iter(train_loader)
                raw_batch = next(iter_loader)

            if step == 0:
                #log warning the first 10 tokens of the first element of the batch, on all gpus
                log.warning(f"First 10 tokens of the first element of the batch for rank {rank}: {raw_batch['input_ids_A'][:10]}")

            if group_n > 1:
                batch = {}
                for k, v in raw_batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.repeat_interleave(group_n, dim=0)
                    else:
                        batch[k] = v
            else:
                batch = raw_batch

            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            if step == 0 and is_main():
                if config['GRPO_beta'] == 0 and config['lm_base_weight'] == 0:
                    orig_model.to(device)
                do_all_initial_validation(batch, orig_model, tokenizer, device, log, activation_dir)
                if config['GRPO_beta'] == 0 and config['lm_base_weight'] == 0 and shared_base_model is None:
                    orig_model.to('cpu')
                    log.info("Moved orig_model to cpu")

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
                rank,
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
            # This is equivalent to effective_batch_size / (avg_step_time * gradient_accumulation_steps).
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
            tokens_per_second = samples_per_sec * config.get('t_text') # t_text is tokens per sample

            # Update progress bar description (only on main process and every log_interval)
            if is_main() and (step % log_interval == 0 or step == max_steps - 1):
                desc = (
                    f"Step {step}/{max_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Samples/sec: {samples_per_sec:.1f} | "
                    f"Acc: {accumulation_step}/{gradient_accumulation_steps}"
                )
                pbar.set_description(desc)
                # Also print to ensure it's captured in logs
                log.info(desc)

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
                
                # Get current entropy weight for logging
                entropy_schedule_log = config.get('entropy_schedule', None)
                if entropy_schedule_log:
                    current_entropy_log = get_schedule_value(entropy_schedule_log, step, max_steps,
                                                            current_epoch=current_epoch,
                                                            steps_per_epoch=steps_per_epoch,
                                                            grad_accum_steps=gradient_accumulation_steps)
                else:
                    current_entropy_log = config.get('entropy_weight', 0.0)

                wandb_metrics = {
                    'train/loss': avg_loss, # This is the running average loss
                    'loss/total': metrics['loss'], # This is the current step's synced loss
                    'loss/mse': metrics['loss_mse'],
                    'loss/lm': metrics['loss_lm'],
                    'loss/kl': metrics['loss_kl'],
                    'loss/entropy': metrics['loss_entropy'],
                    'loss/GRPO': metrics['loss_GRPO'],
                    'loss/KL_GRPO': metrics['KL_GRPO'],
                    'loss/advantages_mean': metrics['advantages_mean'],
                    'loss/advantages_std': metrics['advantages_std'],
                    'loss/mean_reward': metrics['mean_reward'],
                    'loss/mean_reward_std': metrics['mean_reward_std'],
                    'params/tau': current_tau_log,
                    'params/alpha': current_alpha_log,
                    'params/lm_w': config.get('lm_base_weight'),
                    'params/kl_w': config.get('kl_base_weight', 1.0),
                    'params/entropy_w': current_entropy_log,
                    'params/GRPO_entropy_w': config.get('GRPO_entropy_weight', 0.0),
                    'params/GRPO_w': config.get('GRPO_weight', 0.0),
                    'params/GRPO_beta': config.get('GRPO_beta', 0.0),
                    'optim/lr': lr_scheduler.get_last_lr()[0] if lr_scheduler else current_lr,
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


            # Validation and Verbose Samples
            val_loss = None
            if val_loader and val_interval > 0 and (step % val_interval == 0):
                should_print_val = step % (10*val_interval) == 0
                if is_main():
                    log.info(f"Running validation at step {step}")
                with Timer("Validation", log, main_process=is_main(), log_wandb=True, wandb_step=step):
                    val_loss = validate_distributed(
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
                        steps_per_epoch=steps_per_epoch,
                        current_epoch=current_epoch,
                        max_steps=max_steps,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        val_interval=val_interval,
                        comparison_tuned_lens=comparison_tuned_lens_cpu, # Pass CPU version
                        should_print_val = should_print_val,
                        shared_base_model=shared_base_model
                    )
                    most_recent_val_loss = val_loss
                if is_main() and (math.isnan(val_loss)):
                    consecutive_nan_losses += 1
                    if consecutive_nan_losses >= max_consecutive_nan_losses:
                        log.info(f"Stopping training at step {step} because of {consecutive_nan_losses} consecutive nan val_loss or loss.")
                        break
                else:
                    consecutive_nan_losses = 0

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
                if not math.isnan(most_recent_val_loss):
                    saved_path = checkpoint_manager.save_checkpoint(
                        step=step, # Save with micro-step
                        epoch=current_epoch_num_for_ckpt_filename, # Use 1-based epoch based on micro-steps
                        models={'decoder': decoder_base, 'encoder': encoder_base},
                        optimizer=optimizer,
                        scheduler=lr_scheduler, # Scheduler state is based on optimizer steps
                        metrics=metrics, 
                        config=config, 
                        val_loss=most_recent_val_loss,
                        tau=current_tau_ckpt,
                        alpha=current_alpha_ckpt,
                        wandb_run_id=current_wandb_run_id,
                        additional_name="",
                        # Add additional metadata for proper resuming
                        current_epoch=current_epoch,
                        batch_within_epoch=step % steps_per_epoch if steps_per_epoch > 0 else step,
                        steps_per_epoch=steps_per_epoch,
                        # Save gradient scaler state if using mixed precision
                        scaler=scaler.state_dict() if scaler is not None else None,
                        # Save whether we're mid-accumulation
                        accumulation_step=(step % gradient_accumulation_steps) + 1
                    )
                    if saved_path:
                        log.info(f"Checkpoint saved: {saved_path}")
                    else:
                        log.info(f"Checkpoint not saved at step {step} (e.g., max_checkpoints reached or interval not met).")
                else:
                    log.info(f"Checkpoint not saved at step {step} (e.g., val_loss is nan).")
    except KeyboardInterrupt:
        if is_main():
            log.warning("KeyboardInterrupt detected! Saving checkpoint before exit. {e}")
        # Use a different name for the checkpoint to indicate interruption
        final_ckpt_name = "interrupt"
        min_to_save_ckpt = 1500  
        if step < min_to_save_ckpt:
            log.warning(f"KeyboardInterrupt detected! Not saving as step is less than {min_to_save_ckpt}.")
            raise KeyboardInterrupt(f"KeyboardInterrupt detected! Not saving as step is less than {min_to_save_ckpt}.")
    else:
        final_ckpt_name = "final"

    # Final checkpoint (also used for KeyboardInterrupt)
    if is_main() and checkpoint_manager.save_at_end:
        current_epoch_for_ckpt = max_steps // steps_per_epoch if steps_per_epoch > 0 else 0
        current_epoch_num_for_ckpt = current_epoch_for_ckpt + 1 if max_steps > 0 else 1
        step_for_ckpt = max_steps - 1 if max_steps > 0 else -1

        final_micro_step_for_sched = max_steps - 1 if max_steps > 0 else 0
        final_epoch_for_sched = final_micro_step_for_sched // steps_per_epoch if steps_per_epoch > 0 else 0

        final_tau = get_schedule_value(
            config['gumbel_tau_schedule'],
            final_micro_step_for_sched,
            max_steps,
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

        final_metrics_to_save = metrics if 'metrics' in locals() else {}

        final_checkpoint_path = checkpoint_manager.save_checkpoint(
            step=step_for_ckpt,
            epoch=current_epoch_num_for_ckpt,
            models={'decoder': decoder_base, 'encoder': encoder_base},
            optimizer=optimizer,
            scheduler=lr_scheduler,
            metrics=final_metrics_to_save,
            config=config,
            tau=final_tau,
            alpha=final_alpha,
            val_loss=most_recent_val_loss,
            wandb_run_id=current_wandb_run_id,
            additional_name=final_ckpt_name
        )
        if final_checkpoint_path:
            log.info(f"Final checkpoint saved: {final_checkpoint_path}")
        else:
            log.info("Final checkpoint not saved (e.g. disabled).")

    if is_main():
        log.info("Training completed!")
    
    # Cleanup distributed training
    cleanup_distributed()


if __name__ == "__main__":
    main()