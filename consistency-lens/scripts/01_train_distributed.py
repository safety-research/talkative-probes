#!/usr/bin/env python3
"""Distributed training script for Consistency Lens with multi-GPU support and proper gradient accumulation."""

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


class Timer:
    def __init__(self, name: str, logger: logging.Logger, main_process: bool = True):
        self.name = name
        self.logger = logger
        self.main_process = main_process
        self.start_time = None

    def __enter__(self):
        if self.main_process:
            self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.main_process and self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            self.logger.info(f"Phase '{self.name}' took {elapsed_time:.2f}s")


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
                                step // gradient_accumulation_steps if gradient_accumulation_steps > 0 else 0, 
                                gradient_accumulation_steps),
        "alpha": get_schedule_value(config['alpha_schedule'], step, config['max_train_steps'],
                                  step // gradient_accumulation_steps if gradient_accumulation_steps > 0 else 0, 
                                  gradient_accumulation_steps),
        "kl_base_weight": config.get('kl_base_weight', 1.0),
        "entropy_weight": config.get('entropy_weight', 0.0),
        "mse_weight": config.get('mse_weight', 0.0),
        "lm_weight": get_schedule_value(config['alpha_schedule'], step, config['max_train_steps'],
                                      step // gradient_accumulation_steps if gradient_accumulation_steps > 0 else 0, 
                                      gradient_accumulation_steps),
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
        # Forward pass
        losses = original_train_step(
            batch=batch,
            models=models,
            _loss_fns=loss_fns,
            lm_loss_natural_prefix=config.get('lm_loss_natural_prefix'),
            tokenizer=tokenizer,
            cached_prefix_ids=cached_prefix_ids
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
        
        # Optimizer step
        if device.type == "cuda" and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Update learning rate scheduler AFTER optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        # Zero gradients for next iteration
        optimizer.zero_grad(set_to_none=True)
    else:
        grad_norm = None
    
    # Return metrics
    metrics = {
        'loss': losses['total'].item(),
        'loss_mse': losses['mse'].item(),
        'loss_lm': losses['lm'].item(),
        'loss_kl': losses['kl'].item(),
        'loss_entropy': losses.get('entropy', 0.0).item() if 'entropy' in losses else 0.0,
        'grad_norm': grad_norm.item() if grad_norm is not None else 0.0,
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
        sampler = DistributedSampler(
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

    # --- Timer for Model Setup ---
    with Timer("Model Setup (Init, DDP)", log, main_process=is_main()):
        # Extract trainable components configuration
        trainable_components_config = config.get('trainable_components', {})
        decoder_train_cfg = trainable_components_config.get('decoder', {})
        encoder_train_cfg = trainable_components_config.get('encoder', {})
        
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
        if 'decoder_prompt' in config and config['decoder_prompt']:
            if is_main():
                log.info(f"Setting decoder prompt: \"{config['decoder_prompt']}\"")
            decoder.set_prompt(config['decoder_prompt'], tokenizer)
        elif is_main():
            log.warning("Decoder prompt ('decoder_prompt') not found in config or is empty. Decoder soft prompts will not be initialized from text.")

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
        
        train_loader = get_dataloader_for_distributed(
            train_ds, batch_size=batch_size, world_size=world_size, rank=rank, shuffle=True,
            collate_fn=collate, num_workers=0, pin_memory=False,
        )
        
        if val_ds is not None:
            val_loader = get_dataloader_for_distributed(
                val_ds, batch_size=batch_size, world_size=world_size, rank=rank, shuffle=False,
                collate_fn=collate, num_workers=0, pin_memory=False,
            )
        
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

        # Calculate the number of optimizer steps
        max_optimizer_steps = max_steps // gradient_accumulation_steps
        if max_steps % gradient_accumulation_steps != 0: # Account for any remaining steps
            max_optimizer_steps +=1
        if is_main():
            log.info(f"Total micro-steps (fwd/bwd passes): {max_steps}")
            log.info(f"Total optimizer steps (scheduler steps): {max_optimizer_steps}")

        # Parse flexible interval settings (log / wandb / val) now that steps_per_epoch is known
        # These intervals are based on micro-steps (main loop steps)
        log_interval = _resolve_schedule_to_steps(config['log_interval'], steps_per_epoch, log, "log_interval")
        wandb_log_interval = _resolve_schedule_to_steps(config['wandb_log_interval'], steps_per_epoch, log, "wandb_log_interval")
        val_interval_str = config.get('val_interval', "500s")
        val_interval = _resolve_schedule_to_steps(val_interval_str, steps_per_epoch, log, "val_interval")

        if log_interval <= 0:
            raise ValueError(f"log_interval must be positive, got {config['log_interval']}")
        if wandb_log_interval <= 0:
            raise ValueError(f"wandb_log_interval must be positive, got {config['wandb_log_interval']}")

    # --- Timer for Optimizer and Scheduler Setup ---
    with Timer("Optimizer and Scheduler Setup", log, main_process=is_main()):
        trainable_components_config = config.get('trainable_components', {})
        decoder_train_cfg = trainable_components_config.get('decoder', {})
        encoder_train_cfg = trainable_components_config.get('encoder', {})
        custom_lr_multipliers = config.get('custom_lr_multipliers', {})
        
        # Get base models for parameter groups
        decoder_base = decoder.module if hasattr(decoder, 'module') else decoder
        encoder_base = encoder.module if hasattr(encoder, 'module') else encoder
        
        # Extract learning rate multipliers from config
        projection_lr_multiplier = custom_lr_multipliers.get('projection_layers', 1.0)
        embedding_lr_multiplier = custom_lr_multipliers.get('embedding_layers', 1.0)
        prompt_lr_multiplier = custom_lr_multipliers.get('prompt_layers', 1.0)
        
        params = param_groups(
            [decoder_base, encoder_base], 
            learning_rate, 
            projection_lr_multiplier, 
            embedding_lr_multiplier, 
            prompt_lr_multiplier
        )
        optimizer = torch.optim.AdamW(params)
        
        # Learning rate scheduler
        lr_scheduler_cfg = config.get('lr_scheduler', {})
        # Initialize scheduler with max_optimizer_steps
        lr_scheduler = get_lr_scheduler(optimizer, lr_scheduler_cfg, max_optimizer_steps) 
        
        # Initialize gradient scaler for mixed precision
        scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    # Initialize CheckpointManager (after steps_per_epoch is known)
    # It uses the original Hydra 'cfg' to correctly parse schedule strings for save intervals.
    checkpoint_manager = CheckpointManager(config, log, steps_per_epoch)
    
    # Initialize W&B (only on main process)
    if is_main():
        wandb_config = config.get('wandb', {})
        
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
            
            rec = checkpoint_manager.load_checkpoint(
                checkpoint_path_str,
                models=models_to_load,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                map_location=device
            )
            start_step = int(rec.get("step", -1)) + 1
            
            if is_main():
                log.info(f"Resumed from micro-step {start_step}")
                if lr_scheduler and ('scheduler_state_dict' not in rec or 'scheduler' not in rec):
                    log.warning("Scheduler state not found in checkpoint, or not loaded by CheckpointManager.")
                elif lr_scheduler and ('scheduler_state_dict' in rec or 'scheduler' in rec):
                    log.info("Scheduler state successfully loaded via CheckpointManager.")
        else:
            if is_main():
                log.warning(f"Resume checkpoint path not found: {checkpoint_path_str}. Starting from scratch.")
    
    # Training loop
    if is_main():
        log.info("Starting training...")
        # CheckpointManager is already initialized
    
    # Initialize metrics
    running_losses = deque(maxlen=100)
    step_times = deque(maxlen=100)
    
    # Zero gradients at start
    optimizer.zero_grad(set_to_none=True)
    
    # Main training loop
    for step in range(start_step, max_steps):
        step_start_time = time.time()
        
        current_epoch = step // steps_per_epoch if steps_per_epoch > 0 else 0
        
        # Set epoch for DistributedSampler
        if world_size > 1 and hasattr(train_loader.sampler, 'set_epoch'):
            epoch = step // len(train_loader)
            train_loader.sampler.set_epoch(epoch)
        
        # Training step with optimized gradient accumulation
        batch = next(iter(train_loader))
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)} # Move batch to device

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
            lr_scheduler=lr_scheduler
        )
        
        # Synchronize metrics across processes
        metrics = sync_metrics(metrics, world_size)
        
        # Update running metrics
        running_losses.append(metrics['loss'])
        step_times.append(time.time() - step_start_time)
        
        # Average metrics for logging
        avg_loss = sum(running_losses) / len(running_losses)
        avg_step_time = sum(step_times) / len(step_times)
        samples_per_sec = effective_batch_size / avg_step_time

        current_lr = optimizer.param_groups[0]['lr']
        accumulation_step = (step % gradient_accumulation_steps) + 1
        
        # Performance metrics (calculated similarly to 01_train.py)
        steps_per_second = 1.0 / avg_step_time if avg_step_time > 0 else 0
        tokens_per_second = samples_per_sec * config.get('t_text', 10)

        # Console logging (only on main process)
        if is_main() and (step % log_interval == 0 or step == max_steps - 1):
            log.info(
                f"Step {step}/{max_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Samples/sec: {samples_per_sec:.1f} | "
                f"Acc: {accumulation_step}/{gradient_accumulation_steps}"
            )

        # W&B logging (only on main process)
        if is_main() and (step % wandb_log_interval == 0 or step == max_steps - 1):
            # Get schedule values for logging (consistent with 01_train.py)
            # Current optimizer step for schedule value calculation
            current_optimizer_step_for_sched = step // gradient_accumulation_steps
            
            current_tau_log = get_schedule_value(config['gumbel_tau_schedule'], step, max_steps,
                                                         current_optimizer_step_for_sched, gradient_accumulation_steps) # pass micro_step, total_micro_steps, optimizer_step, grad_accum
            current_alpha_log = get_schedule_value(config['alpha_schedule'], step, max_steps,
                                                           current_optimizer_step_for_sched, gradient_accumulation_steps)
            
            wandb_metrics = {
                'train/loss': avg_loss, # This is the running average loss
                'loss/total': metrics['loss'], # This is the current step's synced loss
                'loss/mse': metrics['loss_mse'],
                'loss/lm': metrics['loss_lm'],
                'loss/kl': metrics['loss_kl'],
                'loss/entropy': metrics['loss_entropy'],
                'params/tau': current_tau_log,
                'params/alpha': current_alpha_log,
                'params/lm_w': get_schedule_value(config['alpha_schedule'], step, max_steps, current_optimizer_step_for_sched, gradient_accumulation_steps), # lm_weight is alpha
                'params/kl_w': config.get('kl_base_weight', 1.0),
                'params/entropy_w': config.get('entropy_weight', 0.0),
                'optim/lr': current_lr,
                'grads/norm': metrics.get('grad_norm', 0.0), # grad_norm is from the accumulation end step
                # 'updates/norm' and 'updates/ratio' are harder to get accurately here without DDP sync after optimizer step
                # and before next backward pass if not an accumulation_end step.
                # For simplicity, we'll omit them or log them as 0 if not an accumulation_end step.
                # 'update_norm' and 'param_norm' would require synchronizing model parameters or their checksums,
                # which is complex and potentially slow.
                # We can log grad_norm as it's available after clipping on accumulation_end.
                'gradient_accumulation/is_update_step': 1 if ((step + 1) % gradient_accumulation_steps == 0) or (step == max_steps - 1) else 0,
                'gradient_accumulation/accumulation_step': accumulation_step,
                'performance/steps_per_second': steps_per_second,
                'performance/samples_per_second': samples_per_sec, # effective_batch_size / avg_step_time
                'performance/tokens_per_second': tokens_per_second,
                'performance/avg_step_time': avg_step_time,
                'train/gpu_memory_allocated': torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                'train/gradient_accumulation_step': accumulation_step,
            }
            
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
        
        # Checkpointing (only on main process)
        if is_main() and checkpoint_manager.should_save_step(step):
            # Get current tau and alpha for checkpointing, similar to 01_train.py
            # Checkpointing metadata should reflect the state at the current *micro-step*
            # but schedules are evaluated based on optimizer steps.
            current_optimizer_step_for_ckpt = step // gradient_accumulation_steps
            current_tau_ckpt = get_schedule_value(config['gumbel_tau_schedule'], step, max_steps, 
                                           current_optimizer_step_for_ckpt, 
                                           gradient_accumulation_steps)
            current_alpha_ckpt = get_schedule_value(config['alpha_schedule'], step, max_steps,
                                             current_optimizer_step_for_ckpt, 
                                             gradient_accumulation_steps)
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
                wandb_run_id=current_wandb_run_id
            )
            if saved_path:
                log.info(f"Checkpoint saved: {saved_path}")
            else:
                log.info(f"Checkpoint not saved at step {step} (e.g., max_checkpoints reached or interval not met).")
        
        # Validation (only on main process for now)
        if is_main() and val_loader and val_interval > 0 and (step % val_interval == 0):
            # TODO: Implement distributed validation
            log.info(f"Skipping validation at step {step} (TODO: Implement)")
            pass # Placeholder for validation logic

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
                        verbose_interval_str, steps_per_epoch, log, "verbose_interval"
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
                    log.info(f"Generating verbose samples at step {step}, epoch {current_epoch}")
                    decoder.eval()
                    encoder.eval()
                    orig_model.model.eval() # Ensure orig model is also in eval

                    # Get schedule arguments for verbose samples - these use optimizer steps
                    current_optimizer_step_for_verbose = step // gradient_accumulation_steps
                    sch_args_verbose = {
                        "tau": get_schedule_value(config['gumbel_tau_schedule'], step, max_steps,
                                                 current_optimizer_step_for_verbose, 
                                                 gradient_accumulation_steps),
                        "T_text": config.get('t_text', 8),
                        "alpha": get_schedule_value(config['alpha_schedule'], step, max_steps,
                                                   current_optimizer_step_for_verbose, 
                                                   gradient_accumulation_steps),
                        "lm_weight": get_schedule_value(config['alpha_schedule'], step, max_steps, # This is alpha
                                                       current_optimizer_step_for_verbose, 
                                                       gradient_accumulation_steps),
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
                        capture_output=True
                    )
                    
                    # current_wandb_run_id is already defined
                    if captured_text and current_wandb_run_id:
                        verbose_samples_logger.log_verbose_samples(
                            captured_text,
                            step=step,
                            table_name="training_verbose_samples_dist" # Differentiate if needed
                        )

                    decoder.train()
                    encoder.train()
                    orig_model.model.train() # Set orig model back to train if it has trainable parts (usually not)

    
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
        # Note: get_schedule_value expects micro_step, total_micro_steps, optimizer_step, grad_accum_steps
        final_micro_step_for_sched = max_steps -1 if max_steps > 0 else 0

        final_tau = get_schedule_value(
            config['gumbel_tau_schedule'], 
            final_micro_step_for_sched, # current micro_step
            max_steps,                   # total micro_steps
            calc_optimizer_step_for_final_sched, # current optimizer_step
            gradient_accumulation_steps
        )
        final_alpha = get_schedule_value(
            config['alpha_schedule'], 
            final_micro_step_for_sched,
            max_steps,
            calc_optimizer_step_for_final_sched,
            gradient_accumulation_steps
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
            wandb_run_id=current_wandb_run_id
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