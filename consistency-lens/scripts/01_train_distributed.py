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
from lens.utils.checkpoint import (
    save_checkpoint, 
    load_checkpoint, 
    CheckpointManager,
    save_training_checkpoint,
    save_legacy_checkpoint,
)
from lens.evaluation.metrics import build_metrics
from lens.evaluation.verbose_samples import maybe_print_verbose_samples
from lens.models.utils import prepare_inputs
from lens.utils.schedule_parser import parse_schedule
from lens.training.distributed import set_seed

# Import all the utility functions from the original training script
sys.path.insert(0, str(Path(__file__).parent))
import importlib
train_module = importlib.import_module("01_train")
get_gpu_memory_info = train_module.get_gpu_memory_info
get_gpu_utilization = train_module.get_gpu_utilization
extract_dataset_info = train_module.extract_dataset_info
resolve_path = train_module.resolve_path
generate_run_name = train_module.generate_run_name  
count_dataset_samples = train_module.count_dataset_samples
check_explicit_requirements = train_module.check_explicit_requirements
prepare_dataset_and_loaders = train_module.prepare_dataset_and_loaders


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
    gradient_accumulation_steps=1,
    is_distributed=False,
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
    encoder_base = encoder.module if hasattr(encoder, 'module') else encoder
    
    # Prepare models dict for train_step
    models = {
        "dec": decoder_base,
        "enc": encoder_base,
        "orig": orig_model
    }
    
    # Loss function parameters
    loss_fns = {
        "T_text": config.get('t_text', 8),
        "tau": get_schedule_value(config, 'gumbel_temperature', step),
        "alpha": get_schedule_value(config, 'lm_weight', step),
        "kl_base_weight": config.get('kl_base_weight', 1.0),
        "entropy_weight": config.get('entropy_weight', 0.0),
        "mse_weight": config.get('mse_weight', 0.0),
        "lm_weight": get_schedule_value(config, 'lm_weight', step),
    }
    
    # Context manager for gradient synchronization
    # Key optimization: Use no_sync() to skip gradient synchronization except at accumulation boundaries
    if is_distributed and not is_accumulation_end:
        # Don't sync gradients until the last accumulation step
        sync_context = decoder.no_sync() if hasattr(decoder, 'no_sync') else nullcontext()
        if hasattr(encoder, 'no_sync'):
            sync_context = sync_context.__enter__()
            encoder_sync = encoder.no_sync().__enter__()
        else:
            encoder_sync = None
    else:
        # Normal gradient sync (single GPU or accumulation boundary)
        sync_context = nullcontext()
        encoder_sync = None
    
    try:
        with sync_context:
            # Forward pass
            losses = original_train_step(
                batch=batch,
                models=models,
                _loss_fns=loss_fns,
                lm_loss_natural_prefix=config.get('lm_loss_natural_prefix'),
                tokenizer=None,  # TODO: Add tokenizer if needed
                cached_prefix_ids=None  # TODO: Add cached prefix if needed
            )
            
            # Scale loss by accumulation steps
            loss = losses['total'] / gradient_accumulation_steps
            
            # Backward pass
            if device.type == "cuda" and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
    finally:
        # Clean up context managers
        if encoder_sync is not None:
            encoder_sync.__exit__(None, None, None)
        if is_distributed and not is_accumulation_end and sync_context != nullcontext():
            sync_context.__exit__(None, None, None)
    
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
    
    # Convert Hydra config to a plain Python dict
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
    
    # Extract configuration values
    model_name = config['model_name']
    tokenizer_name = config.get("tokenizer_name", model_name)
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
    # Each GPU will process batch_size samples
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
        config_name = None
        try:
            hydra_cfg = HydraConfig.get()
            if hasattr(hydra_cfg, 'job') and hasattr(hydra_cfg.job, 'config_name'):
                config_name = hydra_cfg.job.config_name
            elif hasattr(hydra_cfg, 'runtime') and hasattr(hydra_cfg.runtime, 'choices'):
                config_name = hydra_cfg.runtime.choices.get('config_name', 'config')
        except:
            pass
        
        run_name_override = config.get('run_name')
        if run_name_override:
            run_name = run_name_override
        else:
            run_name = generate_run_name(config, dataset_info, config.get('resume'), config_name, config.get('run_suffix'))
            # Add distributed suffix
            run_name = f"{run_name}_dist{world_size}"
        
        log.info(f"Run name: {run_name}")
    else:
        run_name = None
    
    # Synchronize run name across processes
    if world_size > 1:
        run_name_tensor = torch.zeros(256, dtype=torch.uint8, device=device)
        if is_main():
            run_name_bytes = run_name.encode('utf-8')[:256]
            run_name_tensor[:len(run_name_bytes)] = torch.tensor(list(run_name_bytes), dtype=torch.uint8)
        dist.broadcast(run_name_tensor, src=0)
        if not is_main():
            run_name_bytes = bytes(run_name_tensor.cpu().numpy())
            run_name = run_name_bytes.decode('utf-8').rstrip('\x00')
    
    # Setup checkpoint directory
    checkpoint_config = config.get('checkpoint', {})
    base_checkpoint_dir = resolve_path(checkpoint_config.get('output_dir', 'outputs'))
    run_checkpoint_dir = base_checkpoint_dir / run_name
    checkpoint_config['output_dir'] = str(run_checkpoint_dir)
    
    # Create checkpoint directory (only on main process)
    if is_main():
        run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Checkpoints will be saved to: {run_checkpoint_dir}")
    
    # Initialize models
    decoder = Decoder(DecoderConfig(**config["decoder"]))
    encoder = Encoder(EncoderConfig(**config["encoder"]))
    orig_model = OrigWrapper(config)
    
    # Wrap models with DDP if needed
    decoder, encoder, orig_model = setup_distributed_models(
        decoder, encoder, orig_model, device, rank, world_size
    )
    
    # Count parameters (only on main process)
    if is_main():
        decoder_base = decoder.module if hasattr(decoder, 'module') else decoder
        encoder_base = encoder.module if hasattr(encoder, 'module') else encoder
        
        num_params_decoder = sum(p.numel() for p in decoder_base.parameters())
        num_params_encoder = sum(p.numel() for p in encoder_base.parameters())
        log.info(f"Decoder parameters: {num_params_decoder:,}")
        log.info(f"Encoder parameters: {num_params_encoder:,}")
    
    # Prepare datasets and dataloaders
    train_dataset, train_loader, val_dataset, val_loader = prepare_dataset_and_loaders(
        config, activation_dir, effective_val_activation_dir, device, log
    )
    
    # Replace dataloaders with distributed versions
    train_loader = get_dataloader_for_distributed(
        train_dataset,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        pin_memory=False,
    )
    
    if val_dataset is not None:
        val_loader = get_dataloader_for_distributed(
            val_dataset,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
            pin_memory=False,
        )
    
    # Setup optimizer (same as original)
    trainable_components_config = config.get('trainable_components', {})
    decoder_train_cfg = trainable_components_config.get('decoder', {})
    encoder_train_cfg = trainable_components_config.get('encoder', {})
    custom_lr_multipliers = config.get('custom_lr_multipliers', {})
    
    # Get base models for parameter groups
    decoder_base = decoder.module if hasattr(decoder, 'module') else decoder
    encoder_base = encoder.module if hasattr(encoder, 'module') else encoder
    
    params = param_groups(
        decoder_base, 
        encoder_base, 
        learning_rate, 
        config, 
        decoder_train_cfg, 
        encoder_train_cfg,
        custom_lr_multipliers
    )
    optimizer = torch.optim.AdamW(params, weight_decay=config["weight_decay"])
    
    # Learning rate scheduler
    lr_scheduler_cfg = config.get('lr_scheduler', {})
    lr_scheduler = get_lr_scheduler(optimizer, lr_scheduler_cfg, max_steps)
    
    # Initialize gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    
    # Initialize W&B (only on main process)
    if is_main():
        wandb_config = config.get('wandb', {})
        run_info = log_init(config, run_name=run_name)
        
        # Save config
        config_save_path = run_checkpoint_dir / "config.yaml"
        with open(config_save_path, "w") as f:
            OmegaConf.save(cfg, f)
        log.info(f"Config saved to: {config_save_path}")
    else:
        run_info = None
    
    # Resume from checkpoint if specified
    start_step = 0
    if config.get('resume'):
        checkpoint_path = Path(config['resume'])
        if checkpoint_path.exists():
            if is_main():
                log.info(f"Resuming from checkpoint: {checkpoint_path}")
            
            checkpoint_data = load_checkpoint(
                checkpoint_path,
                decoder_base,
                encoder_base,
                optimizer,
                lr_scheduler,
                device,
                log
            )
            start_step = checkpoint_data.get('step', 0) + 1
            
            if is_main():
                log.info(f"Resumed from step {start_step}")
    
    # Training loop
    if is_main():
        log.info("Starting training...")
        checkpoint_manager = CheckpointManager(
            output_dir=run_checkpoint_dir,
            max_checkpoints=checkpoint_config.get('max_checkpoints', 5)
        )
    
    # Initialize metrics
    running_losses = deque(maxlen=100)
    step_times = deque(maxlen=100)
    
    # Zero gradients at start
    optimizer.zero_grad(set_to_none=True)
    
    # Main training loop
    for step in range(start_step, max_steps):
        step_start_time = time.time()
        
        # Set epoch for DistributedSampler
        if world_size > 1 and hasattr(train_loader.sampler, 'set_epoch'):
            epoch = step // len(train_loader)
            train_loader.sampler.set_epoch(epoch)
        
        # Training step with optimized gradient accumulation
        batch = next(iter(train_loader))
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
            gradient_accumulation_steps=gradient_accumulation_steps,
            is_distributed=(world_size > 1),
        )
        
        # Update learning rate scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        # Synchronize metrics across processes
        metrics = sync_metrics(metrics, world_size)
        
        # Update running metrics
        running_losses.append(metrics['loss'])
        step_times.append(time.time() - step_start_time)
        
        # Logging (only on main process)
        if is_main() and step % config.get('log_interval', 10) == 0:
            avg_loss = sum(running_losses) / len(running_losses)
            avg_step_time = sum(step_times) / len(step_times)
            samples_per_sec = effective_batch_size / avg_step_time
            
            current_lr = optimizer.param_groups[0]['lr']
            accumulation_step = (step % gradient_accumulation_steps) + 1
            
            log.info(
                f"Step {step}/{max_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Samples/sec: {samples_per_sec:.1f} | "
                f"Acc: {accumulation_step}/{gradient_accumulation_steps}"
            )
            
            # Log to W&B
            if run_info:
                log_metrics({
                    'train/loss': avg_loss,
                    'train/learning_rate': current_lr,
                    'train/samples_per_second': samples_per_sec,
                    'train/gpu_memory_allocated': torch.cuda.memory_allocated() / 1e9,
                    'train/gradient_accumulation_step': accumulation_step,
                    **metrics
                }, step)
        
        # Checkpointing (only on main process)
        if is_main() and step > 0 and step % checkpoint_config.get('save_interval', 100) == 0:
            checkpoint_path = checkpoint_manager.save(
                step=step,
                model_state_dict={
                    'decoder': decoder_base.state_dict(),
                    'encoder': encoder_base.state_dict(),
                },
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=lr_scheduler.state_dict() if lr_scheduler else None,
                metrics=metrics,
                config=config,
            )
            log.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Validation (only on main process for now)
        if is_main() and val_loader and step % config.get('val_interval', 500) == 0:
            # TODO: Implement distributed validation
            pass
    
    # Final checkpoint
    if is_main():
        final_checkpoint_path = checkpoint_manager.save(
            step=max_steps,
            model_state_dict={
                'decoder': decoder_base.state_dict(),
                'encoder': encoder_base.state_dict(),
            },
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=lr_scheduler.state_dict() if lr_scheduler else None,
            metrics=metrics,
            config=config,
            is_final=True,
        )
        log.info(f"Final checkpoint saved: {final_checkpoint_path}")
        log.info("Training completed!")
    
    # Cleanup distributed training
    cleanup_distributed()


if __name__ == "__main__":
    main()