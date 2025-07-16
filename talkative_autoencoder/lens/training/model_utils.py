"""Helper utilities for model management in distributed training."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging


def should_move_orig_to_cpu(
    config: Dict[str, Any],
    shared_base_model: Optional[nn.Module],
    is_validation: bool = False,
) -> bool:
    """
    Determine if orig_model should be moved to CPU for memory optimization.
    
    The model should only be moved to CPU if:
    1. It's not being used for GRPO or LM loss computation
    2. There's no shared base model (which would make moving inefficient)
    3. We're in validation phase (orig_model only used for KL computation)
    
    Args:
        config: Training configuration
        shared_base_model: The shared base model object (None if not sharing)
        is_validation: Whether we're in validation phase
        
    Returns:
        bool: True if model should be moved to CPU
    """
    # Check if GRPO or LM losses are active
    grpo_active = config.get('GRPO_beta', 0) > 0
    lm_active = config.get('lm_base_weight', 0) > 0
    
    # Don't move to CPU if model is actively used in training
    if grpo_active or lm_active:
        return False
    
    # Don't move to CPU if using shared base model (would break sharing)
    if shared_base_model is not None:
        return False
    
    # In validation, we can safely move to CPU since orig_model is only used
    # for KL computation which happens on device
    return True


def validate_model_setup(
    decoder: nn.Module,
    encoder: nn.Module,
    orig_model: Any,  # OrigWrapper type
    shared_base_model: Optional[nn.Module],
    config: Dict[str, Any],
    log: logging.Logger,
) -> bool:
    """
    Validate that models are properly configured.
    
    Checks:
    - Decoder and encoder are in correct mode
    - OrigWrapper is in eval mode and parameters are frozen
    - Shared base model is properly shared if configured
    - Device placement is consistent
    
    Args:
        decoder: Decoder model
        encoder: Encoder model
        orig_model: OrigWrapper instance
        shared_base_model: Shared base model (if any)
        config: Training configuration
        log: Logger instance
        
    Returns:
        bool: True if validation passes
    """
    issues = []
    
    # Check training modes
    decoder_base = decoder.module if hasattr(decoder, 'module') else decoder
    encoder_base = encoder.module if hasattr(encoder, 'module') else encoder
    
    if not decoder_base.training:
        issues.append("Decoder should be in training mode")
    if not encoder_base.training:
        issues.append("Encoder should be in training mode")
    if orig_model.model.training:
        issues.append("OrigWrapper model should be in eval mode")
    
    # Check orig_model parameters are frozen
    for name, param in orig_model.model.named_parameters():
        if param.requires_grad:
            issues.append(f"OrigWrapper parameter {name} has requires_grad=True (should be frozen)")
            break
    
    # Check shared base model configuration
    decoder_cfg = config.get('trainable_components', {}).get('decoder', {})
    encoder_cfg = config.get('trainable_components', {}).get('encoder', {})
    
    should_share = (
        not decoder_cfg.get('base_model', False) and
        not (encoder_cfg.get('base_model', True) and encoder_cfg.get('use_base_model', False))
    )
    
    if should_share and shared_base_model is None:
        issues.append("Models should share base but shared_base_model is None")
    elif not should_share and shared_base_model is not None:
        issues.append("Models should not share base but shared_base_model is provided")
    
    # Check if shared base model is actually being used
    if shared_base_model is not None:
        # For decoder
        if hasattr(decoder_base, 'base') and decoder_base.base is not shared_base_model:
            if id(decoder_base.base) != id(shared_base_model):
                issues.append("Decoder base model is not using the shared instance")
        
        # For encoder (if it has a base model)
        if hasattr(encoder_base, 'base') and encoder_cfg.get('use_base_model', False):
            if encoder_base.base is not shared_base_model:
                if id(encoder_base.base) != id(shared_base_model):
                    issues.append("Encoder base model is not using the shared instance")
        
        # For orig_model
        if hasattr(orig_model, 'model') and orig_model.model is not shared_base_model:
            if id(orig_model.model) != id(shared_base_model):
                issues.append("OrigWrapper is not using the shared base model instance")
    
    # Log results
    if issues:
        log.warning("Model setup validation found issues:")
        for issue in issues:
            log.warning(f"  - {issue}")
        return False
    else:
        log.info("Model setup validation passed")
        return True


def sync_model_devices(
    models: Dict[str, nn.Module],
    target_device: torch.device,
    shared_base_model: Optional[nn.Module],
    config: Dict[str, Any],
    log: logging.Logger,
) -> None:
    """
    Ensure models are on the expected devices based on configuration.
    
    Args:
        models: Dictionary of models (dec, enc, orig)
        target_device: Target CUDA device
        shared_base_model: Shared base model instance
        config: Training configuration
        log: Logger instance
    """
    # Decoder and encoder should always be on target device
    if 'dec' in models:
        current_device = next(models['dec'].parameters()).device
        if current_device != target_device:
            log.warning(f"Decoder on {current_device}, moving to {target_device}")
            models['dec'].to(target_device)
    
    if 'enc' in models:
        current_device = next(models['enc'].parameters()).device
        if current_device != target_device:
            log.warning(f"Encoder on {current_device}, moving to {target_device}")
            models['enc'].to(target_device)
    
    # Handle orig_model based on CPU movement logic
    if 'orig' in models:
        orig_model = models['orig']
        should_be_on_cpu = should_move_orig_to_cpu(config, shared_base_model)
        
        # Check current device
        if hasattr(orig_model, 'model'):
            current_device = next(orig_model.model.parameters()).device
            
            if should_be_on_cpu and current_device.type != 'cpu':
                log.info("Moving orig_model to CPU for memory optimization")
                orig_model.to('cpu')
            elif not should_be_on_cpu and current_device != target_device:
                log.info(f"Moving orig_model to {target_device}")
                orig_model.to(target_device)


def log_device_info(
    models: Dict[str, Any],
    shared_base_model: Optional[nn.Module],
    log: logging.Logger,
) -> None:
    """Log current device placement of all models."""
    device_info = []
    
    for name, model in models.items():
        if model is not None:
            # Get the actual model (unwrap DDP if needed)
            base_model = model.module if hasattr(model, 'module') else model
            
            # For OrigWrapper, check the inner model
            if hasattr(base_model, 'model'):
                device = next(base_model.model.parameters()).device
            else:
                device = next(base_model.parameters()).device
            
            device_info.append(f"{name}: {device}")
    
    if shared_base_model is not None:
        device = next(shared_base_model.parameters()).device
        device_info.append(f"shared_base: {device}")
    
    log.info(f"Model devices: {', '.join(device_info)}")