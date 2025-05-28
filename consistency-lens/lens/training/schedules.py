from __future__ import annotations

import math
import re
from typing import Dict, Literal, Optional, Union, Any
from dataclasses import dataclass
from contextlib import contextmanager, nullcontext

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, ExponentialLR, PolynomialLR
from torch.amp import autocast

ScheduleType = Literal["constant", "linear_decay", "linear_warmup", "cosine_anneal", "exponential_decay"]
LRSchedulerType = Literal["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "exponential"]

__all__ = [
    "get_schedule_value", 
    "get_lr_scheduler", 
    "get_lr_scheduler_with_warmup",
    "ScheduleSpec",
    "parse_schedule_value",
    "parse_schedule_config", 
    "resolve_schedule_at_step",
    "get_schedule_value_for_logging",
    "get_autocast_context",
    "apply_gradient_scaling",
    "optimizer_step",
    "convert_schedule_strings_in_config",
    "spec_to_steps",
    "parse_schedule_to_steps"
]


def get_schedule_value(
    schedule_config: Dict,
    current_step: int,
    total_steps: int | None = None, # Required for some schedules like cosine_anneal
    current_epoch: int | None = None,  # For epoch-based schedules
    steps_per_epoch: int | None = None,  # For converting epochs to steps
) -> float:
    """Calculates the current value for a hyperparameter based on its schedule.

    Args:
        schedule_config: Dictionary defining the schedule.
            Expected keys: 'type', and type-specific keys like 'value',
            'start_value', 'end_value', 'num_steps'.
        current_step: The current training step.
        total_steps: The total number of training steps, required for schedules
            that depend on the full training duration (e.g., cosine_anneal).
        current_epoch: The current training epoch (for epoch-based schedules).
        steps_per_epoch: Number of steps per epoch (for converting epochs to steps).

    Returns:
        The calculated value for the hyperparameter at the current_step.
    """
    schedule_type: ScheduleType = schedule_config["type"]

    if schedule_type == "constant":
        return float(schedule_config["value"])
    start_value = float(schedule_config["start_value"])
    end_value = float(schedule_config["end_value"])
    
    # Parse num_steps which may be a string with s/e notation
    num_steps_raw = schedule_config.get("num_steps", -1)
    if isinstance(num_steps_raw, str) and num_steps_raw == "-1":
        if total_steps is None:
            raise ValueError("total_steps is required when num_steps is -1")
        num_schedule_steps = total_steps
    else:
        num_schedule_steps = parse_schedule_to_steps(num_steps_raw, steps_per_epoch)
        if num_schedule_steps == -1:
            if total_steps is None:
                raise ValueError("total_steps is required when num_steps is -1")
            num_schedule_steps = total_steps
    
    if num_schedule_steps == 0:
        raise ValueError(f"num_schedule_steps cannot be 0 (got {num_steps_raw})")
        
    progress = min(1.0, current_step / num_schedule_steps)

    if schedule_type == "linear_decay":
        return start_value - progress * (start_value - end_value)
    if schedule_type == "linear_decay_after_constant":
        constant_steps_raw = schedule_config.get("constant_steps_before_linear_decay", 0)
        constant_steps = parse_schedule_to_steps(constant_steps_raw, steps_per_epoch)
        
        if current_step < constant_steps:
            return start_value
        else:
            return start_value - (current_step - constant_steps) * (start_value - end_value) / (num_schedule_steps - constant_steps)
    elif schedule_type == "linear_warmup":
        return start_value + progress * (end_value - start_value)
    elif schedule_type == "cosine_anneal":
        if total_steps is None and num_schedule_steps == current_step + 1:
             # User likely didn't provide total_steps, and num_steps wasn't set for cosine
             # Default to a full cosine cycle over num_schedule_steps if it was provided, or warn/error
             pass # Or raise error if total_steps is critical and not inferable
        # Ensuring progress for cosine is over the intended schedule duration
        # If num_schedule_steps is different from total_steps, cosine anneals over num_schedule_steps
        cosine_progress = min(1.0, current_step / num_schedule_steps) 
        return end_value + 0.5 * (start_value - end_value) * (1 + math.cos(math.pi * cosine_progress))
    elif schedule_type == "exponential_decay":
        if start_value <= 0 or end_value <= 0: # Avoid log(0) or issues with negative values
            raise ValueError("Exponential decay requires positive start and end values.")
        decay_rate = (end_value / start_value) ** (1 / num_schedule_steps)
        return start_value * (decay_rate**current_step)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def get_lr_scheduler(
    optimizer: Optimizer,
    scheduler_config: Dict,
    num_training_steps: int,
    last_epoch: int = -1,
    current_epoch: int = 0,
    steps_per_epoch: int = None,
) -> Optional[_LRScheduler]:
    """Creates a learning rate scheduler based on configuration.
    
    Args:
        optimizer: The optimizer to schedule
        scheduler_config: Configuration dictionary with 'type' and scheduler-specific params
        num_training_steps: Total number of training steps
        last_epoch: The index of the last epoch (default: -1)
        current_epoch: Current training epoch (for epoch-based scheduling)
        steps_per_epoch: Number of steps per epoch (for converting epochs to steps)
        
    Returns:
        A PyTorch learning rate scheduler or None if type is 'constant'
    """
    scheduler_type = scheduler_config.get('type', 'constant')
    
    if scheduler_type == 'constant':
        return None
    
    # Handle warmup if specified
    warmup_steps_raw = scheduler_config.get('warmup_steps', 0)
    warmup_steps = parse_schedule_to_steps(warmup_steps_raw, steps_per_epoch)
    
    if warmup_steps > 0:
        return get_lr_scheduler_with_warmup(
            optimizer, scheduler_config, num_training_steps, last_epoch,
            current_epoch, steps_per_epoch
        )
    
    # Create scheduler based on type
    if scheduler_type == 'linear':
        end_factor = scheduler_config.get('end_factor', 0.0)
        return PolynomialLR(
            optimizer,
            total_iters=num_training_steps,
            power=1.0,
            last_epoch=last_epoch,
            end_lr=optimizer.param_groups[0]['lr'] * end_factor,
        )
    
    elif scheduler_type == 'cosine':
        eta_min = scheduler_config.get('eta_min', 0.0)
        return CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )
    
    elif scheduler_type == 'cosine_with_restarts':
        eta_min = scheduler_config.get('eta_min', 0.0)
        T_0 = parse_schedule_to_steps(scheduler_config.get('T_0', 500), steps_per_epoch)
        T_mult = scheduler_config.get('T_mult', 2)
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )
    
    elif scheduler_type == 'polynomial':
        power = scheduler_config.get('power', 1.0)
        end_factor = scheduler_config.get('end_factor', 0.0)
        return PolynomialLR(
            optimizer,
            total_iters=num_training_steps,
            power=power,
            last_epoch=last_epoch,
            end_lr=optimizer.param_groups[0]['lr'] * end_factor,
        )
    
    elif scheduler_type == 'exponential':
        gamma = scheduler_config.get('gamma', 0.95)
        return ExponentialLR(
            optimizer,
            gamma=gamma,
            last_epoch=last_epoch,
        )
    
    else:
        raise ValueError(f"Unknown LR scheduler type: {scheduler_type}")


def get_lr_scheduler_with_warmup(
    optimizer: Optimizer,
    scheduler_config: Dict,
    num_training_steps: int,
    last_epoch: int = -1,
    current_epoch: int = 0,
    steps_per_epoch: int = None,
) -> _LRScheduler:
    """Creates a learning rate scheduler with linear warmup.
    
    Combines a linear warmup phase with any of the supported schedulers.
    """
    warmup_steps = parse_schedule_to_steps(scheduler_config.get('warmup_steps', 0), steps_per_epoch)
    warmup_start_factor = scheduler_config.get('warmup_start_factor', 0.1)
    scheduler_type = scheduler_config.get('type', 'constant')
    
    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < warmup_steps:
            return warmup_start_factor + (1.0 - warmup_start_factor) * (current_step / warmup_steps)
        
        # Post-warmup phase
        progress = (current_step - warmup_steps) / max(1, num_training_steps - warmup_steps)
        
        if scheduler_type == 'constant':
            return 1.0
        
        elif scheduler_type == 'linear':
            end_factor = scheduler_config.get('end_factor', 0.0)
            return 1.0 - progress * (1.0 - end_factor)
        
        elif scheduler_type == 'cosine':
            eta_min_ratio = scheduler_config.get('eta_min', 0.0) / optimizer.param_groups[0]['lr']
            return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (1 + math.cos(math.pi * progress))
        
        elif scheduler_type == 'polynomial':
            power = scheduler_config.get('power', 1.0)
            end_factor = scheduler_config.get('end_factor', 0.0)
            return (1.0 - progress) ** power * (1.0 - end_factor) + end_factor
        
        elif scheduler_type == 'exponential':
            gamma = scheduler_config.get('gamma', 0.95)
            steps_after_warmup = current_step - warmup_steps
            return gamma ** steps_after_warmup
        
        else:
            return 1.0
    
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


@dataclass
class ScheduleSpec:
    """Represents a parsed schedule specification."""
    value: int
    unit: str  # 'steps' or 'epochs'
    
    def __str__(self):
        return f"{self.value}{self.unit[0]}"  # e.g., "1000s" or "5e"


def parse_schedule_value(value: Union[str, int, None]) -> Optional[ScheduleSpec]:
    """Parse a schedule value with optional suffix.
    
    Args:
        value: Can be:
            - int: Interpreted as steps (legacy behavior)
            - str with suffix: "1000s" (steps), "5e" (epochs), "2000steps", "10epochs"
            - None: Returns None
    
    Returns:
        ScheduleSpec with parsed value and unit, or None
        
    Examples:
        parse_schedule_value("1000s") -> ScheduleSpec(1000, "steps")
        parse_schedule_value("5e") -> ScheduleSpec(5, "epochs")
        parse_schedule_value("2000steps") -> ScheduleSpec(2000, "steps")
        parse_schedule_value("10epochs") -> ScheduleSpec(10, "epochs")
        parse_schedule_value(1000) -> ScheduleSpec(1000, "steps")  # legacy
    """
    if value is None:
        return None
    
    if isinstance(value, int):
        # Legacy: bare integers are interpreted as steps
        return ScheduleSpec(value, "steps")
    
    if isinstance(value, str):
        # Try to match patterns like "1000s", "5e", "2000steps", "10epochs"
        patterns = [
            (r'^(\d+)s$', 'steps'),           # "1000s"
            (r'^(\d+)e$', 'epochs'),          # "5e"
            (r'^(\d+)steps?$', 'steps'),      # "1000step" or "1000steps"
            (r'^(\d+)epochs?$', 'epochs'),    # "5epoch" or "5epochs"
        ]
        
        for pattern, unit in patterns:
            match = re.match(pattern, value.lower())
            if match:
                return ScheduleSpec(int(match.group(1)), unit)
        
        # Try parsing as plain integer (legacy string format)
        try:
            return ScheduleSpec(int(value), "steps")
        except ValueError:
            pass
    
    raise ValueError(f"Invalid schedule value format: {value}. "
                     f"Expected formats: '1000s', '5e', '2000steps', '10epochs', or integer.")


def spec_to_steps(spec: ScheduleSpec, steps_per_epoch: int = None) -> int:
    """Convert a ScheduleSpec to steps, handling epoch conversions."""
    if spec.unit == "epochs" and steps_per_epoch:
        return spec.value * steps_per_epoch
    return spec.value


def parse_schedule_to_steps(value: Union[str, int], steps_per_epoch: int = None) -> int:
    """Parse a schedule value (string or int) and convert to steps."""
    if isinstance(value, str):
        if value == "-1":
            return -1
        spec = parse_schedule_value(value)
        return spec_to_steps(spec, steps_per_epoch)
    return int(value)


def convert_schedule_strings_in_config(value: Any, current_epoch: int = 0, steps_per_epoch: int = None) -> Any:
    """Convert schedule strings in a config value to concrete numbers.
    
    This handles strings with s/e notation in config values, converting them
    to actual step counts based on current training state.
    """
    # List of string values that should NOT be parsed as schedule values
    RESERVED_STRINGS = {"-1", "constant", "linear", "cosine", "cosine_with_restarts", 
                        "polynomial", "exponential", "linear_decay", "linear_warmup", 
                        "cosine_anneal", "exponential_decay", "linear_decay_after_constant"}
    
    if isinstance(value, str) and value not in RESERVED_STRINGS:
        # Try to parse as schedule value
        try:
            spec = parse_schedule_value(value)
            return spec_to_steps(spec, steps_per_epoch)
        except ValueError:
            # Not a schedule value, return as-is
            return value
    elif isinstance(value, dict):
        return {k: convert_schedule_strings_in_config(v, current_epoch, steps_per_epoch) 
                for k, v in value.items()}
    elif isinstance(value, list):
        return [convert_schedule_strings_in_config(v, current_epoch, steps_per_epoch) 
                for v in value]
    else:
        return value


def parse_schedule_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a config dict and convert suffixed schedule values.
    
    This function walks through the config and converts any values that look like
    schedule specifications into detailed format.
    
    Args:
        config: Configuration dictionary that may contain suffixed values
        
    Returns:
        New configuration dictionary with parsed schedule specifications
    """
    def parse_recursive(obj):
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if key.endswith('_at_step') or key.endswith('_at_epoch') or key.endswith('_steps') or key.endswith('_epochs'):
                    # Skip keys that are already explicitly step/epoch specific
                    result[key] = value
                elif key in ['unfreeze_at', 'freeze_at', 'warmup_steps', 'warmup_epochs', 'interval']:
                    # Parse scheduling keys
                    try:
                        parsed = parse_schedule_value(value)
                        if parsed:
                            # Store both the parsed spec and the original value for compatibility
                            result[key] = value  # Keep original
                            result[f"{key}_parsed"] = {
                                'value': parsed.value,
                                'unit': parsed.unit
                            }
                        else:
                            result[key] = value
                    except ValueError as e:
                        print(f"Warning: Failed to parse schedule value for {key}: {e}")
                        result[key] = value
                else:
                    result[key] = parse_recursive(value)
            return result
        elif isinstance(obj, list):
            return [parse_recursive(item) for item in obj]
        else:
            return obj
    
    return parse_recursive(config)


def resolve_schedule_at_step(spec: Optional[ScheduleSpec], current_step: int, current_epoch: int) -> bool:
    """Check if a schedule spec should trigger at the current step/epoch.
    
    Args:
        spec: Parsed schedule specification (can be None)
        current_step: Current training step
        current_epoch: Current training epoch
        
    Returns:
        True if the schedule should trigger now
    """
    if spec is None:
        return False
    
    if spec.unit == "steps":
        return current_step >= spec.value
    elif spec.unit == "epochs":
        return current_epoch >= spec.value
    else:
        raise ValueError(f"Unknown schedule unit: {spec.unit}")


def get_schedule_value_for_logging(spec: Optional[ScheduleSpec]) -> str:
    """Get a human-readable string for logging purposes."""
    if spec is None:
        return "never"
    return f"{spec.value} {spec.unit}"


def get_autocast_context(device: torch.device):
    """Get the appropriate autocast context for the given device.
    
    This function returns a context manager that handles automatic mixed precision
    casting based on the device type. It avoids code duplication by centralizing
    the logic for determining the appropriate dtype and context.
    
    Args:
        device: The torch device to use for autocast
        
    Returns:
        A context manager for autocast (nullcontext if not CUDA)
        
    Example:
        with get_autocast_context(device):
            # Your forward pass code here
            output = model(input)
    """
    if device.type == "cuda":
        # Use bfloat16 if supported (better for training stability), otherwise float16
        preferred_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        return autocast(device_type="cuda", dtype=preferred_dtype)
    else:
        # No autocast for CPU or other devices
        return nullcontext()


def apply_gradient_scaling(loss, optimizer, scaler, trainable_params, grad_clip, device):
    """Apply gradient scaling and clipping in a device-agnostic way.
    
    This function handles the gradient computation, scaling, and clipping
    for both CUDA (with AMP scaler) and CPU devices, avoiding code duplication.
    
    Args:
        loss: The loss tensor to backpropagate
        optimizer: The optimizer instance
        scaler: GradScaler instance (used only for CUDA)
        trainable_params: Parameters to clip gradients for
        grad_clip: Maximum gradient norm
        device: The torch device
        
    Returns:
        tuple: (grad_norm, param_before) where param_before is a list of parameter clones
    """
    if device.type == "cuda":
        # Use gradient scaler for mixed precision training
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        param_before = [p.detach().clone() for p in trainable_params]
    else:
        # Standard backward pass for CPU
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        param_before = [p.detach().clone() for p in trainable_params]
    
    return grad_norm, param_before


def optimizer_step(optimizer, scaler, device):
    """Perform optimizer step in a device-agnostic way.
    
    Args:
        optimizer: The optimizer instance
        scaler: GradScaler instance (used only for CUDA)
        device: The torch device
    """
    if device.type == "cuda":
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step() 