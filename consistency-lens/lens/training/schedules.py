from __future__ import annotations

import math
from typing import Dict, Literal, Optional
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, ExponentialLR, PolynomialLR

ScheduleType = Literal["constant", "linear_decay", "linear_warmup", "cosine_anneal", "exponential_decay"]
LRSchedulerType = Literal["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "exponential"]

__all__ = ["get_schedule_value", "get_lr_scheduler", "get_lr_scheduler_with_warmup"]


def get_schedule_value(
    schedule_config: Dict,
    current_step: int,
    total_steps: int | None = None, # Required for some schedules like cosine_anneal
) -> float:
    """Calculates the current value for a hyperparameter based on its schedule.

    Args:
        schedule_config: Dictionary defining the schedule.
            Expected keys: 'type', and type-specific keys like 'value',
            'start_value', 'end_value', 'num_steps'.
        current_step: The current training step.
        total_steps: The total number of training steps, required for schedules
            that depend on the full training duration (e.g., cosine_anneal).

    Returns:
        The calculated value for the hyperparameter at the current_step.
    """
    schedule_type: ScheduleType = schedule_config["type"]

    if schedule_type == "constant":
        return float(schedule_config["value"])

    start_value = float(schedule_config["start_value"])
    end_value = float(schedule_config["end_value"])
    num_schedule_steps = int(schedule_config.get("num_steps", total_steps if total_steps is not None else current_step +1 ))

    if num_schedule_steps == 0: # Avoid division by zero if num_steps is 0 or not properly set
        return end_value
        
    progress = min(1.0, current_step / num_schedule_steps)

    if schedule_type == "linear_decay":
        return start_value - progress * (start_value - end_value)
    if schedule_type == "linear_decay_after_constant":
        constant_steps = schedule_config.get("constant_steps_before_linear_decay", 0)
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
) -> Optional[_LRScheduler]:
    """Creates a learning rate scheduler based on configuration.
    
    Args:
        optimizer: The optimizer to schedule
        scheduler_config: Configuration dictionary with 'type' and scheduler-specific params
        num_training_steps: Total number of training steps
        last_epoch: The index of the last epoch (default: -1)
        
    Returns:
        A PyTorch learning rate scheduler or None if type is 'constant'
    """
    scheduler_type = scheduler_config.get('type', 'constant')
    
    if scheduler_type == 'constant':
        return None
    
    # Handle warmup if specified
    warmup_steps = scheduler_config.get('warmup_steps', 0)
    if warmup_steps > 0:
        return get_lr_scheduler_with_warmup(
            optimizer, scheduler_config, num_training_steps, last_epoch
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
        T_0 = scheduler_config.get('T_0', 500)
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
) -> _LRScheduler:
    """Creates a learning rate scheduler with linear warmup.
    
    Combines a linear warmup phase with any of the supported schedulers.
    """
    warmup_steps = scheduler_config.get('warmup_steps', 0)
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