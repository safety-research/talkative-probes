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
import logging
from lens.training.optim import param_groups
from lens.utils.logging import init as log_init, log as log_metrics
import os

ScheduleType = Literal["constant", "linear_decay", "linear_warmup", "cosine_anneal", "exponential_decay"]
LRSchedulerType = Literal["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "exponential", "resume_with_transition"]

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
    "parse_schedule_to_steps",
    "SmoothTransitionScheduler"
]


class SmoothTransitionScheduler(_LRScheduler):
    """LR Scheduler that smoothly transitions from checkpoint LR to target scheduler LR.
    
    This is useful when resuming from a checkpoint where the learning rate might be
    different from what the scheduler would normally produce at that step.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        base_scheduler: _LRScheduler,
        transition_steps: int,
        start_lr: float = None, # The LR to start the transition from
        last_epoch: int = -1    # Should be absolute_current_optimizer_step - 1 when resuming
    ):
        """Initialize the smooth transition scheduler.
        
        Args:
            optimizer: The optimizer to schedule
            base_scheduler: The underlying scheduler to transition to
            transition_steps: Number of optimizer steps over which to transition
            start_lr: Starting LR for the transition (e.g., LR from checkpoint)
            last_epoch: The index of the last optimizer step completed before this scheduler starts.
                        Typically absolute_current_optimizer_step - 1.
        """
        self.base_scheduler = base_scheduler
        # Ensure transition_steps is at least 1 to avoid division by zero and ensure a discrete step.
        self.transition_steps = max(1, int(transition_steps))
        
        # Store the starting LRs for each param group, based on the provided start_lr
        if start_lr is not None:
            # Scale start_lr proportionally for all parameter groups
            # based on the optimizer's current LRs (which should be new_base_lr from main script)
            # Handle cases where the leading learning rate might be zero
            first_pg_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 1.0
            if first_pg_lr == 0: # Avoid division by zero if base LR is 0
                 self.start_lrs = [start_lr for _ in optimizer.param_groups]
            else:
                self.start_lrs = [start_lr * (pg['lr'] / first_pg_lr)
                                  for pg in optimizer.param_groups]
        else:
            # If start_lr is not given, use current LRs in optimizer as starting points
            self.start_lrs = [pg['lr'] for pg in optimizer.param_groups]
        
        # Crucially, synchronize base_scheduler's last_epoch with ours *after* super().__init__
        # This ensures they are aligned from the very beginning of this scheduler's life.
        # self.last_epoch is now set by super().__init__ to the passed 'last_epoch' argument.
        self._sts_transition_log_count = 0 # Initialize log counter
        super().__init__(optimizer, last_epoch) # Initializes self.last_epoch and self._step_count
        self.base_scheduler.last_epoch = self.last_epoch

    
    def get_lr(self):
        """Calculate learning rates with smooth transition."""
        # self._step_count is the number of times step() has been called.
        # It's 1 after __init__ completes (due to internal step()), 
        # 2 after the first explicit step() call from training loop, and so on.
        current_scheduler_call_count = self._step_count 

        # Get a logger instance
        logger = logging.getLogger(__name__)
        is_main_proc = os.environ.get('RANK') is None or os.environ.get('RANK') == '0'

        # step_in_transition is 0-indexed for how many steps we are *into* the transition period.
        # If _step_count = 1 (first get_lr call after init), step_in_transition = 0.
        step_in_transition = current_scheduler_call_count - 1
        
        if step_in_transition < 0: # Should not happen if _step_count starts at 1
            if is_main_proc and logger.isEnabledFor(logging.WARNING):
                logger.warning(f"SmoothTransitionScheduler (call count {current_scheduler_call_count}): step_in_transition is negative ({step_in_transition}). Using start_lrs: {self.start_lrs}")
            return self.start_lrs

        if step_in_transition >= self.transition_steps:
            # Transition is complete.
            final_lrs = self.base_scheduler.get_lr()
            if is_main_proc and logger.isEnabledFor(logging.DEBUG) and (step_in_transition == self.transition_steps or (step_in_transition - self.transition_steps) % 100 == 0) : # Log first time and periodically after
                logger.debug(f"SmoothTransitionScheduler (call count {current_scheduler_call_count}, step_in_trans {step_in_transition}): Transition complete. Using base_scheduler LRs: {final_lrs}")
            return final_lrs
        
        # During transition:
        # transition_progress will go from 0 up to (transition_steps-1)/transition_steps
        transition_progress = step_in_transition / self.transition_steps
        
        # The target LRs for the interpolation are what the base_scheduler would provide
        # for the current_scheduler_call_count. Since they are synchronized, base_scheduler.get_lr() is correct.
        target_lrs_for_current_step = self.base_scheduler.get_lr()
        
        interpolated_lrs = []
        for i in range(len(self.start_lrs)):
            # Interpolate from the fixed self.start_lrs
            # towards the target LR for the current step from the base_scheduler.
            s_lr = self.start_lrs[i]
            t_lr = target_lrs_for_current_step[i]
            lr = s_lr + transition_progress * (t_lr - s_lr)
            interpolated_lrs.append(lr)
        
        if is_main_proc and self._sts_transition_log_count < 5:
            logger.info(f"SmoothTransitionScheduler (call count {current_scheduler_call_count}): Transitioning.")
            logger.info(f"  transition_steps_optimizer: {self.transition_steps}")
            logger.info(f"  step_in_transition_optimizer (0-indexed): {step_in_transition}")
            logger.info(f"  transition_progress: {transition_progress:.4f}")
            logger.info(f"  start_lrs: {[f'{slr:.3e}' for slr in self.start_lrs]}")
            logger.info(f"  target_lrs_current_step (from base): {[f'{tlr:.3e}' for tlr in target_lrs_for_current_step]}")
            logger.info(f"  interpolated_lrs: {[f'{ilr:.3e}' for ilr in interpolated_lrs]}")
            self._sts_transition_log_count += 1

        return interpolated_lrs
    
    def step(self, epoch=None):
        """Step the scheduler."""
        # Increment self.last_epoch (and _step_count via super)
        # Note: PyTorch schedulers typically update optimizer LRs within their step()
        # by calling self.get_lr() and then setting param_group['lr'].
        # Our super().step() will do this using the LRs from self.get_lr().
        super().step(epoch)
        
        # Also step the base scheduler to keep it in sync.
        # This ensures its internal state (e.g., last_epoch) also advances.
        # Its LRs will be updated in the optimizer by its own step() if it were used alone,
        # but here, self.get_lr() is what dictates the optimizer's LRs.
        # We just need its state advanced.
        self.base_scheduler.step(epoch) # Keep base_scheduler's state (last_epoch) aligned
    
    def state_dict(self):
        """Returns the state of the scheduler as a dict."""
        state = super().state_dict()
        state['base_scheduler'] = self.base_scheduler.state_dict()
        state['transition_steps'] = self.transition_steps
        state['start_lrs'] = self.start_lrs
        return state
    
    def load_state_dict(self, state_dict):
        """Loads the schedulers state."""
        base_state = state_dict.pop('base_scheduler')
        self.transition_steps = state_dict.pop('transition_steps')
        self.start_lrs = state_dict.pop('start_lrs')
        
        super().load_state_dict(state_dict)
        self.base_scheduler.load_state_dict(base_state)


def get_schedule_value(
    schedule_config: Dict,
    current_step: int,
    total_steps: int | None = None, # Required for some schedules like cosine_anneal
    current_epoch: int | None = None,  # For epoch-based schedules
    steps_per_epoch: int | None = None,  # For converting epochs to steps
    grad_accum_steps: int | None = None,  # For converting optimizer steps to steps
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
    if (isinstance(num_steps_raw, str) and num_steps_raw == "-1") or num_steps_raw == -1:
        if total_steps is None:
            raise ValueError("total_steps is required when num_steps is -1")
        num_schedule_steps = total_steps
    else:
        num_schedule_steps = parse_schedule_to_steps(num_steps_raw, steps_per_epoch, grad_accum_steps)
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
        constant_steps_raw = schedule_config["constant_steps_before_decay"]
        constant_steps = parse_schedule_to_steps(constant_steps_raw, steps_per_epoch, grad_accum_steps)
        
        if current_step < constant_steps:
            return start_value
        elif current_step <= num_schedule_steps+constant_steps:
            return start_value + (current_step - constant_steps) * (end_value - start_value) / (num_schedule_steps)
        else:
            return end_value
    if schedule_type == "exponential_decay_after_constant":
        constant_steps_raw = schedule_config["constant_steps_before_decay"]
        constant_steps = parse_schedule_to_steps(constant_steps_raw, steps_per_epoch, grad_accum_steps)
        if current_step < constant_steps:
            return start_value
        else:
            decay_steps = num_schedule_steps - constant_steps
            if decay_steps <= 0:
                raise ValueError("Decay steps must be > 0 for exponential_decay_after_constant.")
            decay_progress = min(1.0, (current_step - constant_steps) / decay_steps)
            return start_value * (end_value / start_value) ** decay_progress

    if schedule_type == "cosine_decay_after_constant":
        constant_steps_raw = schedule_config["constant_steps_before_decay"]
        constant_steps = parse_schedule_to_steps(constant_steps_raw, steps_per_epoch, grad_accum_steps)
        if current_step < constant_steps:
            return start_value
        else:
            decay_steps = num_schedule_steps - constant_steps
            if decay_steps <= 0:
                raise ValueError("Decay steps must be > 0 for cosine_decay_after_constant.")
            decay_progress = min(1.0, (current_step - constant_steps) / decay_steps)
            # Cosine decay: value = end_value + 0.5 * (start_value - end_value) * (1 + cos(pi * decay_progress))
            return end_value + 0.5 * (start_value - end_value) * (1 + math.cos(math.pi * decay_progress))
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

    elif schedule_type == "cosine_anneal_after_linear_decay":
        linear_decay_steps_raw = schedule_config.get("linear_decay_steps")
        initial_high_lr_raw = schedule_config.get("initial_high_lr")
        linear_decay_steps = parse_schedule_to_steps(linear_decay_steps_raw, steps_per_epoch, grad_accum_steps)
        initial_high_lr = float(initial_high_lr_raw)
        if current_step < linear_decay_steps:
            return initial_high_lr - progress * (initial_high_lr - start_value)
        else:
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
    grad_accum_steps: int = None,
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
    warmup_steps = parse_schedule_to_steps(warmup_steps_raw, steps_per_epoch, grad_accum_steps)
    
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
        T_0 = parse_schedule_to_steps(scheduler_config.get('T_0', 500), steps_per_epoch, grad_accum_steps)
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
    grad_accum_steps: int = None,
) -> _LRScheduler:
    """Creates a learning rate scheduler with linear warmup.
    
    Combines a linear warmup phase with any of the supported schedulers.
    """
    warmup_steps = parse_schedule_to_steps(scheduler_config.get('warmup_steps', 0), steps_per_epoch, grad_accum_steps)
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
            return get_schedule_value(scheduler_config, current_step, num_training_steps, current_epoch, steps_per_epoch)
    
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
            (r'^(\d+)os$', 'optimizer_steps'),          # "1000os"
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
                     f"Expected formats: '1000s', '5e', '2000steps', '10epochs', '1000os', or integer.")


def spec_to_steps(spec: ScheduleSpec, steps_per_epoch: int = None, grad_accum_steps: int = None) -> int:
    """Convert a ScheduleSpec to steps, handling epoch conversions."""
    if spec.unit == "epochs" and steps_per_epoch:
        return spec.value * steps_per_epoch
    elif spec.unit == "optimizer_steps" and grad_accum_steps:
        return spec.value * grad_accum_steps
    return spec.value


def parse_schedule_to_steps(value: Union[str, int], steps_per_epoch: int = None, grad_accum_steps: int = None) -> int:
    """Parse a schedule value (string or int) and convert to steps."""
    if isinstance(value, str):
        if value == "-1":
            return -1
        spec = parse_schedule_value(value)
        return spec_to_steps(spec, steps_per_epoch, grad_accum_steps)
    return int(value)


def convert_schedule_strings_in_config(value: Any, current_epoch: int = 0, steps_per_epoch: int = None, grad_accum_steps: int = None) -> Any:
    """Convert schedule strings in a config value to concrete numbers.
    
    This handles strings with s/e notation in config values, converting them
    to actual step counts based on current training state.
    """
    # List of string values that should NOT be parsed as schedule values
    RESERVED_STRINGS = {"-1", "constant", "linear", "cosine", "cosine_with_restarts", 
                        "polynomial", "exponential", "linear_decay", "linear_warmup", 
                        "cosine_anneal", "exponential_decay", "linear_decay_after_constant",
                        "cosine_anneal_after_linear_decay"}
    
    if isinstance(value, str) and value not in RESERVED_STRINGS:
        # Try to parse as schedule value
        try:
            spec = parse_schedule_value(value)
            return spec_to_steps(spec, steps_per_epoch, grad_accum_steps)
        except ValueError:
            # Not a schedule value, return as-is
            return value
    elif isinstance(value, dict):
        return {k: convert_schedule_strings_in_config(v, current_epoch, steps_per_epoch, grad_accum_steps) 
                for k, v in value.items()}
    elif isinstance(value, list):
        return [convert_schedule_strings_in_config(v, current_epoch, steps_per_epoch, grad_accum_steps) 
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


def resolve_schedule_at_step(spec: Optional[ScheduleSpec], current_step: int, current_epoch: int, grad_accum_steps: int) -> bool:
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
    elif spec.unit == "optimizer_steps":
        return current_step*grad_accum_steps >= spec.value
    else:
        raise ValueError(f"Unknown schedule unit: {spec.unit}")


def get_schedule_value_for_logging(spec: Optional[ScheduleSpec]) -> str:
    """Get a human-readable string for logging purposes."""
    if spec is None:
        return "never"
    return f"{spec.value} {spec.unit}"


def get_autocast_context(device: torch.device, mixed_precision_config: Dict[str, Any] = None):
    """Get the appropriate autocast context for the given device and mixed precision config.
    
    This function returns a context manager that handles automatic mixed precision
    casting based on the device type and configuration. It avoids code duplication by centralizing
    the logic for determining the appropriate dtype and context.
    
    Args:
        device: The torch device to use for autocast
        mixed_precision_config: Dictionary with 'enabled' and 'dtype' keys
            dtype can be: "auto", "float16", "bfloat16", "float32"
        
    Returns:
        A context manager for autocast (nullcontext if not enabled or not CUDA)
        
    Example:
        with get_autocast_context(device, config.get('mixed_precision')):
            # Your forward pass code here
            output = model(input)
    """
    # Default config if none provided
    if mixed_precision_config is None:
        mixed_precision_config = {'enabled': True, 'dtype': 'auto'}
    
    # Check if mixed precision is enabled
    if not mixed_precision_config.get('enabled', True):
        return nullcontext()
    
    # Only use autocast on CUDA devices
    if device.type != "cuda":
        return nullcontext()
    
    # Determine dtype from config
    dtype_str = mixed_precision_config.get('dtype', 'auto')
    
    if dtype_str == 'auto':
        # Auto mode: use bfloat16 if available, otherwise float32 (safest)
        preferred_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    elif dtype_str == 'float16':
        preferred_dtype = torch.float16
    elif dtype_str == 'bfloat16':
        preferred_dtype = torch.bfloat16
    elif dtype_str == 'float32':
        # No autocast for float32 - return nullcontext
        return nullcontext()
    else:
        raise ValueError(f"Unknown mixed precision dtype: {dtype_str}")
    
    return autocast(device_type="cuda", dtype=preferred_dtype)


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

def should_unfreeze_any_component(current_step, current_epoch, freeze_schedule_config, freeze_schedule_enabled, grad_accum_steps):
    """Check if any component should be unfrozen at current step/epoch."""
    if not freeze_schedule_enabled:
        return False
    
    components_config = freeze_schedule_config.get('components', {})
    
    # Check each component individually
    for component_name, component_cfg in components_config.items():
        for param_name, param_cfg in component_cfg.items():
            if isinstance(param_cfg, dict) and 'unfreeze_at' in param_cfg:
                unfreeze_spec_str = param_cfg['unfreeze_at']
                if unfreeze_spec_str is not None:
                    try:
                        unfreeze_spec = parse_schedule_value(unfreeze_spec_str)
                        if resolve_schedule_at_step(unfreeze_spec, current_step, current_epoch, grad_accum_steps):
                            return True
                    except Exception:
                        pass
    
    # Check global unfreeze timing
    global_unfreeze_at = freeze_schedule_config.get('unfreeze_at')
    if global_unfreeze_at is not None:
        try:
            unfreeze_spec = parse_schedule_value(global_unfreeze_at)
            return resolve_schedule_at_step(unfreeze_spec, current_step, current_epoch, grad_accum_steps)
        except Exception:
            pass
    
    # Legacy compatibility
    legacy_step = freeze_schedule_config.get('unfreeze_at_step')
    if legacy_step is not None and current_step >= legacy_step:
        return True
        
    legacy_epoch = freeze_schedule_config.get('unfreeze_at_epoch')
    if legacy_epoch is not None and current_epoch >= legacy_epoch:
        return True
        
    return False


def unfreeze_non_adapters(dec_raw, enc_raw, config, learning_rate, projection_lr_multiplier, embedding_lr_multiplier, prompt_lr_multiplier, base_model_lr_multiplier, overall_encoder_lr_multiplier, opt_state_dict=None, current_step=None, current_epoch=None, grad_accum_steps=None):
    """Unfreeze non-adapter parameters and create new optimizer with all parameters."""
    log = logging.getLogger(__name__)
    
    # Get original trainable settings from config
    decoder_train_cfg = config.get('trainable_components', {}).get('decoder', {})
    encoder_train_cfg = config.get('trainable_components', {}).get('encoder', {})
    
    # Check for freeze schedule overrides
    freeze_schedule = config.get('freeze_schedule', {})
    components_config = freeze_schedule.get('components', {})
    
    # Helper function to get effective config (freeze schedule override or original config)
    def get_effective_config(component, param_name, original_config):
        component_cfg = components_config.get(component, {}).get(param_name, {})
        
        # Check if this component has custom enabled setting
        enabled_override = component_cfg.get('enabled') if isinstance(component_cfg, dict) else None
        if enabled_override is not None:
            return enabled_override
            
        # Use original config setting
        return original_config.get(param_name, False)
    
    # Helper function to check if a component should be unfrozen based on timing
    def should_unfreeze_component(component, param_name):
        component_cfg = components_config.get(component, {}).get(param_name, {})
        
        # Check for component-specific timing
        if isinstance(component_cfg, dict) and 'unfreeze_at' in component_cfg:
            unfreeze_spec_str = component_cfg['unfreeze_at']
            if unfreeze_spec_str is not None:
                try:
                    unfreeze_spec = parse_schedule_value(unfreeze_spec_str)
                    return resolve_schedule_at_step(unfreeze_spec, current_step or 0, current_epoch or 0, grad_accum_steps)
                except Exception as e:
                    print(f"Warning: Failed to parse unfreeze_at for {component}.{param_name}: {e}")
        
        # Fall back to global unfreeze timing
        global_unfreeze_at = freeze_schedule.get('unfreeze_at')
        if global_unfreeze_at is not None:
            try:
                unfreeze_spec = parse_schedule_value(global_unfreeze_at)
                return resolve_schedule_at_step(unfreeze_spec, current_step or 0, current_epoch or 0, grad_accum_steps)
            except Exception as e:
                print(f"Warning: Failed to parse global unfreeze_at: {e}")
        
        # Legacy compatibility
        if current_step is not None:
            legacy_step = freeze_schedule.get('unfreeze_at_step')
            if legacy_step is not None:
                return current_step >= legacy_step
        
        if current_epoch is not None:
            legacy_epoch = freeze_schedule.get('unfreeze_at_epoch')
            if legacy_epoch is not None:
                return current_epoch >= legacy_epoch
                
        return False
    
    # Track which parameters are newly unfrozen
    newly_unfrozen_params = set()
    
    # Helper function to unfreeze embedding heads
    def unfreeze_embedding_heads(model, should_unfreeze):
        if not should_unfreeze:
            return
        
        # Unfreeze input embeddings
        try:
            input_embeddings = model.get_input_embeddings()
            if input_embeddings is not None:
                for param in input_embeddings.parameters():
                    was_frozen = not param.requires_grad
                    param.requires_grad = True
                    if was_frozen:
                        newly_unfrozen_params.add(param)
        except AttributeError:
            pass
        
        # Unfreeze output embeddings
        try:
            output_embeddings = model.get_output_embeddings()
            if output_embeddings is not None:
                for param in output_embeddings.parameters():
                    was_frozen = not param.requires_grad
                    param.requires_grad = True
                    if was_frozen:
                        newly_unfrozen_params.add(param)
        except AttributeError:
            # Fallback for models that expose `lm_head`
            if hasattr(model, 'lm_head'):
                for param in model.lm_head.parameters():
                    was_frozen = not param.requires_grad
                    param.requires_grad = True
                    if was_frozen:
                        newly_unfrozen_params.add(param)
    
    # Unfreeze based on effective config and timing
    for name, param in dec_raw.named_parameters():
        if 'base' in name and '.embed' not in name and '.wte' not in name:  # Base model params (excluding embeddings)
            was_frozen = not param.requires_grad
            should_enable = get_effective_config('decoder', 'base_model', decoder_train_cfg)
            should_unfreeze_now = should_unfreeze_component('decoder', 'base_model')
            param.requires_grad = should_enable and should_unfreeze_now
            if was_frozen and param.requires_grad:
                newly_unfrozen_params.add(param)
        elif 'out' in name:  # Output head (self.out layer)
            was_frozen = not param.requires_grad
            should_enable = get_effective_config('decoder', 'output_head', decoder_train_cfg)
            should_unfreeze_now = should_unfreeze_component('decoder', 'output_head')
            param.requires_grad = should_enable and should_unfreeze_now
            if was_frozen and param.requires_grad:
                newly_unfrozen_params.add(param)
        elif 'prompt_left_emb' in name or 'prompt_right_emb' in name:  # Trainable prompts
            was_frozen = not param.requires_grad
            should_enable = get_effective_config('decoder', 'trainable_prompts', decoder_train_cfg)
            should_unfreeze_now = should_unfreeze_component('decoder', 'trainable_prompts')
            param.requires_grad = should_enable and should_unfreeze_now
            if was_frozen and param.requires_grad:
                newly_unfrozen_params.add(param)
        # proj layers remain as they were
    
    # Handle decoder embedding heads separately
    should_enable_dec_embeddings = get_effective_config('decoder', 'embedding_head', decoder_train_cfg)
    should_unfreeze_now_dec_embeddings = should_unfreeze_component('decoder', 'embedding_head')
    if should_enable_dec_embeddings and should_unfreeze_now_dec_embeddings:
        unfreeze_embedding_heads(dec_raw.base, True)
    
    for name, param in enc_raw.named_parameters():
        if 'base' in name and 'embed' not in name and 'wte' not in name:  # Base model params (excluding embeddings)
            was_frozen = not param.requires_grad
            should_enable = get_effective_config('encoder', 'base_model', encoder_train_cfg)
            should_unfreeze_now = should_unfreeze_component('encoder', 'base_model')
            param.requires_grad = should_enable and should_unfreeze_now
            if was_frozen and param.requires_grad:
                newly_unfrozen_params.add(param)
        elif 'soft_prompt_embeddings' in name:  # Soft prompt embeddings
            was_frozen = not param.requires_grad
            should_enable = get_effective_config('encoder', 'trainable_soft_prompt', encoder_train_cfg)
            should_unfreeze_now = should_unfreeze_component('encoder', 'trainable_soft_prompt')
            param.requires_grad = should_enable and should_unfreeze_now
            if was_frozen and param.requires_grad:
                newly_unfrozen_params.add(param)
        # proj layers remain as they were
    
    # Handle encoder embedding heads separately (only if using base model)
    if enc_raw.config.use_base_model:
        should_enable_enc_embeddings = get_effective_config('encoder', 'embedding_head', encoder_train_cfg)
        should_unfreeze_now_enc_embeddings = should_unfreeze_component('encoder', 'embedding_head')
        if should_enable_enc_embeddings and should_unfreeze_now_enc_embeddings:
            unfreeze_embedding_heads(enc_raw.base, True)
    
    # Update trainable params list
    dec = dec_raw if not isinstance(dec_raw, torch._dynamo.eval_frame.OptimizedModule) else dec_raw
    enc = enc_raw if not isinstance(enc_raw, torch._dynamo.eval_frame.OptimizedModule) else enc_raw
    
    trainable_params = [p for p in dec.parameters() if p.requires_grad] + \
                       [p for p in enc.parameters() if p.requires_grad]
    
    # Create new optimizer with updated parameters
    optimizer_groups = param_groups([dec, enc], learning_rate, projection_lr_multiplier, embedding_lr_multiplier, prompt_lr_multiplier, base_model_lr_multiplier, overall_encoder_lr_multiplier)
    new_opt = torch.optim.AdamW(optimizer_groups)
    
    # Set initial_lr for each parameter group (required for LR scheduler)
    for group in new_opt.param_groups:
        if 'initial_lr' not in group:
            group['initial_lr'] = group['lr']
    
    # Restore optimizer state if provided
    if opt_state_dict is not None:
        try:
            new_opt.load_state_dict(opt_state_dict)
            log.info("Restored optimizer state after unfreezing")
        except Exception as e:
            log.warning(f"Failed to restore optimizer state: {e}. Using fresh optimizer.")
    
    # Log parameter counts after unfreezing
    total_trainable = sum(p.numel() for p in trainable_params)
    newly_unfrozen_count = sum(p.numel() for p in newly_unfrozen_params)
    log.info(f"After unfreezing: {total_trainable:,} trainable parameters")
    log.info(f"Newly unfrozen: {newly_unfrozen_count:,} parameters")
    
    return new_opt, trainable_params, newly_unfrozen_params


def apply_unfreeze_warmup(opt, newly_unfrozen_params, unfreeze_transition_step, 
                         unfreeze_warmup_steps, step, freeze_schedule_config, 
                         log_interval, log):
    """Apply learning rate warmup for newly unfrozen parameters.
    
    Returns:
        tuple: (unfreeze_transition_step, should_clear_params) where should_clear_params
               indicates if newly_unfrozen_params should be cleared
    """
    if unfreeze_transition_step is None or not newly_unfrozen_params:
        return unfreeze_transition_step, False
        
    steps_since_unfreeze = step - unfreeze_transition_step
    
    if steps_since_unfreeze < unfreeze_warmup_steps:
        # Calculate warmup factor (linear warmup from start_factor to 1.0)
        warmup_start_factor = freeze_schedule_config.get('unfreeze_warmup_start_factor', 0.01)
        warmup_factor = warmup_start_factor + (1.0 - warmup_start_factor) * (steps_since_unfreeze / unfreeze_warmup_steps)
        
        # Apply warmup factor to newly unfrozen parameters
        for group_idx, group in enumerate(opt.param_groups):
            # Store the current LR as the base for warmup if not already stored
            if not hasattr(opt, '_warmup_base_lrs'):
                opt._warmup_base_lrs = {}
            if group_idx not in opt._warmup_base_lrs:
                opt._warmup_base_lrs[group_idx] = group['lr']
            
            # Apply warmup only to newly unfrozen params in this group
            group_has_unfrozen = False
            for param in group['params']:
                if param in newly_unfrozen_params:
                    group_has_unfrozen = True
                    break
            
            if group_has_unfrozen:
                # Apply warmup factor to entire group if it contains any newly unfrozen params
                group['lr'] = opt._warmup_base_lrs[group_idx] * warmup_factor
        
        # Log warmup progress
        if step % log_interval == 0:
            log.info(f"Unfreeze warmup: step {steps_since_unfreeze}/{unfreeze_warmup_steps}, factor {warmup_factor:.3f}")
        
        # Log metrics
        log_metrics({
            "freeze_schedule/warmup_factor": warmup_factor,
            "freeze_schedule/warmup_steps_remaining": unfreeze_warmup_steps - steps_since_unfreeze,
        }, step=step)
        
        return unfreeze_transition_step, False
        
    elif steps_since_unfreeze == unfreeze_warmup_steps:
        # Warmup complete, restore base learning rates
        if hasattr(opt, '_warmup_base_lrs'):
            for group_idx, group in enumerate(opt.param_groups):
                if group_idx in opt._warmup_base_lrs:
                    group['lr'] = opt._warmup_base_lrs[group_idx]
            delattr(opt, '_warmup_base_lrs')
        
        log.info("Unfreeze warmup complete - all parameters now at full learning rate")
        # Clear the newly unfrozen params set as warmup is done
        return None, True  # Clear unfreeze_transition_step and signal to clear params
    
    return unfreeze_transition_step, False

    