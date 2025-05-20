from __future__ import annotations

import math
from typing import Dict, Literal

ScheduleType = Literal["constant", "linear_decay", "linear_warmup", "cosine_anneal", "exponential_decay"]

__all__ = ["get_schedule_value"]


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