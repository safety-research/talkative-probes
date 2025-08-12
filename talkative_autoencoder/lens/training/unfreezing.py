from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from .optim import param_groups
from .schedules import get_lr_scheduler, parse_schedule_to_steps


def should_unfreeze_encoder_now(
    step: int,
    current_epoch: int,
    config: Dict[str, Any],
    steps_per_epoch: int,
    grad_accum_steps: int,
) -> bool:
    unfreeze_cfg = config.get("unfreeze_encoder", {}) or {}
    if not unfreeze_cfg.get("enabled", False):
        return False

    # Support either 'step' (int or schedule string) or 'at' (schedule string)
    trigger_spec = unfreeze_cfg.get("at", unfreeze_cfg.get("step", -1))
    trigger_steps = parse_schedule_to_steps(trigger_spec, steps_per_epoch, grad_accum_steps)
    if trigger_steps < 0:
        return False
    return step >= trigger_steps


def unfreeze_encoder_and_rebuild_optim(
    *,
    step: int,
    config: Dict[str, Any],
    decoder,
    encoder,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    gradient_accumulation_steps: int,
    max_optimizer_steps: int,
    learning_rate: float,
    projection_lr_multiplier: float,
    embedding_lr_multiplier: float,
    prompt_lr_multiplier: float,
    base_model_lr_multiplier: float,
    overall_encoder_lr_multiplier: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
) -> Tuple[torch.optim.Optimizer, Any]:
    decoder_base = decoder.module if hasattr(decoder, "module") else decoder
    encoder_base = encoder.module if hasattr(encoder, "module") else encoder

    for p in encoder_base.parameters():
        p.requires_grad = True

    old_state = optimizer.state_dict().get("state", {})

    new_param_groups = param_groups(
        [decoder_base, encoder_base],
        learning_rate,
        projection_lr_multiplier,
        embedding_lr_multiplier,
        prompt_lr_multiplier,
        base_model_lr_multiplier,
        overall_encoder_lr_multiplier,
        weight_decay,
    )

    new_optimizer = torch.optim.AdamW(new_param_groups, betas=(beta1, beta2))
    # Load only the internal state; keep the new param_groups structure
    new_optimizer.load_state_dict(
        {
            "state": old_state,
            "param_groups": new_optimizer.state_dict()["param_groups"],
        }
    )

    current_optimizer_step = step // max(1, gradient_accumulation_steps)
    last_epoch = current_optimizer_step - 1 if current_optimizer_step > 0 else -1
    new_scheduler = get_lr_scheduler(
        new_optimizer,
        config["lr_scheduler"],
        max_optimizer_steps,
        last_epoch=last_epoch,
        grad_accum_steps=gradient_accumulation_steps,
    )

    return new_optimizer, new_scheduler
