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

    # Selectively unfreeze only the parameters that were planned to be trainable
    planned_names = getattr(encoder_base, "_planned_trainable_param_names", None)
    if planned_names is None:
        planned_names = getattr(encoder, "_planned_trainable_param_names", [])
    planned_names = set(planned_names)

    for name, p in encoder_base.named_parameters():
        p.requires_grad = name in planned_names

    old_state = optimizer.state_dict().get("state", {})

    # Order models so that shared parameters deduplicate consistently;
    # place encoder first so its overall multiplier applies where relevant.
    new_param_groups = param_groups(
        [encoder_base, decoder_base],
        learning_rate,
        projection_lr_multiplier,
        embedding_lr_multiplier,
        prompt_lr_multiplier,
        base_model_lr_multiplier,
        overall_encoder_lr_multiplier,
        weight_decay,
    )

    new_optimizer = torch.optim.AdamW(new_param_groups, betas=(beta1, beta2))
    # Ensure initial_lr exists for schedulers that expect it
    for group in new_optimizer.param_groups:
        group["initial_lr"] = group.get("initial_lr", group["lr"])  # set if missing

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
