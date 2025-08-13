from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

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
    # Resolve base modules (handle DDP and torch.compile wrappers)
    if hasattr(decoder, "module"):
        decoder_base = decoder.module
    elif hasattr(decoder, "_orig_mod"):
        decoder_base = decoder._orig_mod
    else:
        decoder_base = decoder
    if hasattr(encoder, "module"):
        encoder_base = encoder.module
    elif hasattr(encoder, "_orig_mod"):
        encoder_base = encoder._orig_mod
    else:
        encoder_base = encoder

    # Selectively unfreeze only the parameters that were planned to be trainable
    planned_names = getattr(encoder_base, "_planned_trainable_param_names", None)
    if planned_names is None:
        planned_names = getattr(encoder, "_planned_trainable_param_names", [])
    planned_names = set(planned_names)
    # Debug-friendly print (kept minimal); consider gating under a verbose flag if needed
    # print(f"Unfreezing encoder with {len(planned_names)} planned-trainable parameters")

    for name, p in encoder_base.named_parameters():
        p.requires_grad = name in planned_names

    # We intentionally do NOT carry over the old optimizer "state" here.
    # Carrying state across a structural change in param_groups can mis-map
    # states to new parameters (PyTorch remaps by position), causing shape errors.

    # Flip lr from 0 -> scheduled base lr for encoder groups whose params are now trainable
    encoder_param_ids = {id(p) for p in encoder_base.parameters()}
    for g in optimizer.param_groups:
        group_param_ids = {id(p) for p in g.get("params", [])}
        is_encoder_group = any(pid in encoder_param_ids for pid in group_param_ids)
        if is_encoder_group:
            # If this encoder group is now trainable, restore its base lr from initial_lr
            if any(p.requires_grad for p in g.get("params", [])):
                g["lr"] = g.get("initial_lr", g.get("lr", learning_rate))
        # Ensure initial_lr is set (kept as the base lr for schedulers)
        g["initial_lr"] = g.get("initial_lr", g.get("lr", learning_rate))

    # Rebuild scheduler at the same step index
    current_optimizer_step = step // max(1, gradient_accumulation_steps)
    last_epoch = current_optimizer_step - 1 if current_optimizer_step > 0 else -1
    new_scheduler = get_lr_scheduler(
        optimizer,
        config["lr_scheduler"],
        max_optimizer_steps,
        last_epoch=last_epoch,
        grad_accum_steps=gradient_accumulation_steps,
    )

    return optimizer, new_scheduler
