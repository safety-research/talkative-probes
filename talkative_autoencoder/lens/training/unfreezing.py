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

    # Flip lr from 0 -> computed lr for encoder groups whose params are now trainable
    # Be robust to wrappers by unifying ids from encoder, encoder.module, and encoder._orig_mod
    encoder_param_ids = set()
    for m in (encoder_base, getattr(encoder, "module", None), getattr(encoder, "_orig_mod", None)):
        if m is None:
            continue
        try:
            for p in m.parameters():
                encoder_param_ids.add(id(p))
        except Exception:
            pass
    # Ensure initial_lr exists for all groups (encoder and decoder)
    for g in optimizer.param_groups:
        if "initial_lr" not in g:
            # Compute from category to avoid inheriting a 0 lr
            cat = g.get("category", "other")
            if cat == "proj":
                lr_mult_cat = projection_lr_multiplier
            elif cat == "embedding":
                lr_mult_cat = embedding_lr_multiplier
            elif cat == "prompt":
                lr_mult_cat = prompt_lr_multiplier
            elif cat == "base":
                lr_mult_cat = base_model_lr_multiplier
            else:
                lr_mult_cat = 1.0
            # If this is an encoder group, apply overall encoder multiplier
            is_encoder_group = any(id(p) in encoder_param_ids for p in g.get("params", []))
            overall_mult = overall_encoder_lr_multiplier if is_encoder_group else 1.0
            g["initial_lr"] = learning_rate * (overall_mult * lr_mult_cat)
    for g in optimizer.param_groups:
        group_param_ids = {id(p) for p in g.get("params", [])}
        is_encoder_group = any(pid in encoder_param_ids for pid in group_param_ids)
        if not is_encoder_group:
            continue
        if not any(p.requires_grad for p in g.get("params", [])):
            continue
        # Compute target LR deterministically from category and config multipliers
        cat = g.get("category", "other")
        if cat == "proj":
            lr_mult = projection_lr_multiplier
        elif cat == "embedding":
            lr_mult = embedding_lr_multiplier
        elif cat == "prompt":
            lr_mult = prompt_lr_multiplier
        elif cat == "base":
            lr_mult = base_model_lr_multiplier
        else:
            lr_mult = 1.0
        target_lr = learning_rate * (overall_encoder_lr_multiplier * lr_mult)
        g["lr"] = target_lr
        g["initial_lr"] = target_lr

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
