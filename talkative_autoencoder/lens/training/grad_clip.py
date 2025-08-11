from typing import Dict, Iterable, List, Tuple

import torch

from .optim import get_param_category


def _compute_grad_norm(params: List[torch.nn.Parameter]) -> float:
    total_sq: float = 0.0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        total_sq += g.pow(2).sum().item()
    return float(total_sq ** 0.5)


def _iter_named_trainable_params(module, module_prefix: str):
    for name, param in module.named_parameters():
        if param.requires_grad:
            yield f"{module_prefix}.{name}", param


def build_param_buckets(
    decoder_base, encoder_base
) -> Dict[str, List[torch.nn.Parameter]]:
    prompt_params: List[torch.nn.Parameter] = []
    proj_params: List[torch.nn.Parameter] = []
    other_params: List[torch.nn.Parameter] = []

    named_params: List[Tuple[str, torch.nn.Parameter]] = []
    named_params.extend(_iter_named_trainable_params(decoder_base, "dec"))
    named_params.extend(_iter_named_trainable_params(encoder_base, "enc"))

    for n, p in named_params:
        category = get_param_category(n)
        if category == "proj":
            proj_params.append(p)
        elif category == "prompt":
            prompt_params.append(p)
        else:
            other_params.append(p)

    return {"proj": proj_params, "prompt": prompt_params, "other": other_params}


def build_param_buckets_per_module(
    decoder_base, encoder_base
) -> Dict[str, Dict[str, List[torch.nn.Parameter]]]:
    dec: Dict[str, List[torch.nn.Parameter]] = {"proj": [], "prompt": [], "other": []}
    enc: Dict[str, List[torch.nn.Parameter]] = {"proj": [], "prompt": [], "other": []}

    for n, p in _iter_named_trainable_params(decoder_base, "dec"):
        cat = get_param_category(n)
        if cat == "proj":
            dec["proj"].append(p)
        elif cat == "prompt":
            dec["prompt"].append(p)
        else:
            dec["other"].append(p)

    for n, p in _iter_named_trainable_params(encoder_base, "enc"):
        cat = get_param_category(n)
        if cat == "proj":
            enc["proj"].append(p)
        elif cat == "prompt":
            enc["prompt"].append(p)
        else:
            enc["other"].append(p)

    return {"dec": dec, "enc": enc}


def clip_grads_per_type(
    decoder_base,
    encoder_base,
    clip_proj: float,
    clip_prompt: float,
    clip_other: float,
):
    # Build combined buckets for clipping and per-module buckets for logging
    combined = build_param_buckets(decoder_base, encoder_base)
    per_mod = build_param_buckets_per_module(decoder_base, encoder_base)

    # Pre-clip per-module norms
    dec_norms = {k: _compute_grad_norm(v) for k, v in per_mod["dec"].items()}
    enc_norms = {k: _compute_grad_norm(v) for k, v in per_mod["enc"].items()}

    # Perform actual clipping per type across both modules
    proj_params = combined["proj"]
    prompt_params = combined["prompt"]
    other_params = combined["other"]

    grad_norm_proj = (
        torch.nn.utils.clip_grad_norm_(proj_params, clip_proj) if len(proj_params) > 0 else 0.0
    )
    grad_norm_prompt = (
        torch.nn.utils.clip_grad_norm_(prompt_params, clip_prompt) if len(prompt_params) > 0 else 0.0
    )
    grad_norm_other = (
        torch.nn.utils.clip_grad_norm_(other_params, clip_other) if len(other_params) > 0 else 0.0
    )

    all_params: List[torch.nn.Parameter] = [*proj_params, *prompt_params, *other_params]
    grad_norm_dict = {
        "proj": grad_norm_proj,
        "prompt": grad_norm_prompt,
        "other": grad_norm_other,
    }
    return all_params, grad_norm_dict, dec_norms, enc_norms


