"""Parameter group helpers for fused AdamW."""

from typing import List, Sequence, Union

from torch import nn

__all__ = ["param_groups", "get_param_category"]


def get_param_category(param_name: str) -> str:
    """Categorize a parameter name into a semantic group.

    Returns one of: "proj", "prompt", "embedding", "base", "other".
    """
    is_adapter = (("proj" in param_name) or (".out" in param_name)) and "c_proj" not in param_name
    is_embedding = (
        ("embed" in param_name)
        or ("wte" in param_name)
        or ("wpe" in param_name)
        or ("lm_head" in param_name and "weight" in param_name)
    )
    is_prompt = (
        ("prompt_left_emb" in param_name)
        or ("prompt_right_emb" in param_name)
        or ("soft_prompt_embeddings" in param_name)
        or ("special_last_token_vector" in param_name)
    )
    is_base_model = "base" in param_name

    if is_adapter:
        return "proj"
    if is_prompt:
        return "prompt"
    if is_embedding:
        return "embedding"
    if is_base_model:
        return "base"
    return "other"


def param_groups(
    models: Union[nn.Module, Sequence[nn.Module]],
    lr: float,
    proj_lr_mult: float = 10.0,
    embedding_lr_mult: float = 1.0,
    prompt_lr_mult: float = 10.0,
    base_model_lr_mult: float = 1.0,
    overall_encoder_lr_mult: float = 1.0,
    weight_decay: float = 0.01,
    include_frozen: bool = False,
) -> List[dict]:  # noqa: D401
    """Create parameter groups for AdamW.

    Behaviour:
        • All parameters get the base ``lr``.
        • ``proj`` category: names containing ``.proj`` or ``.out`` (but not ``c_proj``)
          get ``lr * proj_lr_mult`` — lightweight adapters / heads.
        • ``embedding`` category: names containing ``embed``, ``wte``, ``wpe``, or
          ``lm_head`` weight get ``lr * embedding_lr_mult`` — token/pos embeddings and LM head weight.
        • ``prompt`` category: names containing ``prompt_left_emb``, ``prompt_right_emb``,
          or ``soft_prompt_embeddings`` get ``lr * prompt_lr_mult`` — soft prompt embeddings.
        • ``base`` category: names containing ``base`` get ``lr * base_model_lr_mult`` — base model weights.
        • The overall encoder multiplier ``overall_encoder_lr_mult`` is applied to all encoder params.
        • Weight-decay is 0.01 for matrix weights, 0.0 for biases / LayerNorm.

    The function accepts either a single ``nn.Module`` or a sequence. Only
    ``requires_grad=True`` parameters are considered.
    """

    if isinstance(models, nn.Module):
        models = [models]

    groups: List[dict] = []
    seen_param_ids = set()
    for i, m in enumerate(models):
        for n, p in m.named_parameters():
            if (not include_frozen) and (not p.requires_grad):
                continue
            pid = id(p)
            if pid in seen_param_ids:
                continue
            seen_param_ids.add(pid)

            decay = 0.0 if p.ndim == 1 else weight_decay

            category = get_param_category(n)
            if category == "proj":
                lr_scale = proj_lr_mult
            elif category == "embedding":
                lr_scale = embedding_lr_mult
            elif category == "prompt":
                lr_scale = prompt_lr_mult
            elif category == "base":
                lr_scale = base_model_lr_mult
            else:
                lr_scale = 1.0

            if i == 0:
                lr_scale = 1 * lr_scale
            else:
                lr_scale = overall_encoder_lr_mult * lr_scale

            # Attach category for downstream logic (e.g., per-category schedulers)
            # Extra keys are supported by torch.optim param groups
            groups.append(
                {
                    "params": [p],
                    "weight_decay": decay,
                    "lr": lr * lr_scale,
                    "category": category,
                }
            )

    return groups
