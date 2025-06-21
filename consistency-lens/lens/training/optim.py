"""Parameter group helpers for fused AdamW."""

from typing import List, Sequence, Union

from torch import nn

__all__ = ["param_groups"]


def param_groups(
    models: Union[nn.Module, Sequence[nn.Module]],
    lr: float,
    proj_lr_mult: float = 10.0,
    embedding_lr_mult: float = 1.0,
    prompt_lr_mult: float = 10.0,
    base_model_lr_mult: float = 1.0,
    overall_encoder_lr_mult: float = 1.0,
    weight_decay: float = 0.01,
) -> List[dict]:  # noqa: D401
    """Create parameter groups for AdamW.

    Behaviour:
        • All parameters get the base ``lr``.
        • Parameters whose *name* contains ``.proj`` **or** ``.out`` get
          ``lr * proj_lr_mult`` — these are the lightweight adapters / heads.
        • Parameters whose *name* contains ``embed``, ``prompt_left_emb``, 
          ``prompt_right_emb``, or ``soft_prompt_embeddings`` get
          ``lr * embedding_lr_mult`` — these are the embedding layers and soft prompts.
        • Parameters whose *name* contains ``base`` get
          ``lr * base_model_lr_mult`` — these are the base model parameters.
        • Weight-decay is 0.01 for matrix weights, 0.0 for biases / LayerNorm.

    The function accepts either a single ``nn.Module`` or a sequence. Only
    ``requires_grad=True`` parameters are considered.
    """

    if isinstance(models, nn.Module):
        models = [models]

    groups: List[dict] = []
    for i,m in enumerate(models):
        for n, p in m.named_parameters():
            if not p.requires_grad:
                continue

            decay = 0.0 if p.ndim == 1 else weight_decay
            
            # Check parameter type for learning rate scaling
            is_adapter = (("proj" in n) or (".out" in n)) and not ("c_proj" in n)
            is_embedding = (
                ("embed" in n) or 
                ("wte" in n) or 
                ("wpe" in n) or 
                ("lm_head" in n and "weight" in n) 
            )
            is_prompt = (
                ("prompt_left_emb" in n) or 
                ("prompt_right_emb" in n) or 
                ("soft_prompt_embeddings" in n)
            )
            is_base_model = ("base" in n)
            
            if is_adapter:
                lr_scale = proj_lr_mult
            elif is_embedding:
                lr_scale = embedding_lr_mult
            elif is_prompt: 
                lr_scale = prompt_lr_mult
            elif is_base_model:
                lr_scale = base_model_lr_mult
            else:
                lr_scale = 1.0

            if i == 0:
                print("Decoder LR multiplier: 1")
                lr_scale = 1* lr_scale
            else:
                print(f"Overall encoder LR multiplier: {overall_encoder_lr_mult}")
                lr_scale = overall_encoder_lr_mult*lr_scale

            groups.append({"params": [p], "weight_decay": decay, "lr": lr * lr_scale})

    return groups
