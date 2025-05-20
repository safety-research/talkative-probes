"""Encoder that reconstructs activation from decoder-generated text."""

import torch
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel

__all__ = ["Encoder"]


class Encoder(nn.Module):
    """Processes generated tokens and projects final hidden to activation dims."""

    def __init__(self, model_name: str):
        super().__init__()
        self.base: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name)
        d_model = self.base.config.hidden_size
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Project final token embedding back to activation space."""
        last = embeddings[:, -1]  # (B, d_model)
        return self.proj(last)
