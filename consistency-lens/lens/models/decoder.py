"""Decoder with Gumbel-Softmax generation over projection-injected activation."""

from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel

__all__ = ["Decoder", "Generated"]


class Generated(NamedTuple):
    generated_text_embeddings: torch.Tensor
    raw_lm_logits: torch.Tensor
    hard_token_ids: torch.Tensor


@dataclass
class DecoderConfig:
    model_name: str
    n_prompt_tokens: int = 8


class Decoder(nn.Module):
    """Prepends prompt, inserts projected activation, emits differentiable tokens."""

    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        self.base: PreTrainedModel = AutoModelForCausalLM.from_pretrained(cfg.model_name)
        d_model = self.base.config.hidden_size
        self.proj = nn.Linear(d_model, d_model, bias=False)
        # Simple output projection so we don't rely on the base model's full forward.
        self.out = nn.Linear(d_model, self.base.config.vocab_size, bias=False)
        self.n_prompt = cfg.n_prompt_tokens
        self.register_buffer(
            "prompt_ids", torch.full((1, self.n_prompt), self.base.config.bos_token_id, dtype=torch.long)
        )

    def forward(self, *args: Any, **kwargs: Any):  # noqa: D401
        raise NotImplementedError

    def generate_soft(
        self,
        activation_input: torch.Tensor,
        max_length: int,
        gumbel_tau: float,
    ) -> Generated:
        """Project *activation_input* to a single-token embedding and produce logits.

        This deliberately keeps things dead-simple: we *do not* autoregress, we
        just map the activation vector to one token worth of logits.  That is
        enough to exercise gradients end-to-end in the MVP.
        """

        batch = activation_input.size(0)
        # --- 1. Project activation to model hidden size ---
        hid = self.proj(activation_input)  # (B, d_model)

        # --- 2. Gumbel-softmax over vocabulary ---
        logits = self.out(hid)  # (B, V)
        # Straight Gumbel-softmax – sampling hard=False keeps it differentiable.
        soft = torch.nn.functional.gumbel_softmax(logits, tau=gumbel_tau, hard=False)

        # --- 3. Convert soft distribution to embedding vector ---
        emb_weight = self.base.get_input_embeddings().weight  # (V, d_model)
        emb = soft @ emb_weight  # (B, d_model)

        # --- 4. Hard ids (for logging / CE) ---
        hard_ids = logits.argmax(dim=-1)  # (B,)

        # For downstream code we expose a *sequence* length dim.
        emb_seq = emb.unsqueeze(1)  # (B, 1, d_model)
        logits_seq = logits.unsqueeze(1)  # (B, 1, V)

        return Generated(emb_seq, logits_seq, hard_ids.unsqueeze(1))
