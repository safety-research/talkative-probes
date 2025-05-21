"""Decoder with Gumbel-Softmax generation over projection-injected activation."""

from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel
import logging
__all__ = ["Decoder", "Generated"]

log = logging.getLogger(__name__)
class Generated(NamedTuple):
    generated_text_embeddings: torch.Tensor
    raw_lm_logits: torch.Tensor
    hard_token_ids: torch.Tensor


@dataclass
class DecoderConfig:
    model_name: str
    n_prompt_tokens: int = 8
    train_base_model: bool = False
    train_projection_layer: bool = True
    train_output_head: bool = True


class Decoder(nn.Module):
    """Prepends prompt, inserts projected activation, emits differentiable tokens.

    The Decoder is architecturally similar to the base LLM (LLM_orig). It takes a
    projected activation `A_proj` (derived from an internal LLM activation `A`)
    and, optionally, a textual prompt, then autoregressively generates a textual
    explanation.

    Key components:
    - `self.base`: The underlying LLM structure (e.g., a GPT-2 or LLaMA model).
                   Its weights can be frozen or fine-tuned.
    - `self.proj`: A linear layer (`Proj_A_to_D_emb` in the README) that maps the
                   input activation `A` into the embedding space of `self.base`.
    - `self.out`: A linear layer that acts as the language modeling head, mapping
                  the final hidden state from `self.base` to vocabulary logits.
                  This is crucial for generation, especially if `self.base` is frozen,
                  as it allows the Decoder to learn to produce coherent text.
    """

    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        self.base: PreTrainedModel = AutoModelForCausalLM.from_pretrained(cfg.model_name)
        # Configure trainability of the base model
        for p in self.base.parameters():
            p.requires_grad_(cfg.train_base_model)

        d_model = self.base.config.hidden_size
        self.proj = nn.Linear(d_model, d_model, bias=False)
        # Configure trainability of the input projection layer
        for p in self.proj.parameters():
            p.requires_grad_(cfg.train_projection_layer)

        # The output head (self.out) maps hidden states to vocabulary logits.
        # This is essential for the Decoder to generate text.
        # If self.base is frozen, self.out allows the Decoder to adapt the
        # (frozen) base model\'s representations for the explanation generation task.
        # If self.base is trainable, self.out is trained along with it.
        self.out = nn.Linear(d_model, self.base.config.vocab_size, bias=False)
        # Initialise `self.out` with a copy of the original unembedding matrix
        # (i.e. the tied output embedding / LM head from the base model).
        with torch.no_grad():
            try:
                orig_out_w = self.base.get_output_embeddings().weight
            except AttributeError:
                # Fallback for models that expose `lm_head`
                orig_out_w = self.base.lm_head.weight  # type: ignore[attr-defined]

            if orig_out_w.shape == self.out.weight.shape:
                self.out.weight.copy_(orig_out_w)

        # Configure trainability of the output head (separate from the base model)
        for p in self.out.parameters():
            p.requires_grad_(cfg.train_output_head)

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
        """Differentiable autoregressive sampling with Gumbel-Softmax.

        * A_proj is prepended as the first (non-text) token embedding.
        * For *max_length* steps we generate soft tokens and feed them back in.
        * Returns the *text* part only (excludes the A_proj token) so caller
          can hand it to the Encoder directly.
        """

        # Ensure dtype matches linear layer to avoid Half/Float mismatch during eval.
        activation_input = activation_input.to(self.proj.weight.dtype)

        B, d_model = activation_input.shape
        device = activation_input.device

        # Embedding table reference once.
        emb_table = self.base.get_input_embeddings().weight  # (V, d_model)

        # 0) optional textual prompt tokens
        if self.n_prompt and self.n_prompt > 0:
            prompt_emb = emb_table[self.prompt_ids.expand(B, -1)]  # (B, P, d_model)
        else:
            prompt_emb = None

        # 1) projected activation token
        a_proj = self.proj(activation_input).unsqueeze(1)  # (B, 1, d_model)

        seq_embs = torch.cat([prompt_emb, a_proj], dim=1) if prompt_emb is not None else a_proj

        logits_list = []
        hard_ids_list = []

        for _ in range(max_length):
            out = self.base(inputs_embeds=seq_embs, output_hidden_states=True)
            h_last = out.last_hidden_state if hasattr(out, "last_hidden_state") else out.hidden_states[-1]
            logits_t = self.out(h_last[:, -1])  # (B, V)

            # Use gumbel_softmax with hard=True for Straight-Through Estimation (STE)
            # Forward pass uses one-hot samples, backward pass uses soft probabilities.
            ste_token_dist = torch.nn.functional.gumbel_softmax(logits_t, tau=gumbel_tau, hard=True)
            emb_t = ste_token_dist @ emb_table  # (B, d_model)

            seq_embs = torch.cat([seq_embs, emb_t.unsqueeze(1)], dim=1)

            logits_list.append(logits_t)
            # Store hard token IDs derived from the STE output
            hard_ids_list.append(ste_token_dist.argmax(dim=-1))

        # Stack along time dim (B, T, ...)
        logits_seq = torch.stack(logits_list, dim=1)
        hard_ids = torch.stack(hard_ids_list, dim=1)

        # Expose only generated text (exclude prompt & A_proj) embeddings
        start_idx = 1 + (prompt_emb.shape[1] if prompt_emb is not None else 0)
        text_embs = seq_embs[:, start_idx:]

        return Generated(text_embs, logits_seq, hard_ids)

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def set_prompt(self, prompt: str, tokenizer) -> None:  # noqa: D401
        """Tokenise *prompt* and store first *n_prompt* ids in ``prompt_ids``.

        Extra tokens are truncated; shorter prompts are right-padded with BOS.
        Call this **after** constructing the Decoder and once per run.
        """

        toks = tokenizer(prompt, add_special_tokens=False).input_ids[: self.n_prompt]
        if len(toks) < self.n_prompt:
            toks = toks + [self.base.config.bos_token_id] * (self.n_prompt - len(toks))

        with torch.no_grad():
            self.prompt_ids.copy_(torch.tensor([toks], dtype=self.prompt_ids.dtype, device=self.prompt_ids.device))
