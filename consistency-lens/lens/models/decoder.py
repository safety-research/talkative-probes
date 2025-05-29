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
    base_model: bool = False         # YAML `base_model`
    projection_layer: bool = True    # YAML `projection_layer`
    output_head: bool = True         # YAML `output_head`
    embedding_head: bool = False     # YAML `embedding_head`
    eye_init: bool = True            # YAML `eye_init`
    trainable_prompts: bool = True   # YAML `trainable_prompts`
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_prompt_tokens < 0:
            raise ValueError(f"n_prompt_tokens must be non-negative, got {self.n_prompt_tokens}")


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
            p.requires_grad_(cfg.base_model)
        self.config = cfg
        d_model = self.base.config.hidden_size
        self.proj = nn.Linear(d_model, d_model, bias=True)
        # Initialize as identity matrix
        if cfg.eye_init:
            nn.init.eye_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
            log.info("Initialized projection layer as identity matrix")
        # Configure trainability of the input projection layer
        for p in self.proj.parameters():
            p.requires_grad_(cfg.projection_layer)

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
            p.requires_grad_(cfg.output_head)
        
        # Configure trainability of the embedding heads (input/output embeddings)
        # These may be tied in many models like GPT-2
        try:
            input_embeddings = self.base.get_input_embeddings()
            if input_embeddings is not None:
                for p in input_embeddings.parameters():
                    p.requires_grad_(cfg.embedding_head)
        except AttributeError:
            log.warning("Could not access input embeddings for freezing control")
        
        try:
            output_embeddings = self.base.get_output_embeddings()
            if output_embeddings is not None:
                for p in output_embeddings.parameters():
                    p.requires_grad_(cfg.embedding_head)
        except AttributeError:
            # Fallback for models that expose `lm_head`
            if hasattr(self.base, 'lm_head'):
                for p in self.base.lm_head.parameters():
                    p.requires_grad_(cfg.embedding_head)
            else:
                log.warning("Could not access output embeddings for freezing control")
        
        # --- Prompt placeholders -------------------------------------------------
        self.prompt_left_emb = None      # type: nn.Parameter | None
        self.prompt_right_emb = None     # type: nn.Parameter | None
        self.prompt_len = 0
        self.prompt_text = []
        # keep prompt_ids only for logging/debug convenience
        self.register_buffer("prompt_ids", torch.empty(0, dtype=torch.long))


    def forward(self, *args: Any, **kwargs: Any):  # noqa: D401
        raise NotImplementedError

    def swap_base_model(self, model_name_or_path: str, keep_projection: bool = True) -> None:
        """Swap the base model with a different one (e.g., untrained version).
        
        Args:
            model_name_or_path: Model identifier or path to load
            keep_projection: Whether to keep the current projection layer weights
        """
        old_dtype = self.proj.weight.dtype
        old_device = self.proj.weight.device
        
        # Store old projection weights if requested
        if keep_projection:
            old_proj_weight = self.proj.weight.data.clone()
            old_proj_bias = self.proj.bias.data.clone()
        
        # Load new base model
        self.base = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.base = self.base.to(old_device)
        
        # Re-apply trainability settings
        for p in self.base.parameters():
            p.requires_grad_(self.config.base_model)
        
        # Update output head dimensions if needed
        d_model = self.base.config.hidden_size
        vocab_size = self.base.config.vocab_size
        
        if self.out.out_features != vocab_size:
            self.out = nn.Linear(d_model, vocab_size, bias=False).to(old_device)
            # Initialize with new model's output embeddings
            with torch.no_grad():
                try:
                    orig_out_w = self.base.get_output_embeddings().weight
                except AttributeError:
                    orig_out_w = self.base.lm_head.weight
                if orig_out_w.shape == self.out.weight.shape:
                    self.out.weight.copy_(orig_out_w)
            
            for p in self.out.parameters():
                p.requires_grad_(self.config.output_head)
        
        # Restore projection weights if requested and dimensions match
        if keep_projection and self.proj.weight.shape == (d_model, d_model):
            self.proj.weight.data = old_proj_weight.to(old_dtype)
            self.proj.bias.data = old_proj_bias.to(old_dtype)
        else:
            # Reinitialize projection if dimensions changed
            self.proj = nn.Linear(d_model, d_model, bias=True).to(old_device)
            if self.config.eye_init:
                nn.init.eye_(self.proj.weight)
                nn.init.zeros_(self.proj.bias)
            for p in self.proj.parameters():
                p.requires_grad_(self.config.projection_layer)
        
        log.info(f"Swapped base model to: {model_name_or_path}")

    def generate_soft(
        self,
        activation_input: torch.Tensor,
        max_length: int,
        gumbel_tau: float,
        use_projection: bool = True,
        print_prompt: bool = False,
        hard_left_emb: list[int] = None,
        hard_right_emb: list[int] = None,
        override_model_base_and_out  = None,
    ) -> Generated:
        """Differentiable autoregressive sampling with Gumbel-Softmax.

        * A_proj is prepended as the first (non-text) token embedding.
        * For *max_length* steps we generate soft tokens and feed them back in.
        * Returns the *text* part only (excludes the A_proj token) so caller
          can hand it to the Encoder directly.
        
        Args:
            activation_input: Input activation tensor
            max_length: Maximum generation length
            gumbel_tau: Temperature for Gumbel-Softmax
            use_projection: Whether to apply the projection layer to activation_input
            print_prompt: Whether to print the prompt text with activation insertion point
        """
        
        if print_prompt and hasattr(self, 'prompt_text'):
            print(f"Prompt template: {self.prompt_text}")

        # Ensure dtype matches linear layer to avoid Half/Float mismatch during eval.
        if self.prompt_left_emb is not None and self.prompt_left_emb.device != activation_input.device:
            self.prompt_left_emb = self.prompt_left_emb.to(activation_input.device)
        if self.prompt_right_emb is not None and self.prompt_right_emb.device != activation_input.device:
            self.prompt_right_emb = self.prompt_right_emb.to(activation_input.device)


        if override_model_base_and_out is not None:
            main_model = override_model_base_and_out
            main_base = main_model.model
            main_out = main_model.model.lm_head
        else:   
            main_model = self
            main_base = self.base
            main_out = self.out

        if hard_left_emb is not None:
            prompt_left_emb = main_base.get_input_embeddings().weight[hard_left_emb].clone()
        else:
            prompt_left_emb = self.prompt_left_emb
        if hard_right_emb is not None:
            prompt_right_emb = main_base.get_input_embeddings().weight[hard_right_emb].clone()
        else:
            prompt_right_emb = self.prompt_right_emb

        activation_input = activation_input.to(self.proj.weight.dtype)

        B, d_model = activation_input.shape
        device = activation_input.device
        
        # Get both input and output embedding tables
        input_emb_table = main_base.get_input_embeddings().weight  # (V, d_model)
        output_emb_table = main_base.get_output_embeddings().weight  # (V, d_model)
        
        # Check if embeddings are tied (same memory location)
        embeddings_tied = (input_emb_table.data_ptr() == output_emb_table.data_ptr()) # TODO - better check?
        

        # 0) prepend textual prompt (pre-computed at set_prompt)
        parts = []
        if prompt_left_emb is not None:
            parts.append(prompt_left_emb.expand(B, -1, -1))
        if prompt_right_emb is not None:
            parts.append(prompt_right_emb.expand(B, -1, -1))
        if use_projection:
            a_proj = self.proj(activation_input).unsqueeze(1)
        else:
            a_proj = activation_input.unsqueeze(1)
        parts.append(a_proj)
        seq_embs = torch.cat(parts, dim=1)

        logits_list = []
        hard_ids_list = []
        output_embs_list = []  # Store embeddings for encoder

        for _ in range(max_length):
            out = main_base(inputs_embeds=seq_embs, output_hidden_states=True)
            h_last = out.last_hidden_state if hasattr(out, "last_hidden_state") else out.hidden_states[-1]
            logits_t = main_out(h_last[:, -1])  # (B, V)

            # 1. Apply forward sampling temperature
            current_T_sampling = 1.0 # T_sampling_schedule(current_step_or_epoch) # Get from your schedule - TODO may want to add a schedule/config for this.
            logits_t_scaled = logits_t / current_T_sampling

            # 2. Apply Gumbel-Softmax with STE temperature
            ste_token_dist = torch.nn.functional.gumbel_softmax(
                logits_t_scaled,
                tau=gumbel_tau,
                hard=True
            )
            
            # Use input embeddings for autoregressive feedback
            emb_t_input = ste_token_dist @ input_emb_table  # (B, d_model)
            
            # Use output embeddings for the encoder (or reuse input if tied)
            if embeddings_tied:
                emb_t_output = emb_t_input
            else:
                emb_t_output = ste_token_dist @ output_emb_table  # (B, d_model)
            
            # Feed input embedding back for next autoregressive step
            seq_embs = torch.cat([seq_embs, emb_t_input.unsqueeze(1)], dim=1)
            
            # Store output embedding for encoder
            output_embs_list.append(emb_t_output)

            logits_list.append(logits_t)
            # Store hard token IDs derived from the STE output
            hard_ids_list.append(ste_token_dist.argmax(dim=-1))

        # Stack along time dim (B, T, ...)
        logits_seq = torch.stack(logits_list, dim=1)
        hard_ids = torch.stack(hard_ids_list, dim=1)
        
        # Stack output embeddings for encoder
        text_embs = torch.stack(output_embs_list, dim=1)  # (B, T, d_model)

        return Generated(text_embs, logits_seq, hard_ids)

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def tokenize_and_embed_prompt(self, prompt: str, tokenizer) -> tuple[list[int], list[int], torch.Tensor, torch.Tensor]:
        """Tokenise *prompt*, split on '<embed>', cache embeddings."""
        left_str, *right = prompt.split("<embed>")
        right_str = right[0] if right else ""
        left_ids  = tokenizer(left_str,  add_special_tokens=False).input_ids if left_str else []
        right_ids = tokenizer(right_str, add_special_tokens=False).input_ids if right_str else []
        embed_table = self.base.get_input_embeddings().weight
        left_emb = embed_table[torch.tensor(left_ids, device=embed_table.device)].clone() if left_ids else None
        right_emb = embed_table[torch.tensor(right_ids, device=embed_table.device)].clone() if right_ids else None
        return left_ids, right_ids, left_emb, right_emb

    def set_prompt(self, prompt: str, tokenizer) -> None:
        """Tokenise *prompt*, split on '<embed>', cache embeddings.

        '<embed>' marks where the activation embedding is inserted."""
        left_ids, right_ids, left_emb, right_emb = self.tokenize_and_embed_prompt(prompt, tokenizer)

        # store ids for logging
        ids = left_ids + [tokenizer.eos_token_id] + right_ids
        # Properly update the registered buffer instead of reassigning
        ids_tensor = torch.tensor(ids, dtype=torch.long, device=self.prompt_ids.device)
        self.prompt_ids.resize_(len(ids))
        self.prompt_ids.copy_(ids_tensor)
        self.prompt_text = tokenizer.decode(left_ids) + '<embed>' + tokenizer.decode(right_ids)

        # Delete old parameters if they exist to avoid memory leaks
        if hasattr(self, 'prompt_left_emb') and self.prompt_left_emb is not None:
            delattr(self, 'prompt_left_emb')
        if hasattr(self, 'prompt_right_emb') and self.prompt_right_emb is not None:
            delattr(self, 'prompt_right_emb')
            
        # Create trainable parameters initialized from the embedding table
        if left_emb is not None:
            self.prompt_left_emb = nn.Parameter(left_emb)
            self.prompt_left_emb.requires_grad_(self.config.trainable_prompts)
        if right_emb is not None:
            self.prompt_right_emb = nn.Parameter(right_emb)
            self.prompt_right_emb.requires_grad_(self.config.trainable_prompts)
            

        self.prompt_len = len(left_ids) + len(right_ids)
