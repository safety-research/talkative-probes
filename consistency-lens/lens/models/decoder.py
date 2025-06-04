"""Decoder with Gumbel-Softmax generation over projection-injected activation."""

from dataclasses import dataclass
from typing import Any, NamedTuple
from contextlib import contextmanager

import torch
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.cache_utils import DynamicCache 
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
    use_checkpointing: bool = False # YAML `use_checkpointing`
    checkpoint_every_n_tokens: int = 4 # YAML `checkpoint_every_n_tokens`
    use_kv_cache: bool = False  # YAML `use_kv_cache`
    use_flash_attention: bool = False  # YAML `use_flash_attention`
    use_gumbel_for_LMorig: bool = False # YAML `use_gumbel_for_LMorig`
    patch_all_layers: bool = False   # YAML `patch_all_layers` - patch activation at all layers
    per_layer_projections: bool = False  # YAML `per_layer_projections` - use separate projection for each layer
    use_dropout: bool = True         # YAML `use_dropout` - whether to use dropout during training (False = deterministic)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_prompt_tokens < 0:
            raise ValueError(f"n_prompt_tokens must be non-negative, got {self.n_prompt_tokens}")
        if self.per_layer_projections and not self.patch_all_layers:
            raise ValueError("per_layer_projections requires patch_all_layers to be True")


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
        n_layers = self.base.config.num_hidden_layers
        
        # Create projection layer(s)
        if cfg.per_layer_projections:
            # Create a 3D parameter tensor for per-layer projections
            # Shape: (n_layers, d_model, d_model)
            self.proj_weight = nn.Parameter(torch.empty(n_layers, d_model, d_model))
            self.proj_bias = nn.Parameter(torch.empty(n_layers, d_model))
            
            # Initialize as identity matrices
            if cfg.eye_init:
                for i in range(n_layers):
                    nn.init.eye_(self.proj_weight[i])
                nn.init.zeros_(self.proj_bias)
                log.info(f"Initialized {n_layers} per-layer projection matrices as identity")
            else:
                # Default initialization
                for i in range(n_layers):
                    nn.init.xavier_uniform_(self.proj_weight[i])
                nn.init.zeros_(self.proj_bias)
            
            # Configure trainability
            self.proj_weight.requires_grad_(cfg.projection_layer)
            self.proj_bias.requires_grad_(cfg.projection_layer)
            
            # For compatibility, keep self.proj as None
            self.proj = None
        else:
            # Single projection layer (original behavior)
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
    
    def _apply_projection(self, activation_input: torch.Tensor, layer_idx: int = None) -> torch.Tensor:
        """Apply projection to activation input.
        
        Args:
            activation_input: Input activation tensor (B, d_model)
            layer_idx: Layer index for per-layer projections (only used if per_layer_projections=True)
            
        Returns:
            Projected activation (B, d_model)
        """
        if self.config.per_layer_projections:
            if layer_idx is None:
                raise ValueError("layer_idx must be provided when using per_layer_projections")
            # Apply layer-specific projection
            # activation_input: (B, d_model)
            # proj_weight[layer_idx]: (d_model, d_model)
            # proj_bias[layer_idx]: (d_model,)
            return torch.nn.functional.linear(activation_input, self.proj_weight[layer_idx], self.proj_bias[layer_idx])
        else:
            # Use single projection layer
            if self.proj is None:
                raise ValueError("Projection layer not initialized")
            return self.proj(activation_input)

    def swap_base_model(self, model_name_or_path: str, keep_projection: bool = True) -> None:
        """Swap the base model with a different one (e.g., untrained version).
        
        Args:
            model_name_or_path: Model identifier or path to load
            keep_projection: Whether to keep the current projection layer weights
        """
        # Get dtype and device
        if self.config.per_layer_projections:
            old_dtype = self.proj_weight.dtype
            old_device = self.proj_weight.device
        else:
            old_dtype = self.proj.weight.dtype
            old_device = self.proj.weight.device
        
        # Store old projection weights if requested
        if keep_projection:
            if self.config.per_layer_projections:
                old_proj_weight = self.proj_weight.data.clone()
                old_proj_bias = self.proj_bias.data.clone()
            else:
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
        n_layers = self.base.config.num_hidden_layers
        
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
        
        # Restore or reinitialize projection weights
        if self.config.per_layer_projections:
            if keep_projection and old_proj_weight.shape == (n_layers, d_model, d_model):
                self.proj_weight.data = old_proj_weight.to(old_dtype)
                self.proj_bias.data = old_proj_bias.to(old_dtype)
            else:
                # Reinitialize if dimensions changed
                self.proj_weight = nn.Parameter(torch.empty(n_layers, d_model, d_model).to(old_device))
                self.proj_bias = nn.Parameter(torch.empty(n_layers, d_model).to(old_device))
                if self.config.eye_init:
                    for i in range(n_layers):
                        nn.init.eye_(self.proj_weight[i])
                    nn.init.zeros_(self.proj_bias)
                self.proj_weight.requires_grad_(self.config.projection_layer)
                self.proj_bias.requires_grad_(self.config.projection_layer)
        else:
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
    
    @contextmanager
    def _maybe_disable_dropout(self):
        """Context manager to control dropout behavior based on config."""
        if not self.config.use_dropout:
            # Store original training state
            was_training = self.base.training
            try:
                # Set to eval mode to disable dropout
                self.base.eval()
                yield
            finally:
                # Restore original state
                if was_training:
                    self.base.train()
        else:
            # No change needed
            yield

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
        do_patching: bool = True,
        special_token = None
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
        # Parameters are automatically on the correct device when the module is moved


        if override_model_base_and_out is not None:
            main_model = override_model_base_and_out
            # Handle both OrigWrapper (has .model) and raw model cases
            if hasattr(main_model, 'model'):
                main_base = main_model.model
                main_out = main_model.model.lm_head if hasattr(main_model.model, 'lm_head') else main_model.model.get_output_embeddings()
            else:
                # Direct model passed (e.g., GPT2LMHeadModel)
                main_base = main_model
                main_out = main_model.lm_head if hasattr(main_model, 'lm_head') else main_model.get_output_embeddings()
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

        # Get dtype from projection layer
        if self.config.per_layer_projections:
            activation_input = activation_input.to(self.proj_weight.dtype)
        else:
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
        
        # Always insert activation as a token
        if use_projection:
            if self.config.patch_all_layers and self.config.per_layer_projections:
                # Use first layer's projection
                a_proj = self._apply_projection(activation_input, layer_idx=0).unsqueeze(1)
            else:
                # Use single projection
                a_proj = self._apply_projection(activation_input).unsqueeze(1)
        else:
            # No projection
            a_proj = activation_input.unsqueeze(1)
        if do_patching: 
            parts.append(a_proj)
        else: 
            print("patching in special token")
            parts.append(main_base.get_input_embeddings().weight[special_token].clone().unsqueeze(0).unsqueeze(1))
            
        if prompt_right_emb is not None:
            parts.append(prompt_right_emb.expand(B, -1, -1))
            
        seq_embs = torch.cat(parts, dim=1)

        logits_list = []
        hard_ids_list = []
        output_embs_list = []  # Store embeddings for encoder

        for _ in range(max_length):
            if self.config.patch_all_layers:
                # Custom forward pass with activation patching at all layers
                # Get transformer module and detect architecture
                if hasattr(main_base, 'transformer'):
                    # GPT-2 style model
                    transformer = main_base.transformer
                    layers = transformer.h
                    final_norm = transformer.ln_f
                elif hasattr(main_base, 'model'):
                    # LLaMA style model (model.layers)
                    transformer = main_base.model
                    layers = transformer.layers
                    final_norm = transformer.norm
                else:
                    raise ValueError(f"Unknown model architecture. Expected transformer or model attribute.")
                
                # Embedding layer
                hidden_states = seq_embs
                
                # Get position IDs
                seq_length = hidden_states.size(1)
                position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
                
                # Pre-compute single projection if not using per-layer projections
                if not self.config.per_layer_projections:
                    if use_projection:
                        single_proj = self._apply_projection(activation_input)
                    else:
                        single_proj = activation_input
                
                # Calculate embed position (where activation should be patched)
                embed_pos = prompt_left_emb.size(0) if prompt_left_emb is not None else 0
                # Add position embeddings for GPT-2 (but not for LLaMA)
                if hasattr(main_base, 'transformer'):
                    # GPT-2 needs position embeddings added before layers
                    position_embeds = transformer.wpe(position_ids)
                    hidden_states = transformer.drop(hidden_states + position_embeds)

                # Compute rotary embeddings for LLaMA if needed
                if hasattr(main_base, 'model') and hasattr(transformer, 'rotary_emb'):
                    # LLaMA uses rotary embeddings
                    cos, sin = transformer.rotary_emb(hidden_states, position_ids)
                    position_embeddings = (cos, sin)
                else:
                    position_embeddings = None
                
                # Run through transformer layers with activation patching
                for layer_idx, layer_module in enumerate(layers):
                    # Apply layer
                    input_to_this_layer = hidden_states.clone()
                    # Replace activation at the embed position for this layer
                    # Skip layer 0 for per-layer projections since it's already applied
                    if do_patching and (layer_idx > 0 or not self.config.per_layer_projections):
                        if self.config.per_layer_projections:
                            # Use layer-specific projection
                            if use_projection:
                                a_proj_layer = self._apply_projection(activation_input, layer_idx=layer_idx)
                            else:
                                a_proj_layer = activation_input
                            # Replace at embed position
                            input_to_this_layer[:, embed_pos] = a_proj_layer
                        else:
                            # Use pre-computed single projection
                            input_to_this_layer[:, embed_pos] = single_proj
                            
                    with self._maybe_disable_dropout():
                        if hasattr(main_base, 'model'):
                            # LLaMA style - pass position_ids and position_embeddings
                            layer_outputs = layer_module(
                                input_to_this_layer,
                                position_ids=position_ids,
                                position_embeddings=position_embeddings,
                            )
                        else:
                            # GPT-2 style
                            layer_outputs = layer_module(input_to_this_layer)
                    hidden_states = layer_outputs[0]
                    
                    
                
                # Final layer norm
                hidden_states = final_norm(hidden_states)
                
                # Get logits
                h_last = hidden_states
                logits_t = main_out(h_last[:, -1])  # (B, V)
            else:
                # Original behavior
                with self._maybe_disable_dropout():
                    out = main_base(inputs_embeds=seq_embs, output_hidden_states=True)
                h_last = out.last_hidden_state if hasattr(out, "last_hidden_state") else out.hidden_states[-1]
                logits_t = main_out(h_last[:, -1])  # (B, V)

            # 1. Apply forward sampling temperature
            current_T_sampling = 1.0 # T_sampling_schedule(current_step_or_epoch) # Get from your schedule - TODO may want to add a schedule/config for this.
            logits_t_scaled = logits_t / current_T_sampling

            # 2. Apply Gumbel-Softmax with STE temperature
            # Add numerical stability for low tau values
            with torch.amp.autocast('cuda',enabled=False):
                logits_t_f32 = logits_t_scaled.float()
                # Subtract max for numerical stability (detached)
                logits_t_f32 = logits_t_f32 - logits_t_f32.max(dim=-1, keepdim=True)[0].detach()
                ste_token_dist = torch.nn.functional.gumbel_softmax(
                    logits_t_f32,
                    tau=max(gumbel_tau, 0.1),  # Prevent extremely low tau
                    hard=True
                ).to(logits_t_scaled.dtype)
            
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


    def generate_soft_chkpt(
        self,
        activation_input: torch.Tensor,
        max_length: int,
        gumbel_tau: float,
        use_projection: bool = True,
        print_prompt: bool = False,
        hard_left_emb: list[int] = None,
        hard_right_emb: list[int] = None,
        override_model_base_and_out = None,
        checkpoint_every_n_tokens: int = 4,
    ) -> Generated:
        """Differentiable autoregressive generation with gradient checkpointing.
        
        This version maintains full differentiability through all generated tokens
        while using gradient checkpointing to reduce memory usage.
        
        Args:
            Same as generate_soft, plus:
            checkpoint_every_n_tokens: How often to checkpoint (default: every 4 tokens)
        """
        from torch.utils.checkpoint import checkpoint
        
        if print_prompt and hasattr(self, 'prompt_text'):
            print(f"Prompt template: {self.prompt_text}")

        # Ensure dtype matches linear layer to avoid Half/Float mismatch during eval.
        # Parameters are automatically on the correct device when the module is moved

        if override_model_base_and_out is not None:
            main_model = override_model_base_and_out
            # Handle both OrigWrapper (has .model) and raw model cases
            if hasattr(main_model, 'model'):
                main_base = main_model.model
                main_out = main_model.model.lm_head if hasattr(main_model.model, 'lm_head') else main_model.model.get_output_embeddings()
            else:
                # Direct model passed (e.g., GPT2LMHeadModel)
                main_base = main_model
                main_out = main_model.lm_head if hasattr(main_model, 'lm_head') else main_model.get_output_embeddings()
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

        # Get dtype from projection layer
        if self.config.per_layer_projections:
            activation_input = activation_input.to(self.proj_weight.dtype)
        else:
            activation_input = activation_input.to(self.proj.weight.dtype)

        B, d_model = activation_input.shape
        device = activation_input.device
        
        # Get both input and output embedding tables
        input_emb_table = main_base.get_input_embeddings().weight  # (V, d_model)
        output_emb_table = main_base.get_output_embeddings().weight  # (V, d_model)
        
        # Check if embeddings are tied (same memory location)
        embeddings_tied = (input_emb_table.data_ptr() == output_emb_table.data_ptr())

        # 0) prepend textual prompt (pre-computed at set_prompt)
        parts = []
        if prompt_left_emb is not None:
            parts.append(prompt_left_emb.expand(B, -1, -1))
        
        # Always insert activation as a token (will be replaced at each layer if patch_all_layers=True)
        if use_projection and not self.config.patch_all_layers:
            # Only apply projection here if not patching all layers
            a_proj = self._apply_projection(activation_input).unsqueeze(1)
        else:
            # Use unprojected activation as placeholder
            a_proj = activation_input.unsqueeze(1)
        parts.append(a_proj)
            
        if prompt_right_emb is not None:
            parts.append(prompt_right_emb.expand(B, -1, -1))
            
        seq_embs = torch.cat(parts, dim=1)

        logits_list = []
        hard_ids_list = []
        output_embs_list = []  # Store embeddings for encoder
        
        # Define single step function for checkpointing as a method
        def generation_step(seq_embs_input, step_idx, 
                          main_base, main_out, 
                          input_emb_table, output_emb_table, embeddings_tied,
                          activation_input, use_projection, prompt_left_emb,
                          B, device, config, decoder_module,
                          gumbel_tau, max_gumbel_tau=0.1):
            """Single generation step that can be checkpointed."""
            if config.patch_all_layers:
                # Custom forward pass with activation patching at all layers
                # Get transformer module and detect architecture
                if hasattr(main_base, 'transformer'):
                    # GPT-2 style model
                    transformer = main_base.transformer
                    layers = transformer.h
                    final_norm = transformer.ln_f
                elif hasattr(main_base, 'model'):
                    # LLaMA style model (model.layers)
                    transformer = main_base.model
                    layers = transformer.layers
                    final_norm = transformer.norm
                else:
                    raise ValueError(f"Unknown model architecture. Expected transformer or model attribute.")
                
                # Embedding layer
                hidden_states = seq_embs_input
                
                # Get position IDs
                seq_length = hidden_states.size(1)
                position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
                
                # Pre-compute single projection if not using per-layer projections
                if not config.per_layer_projections:
                    if use_projection:
                        single_proj = decoder_module._apply_projection(activation_input)
                    else:
                        single_proj = activation_input
                
                # Calculate embed position (where activation should be patched)
                embed_pos = prompt_left_emb.size(0) if prompt_left_emb is not None else 0
                
                # Compute rotary embeddings for LLaMA if needed
                if hasattr(main_base, 'model') and hasattr(transformer, 'rotary_emb'):
                    # LLaMA uses rotary embeddings
                    cos, sin = transformer.rotary_emb(hidden_states, position_ids)
                    position_embeddings = (cos, sin)
                else:
                    position_embeddings = None
                
                # Run through transformer layers with activation patching
                for layer_idx, layer_module in enumerate(layers):
                    # Apply layer
                    if hasattr(main_base, 'model'):
                        # LLaMA style - pass position embeddings
                        layer_outputs = layer_module(
                            hidden_states,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings,
                        )
                    else:
                        # GPT-2 style
                        layer_outputs = layer_module(hidden_states, position_ids=position_ids)
                    hidden_states = layer_outputs[0]
                    
                    # Replace activation at the embed position for this layer
                    # Skip layer 0 for per-layer projections since it's already applied
                    if layer_idx > 0 or not config.per_layer_projections:
                        if config.per_layer_projections:
                            # Use layer-specific projection
                            if use_projection:
                                a_proj_layer = decoder_module._apply_projection(activation_input, layer_idx=layer_idx)
                            else:
                                a_proj_layer = activation_input
                            hidden_states[:, embed_pos] = a_proj_layer
                        else:
                            # Use pre-computed single projection
                            hidden_states[:, embed_pos] = single_proj
                
                # Final layer norm
                hidden_states = final_norm(hidden_states)
                
                # Get logits
                h_last = hidden_states
                logits_t = main_out(h_last[:, -1])  # (B, V)
            else:
                # Original behavior
                # Model forward pass
                out = main_base(inputs_embeds=seq_embs_input, output_hidden_states=True)
                h_last = out.last_hidden_state if hasattr(out, "last_hidden_state") else out.hidden_states[-1]
                logits_t = main_out(h_last[:, -1])  # (B, V)

            # 1. Apply forward sampling temperature
            current_T_sampling = 1.0  # TODO: could add schedule
            logits_t_scaled = logits_t / current_T_sampling

            # 2. Apply Gumbel-Softmax with STE temperature (hard=True)
            # Add numerical stability for low tau values
            with torch.amp.autocast('cuda',enabled=False):
                logits_t_f32 = logits_t_scaled.float()
                # Subtract max for numerical stability (detached)
                logits_t_f32 = logits_t_f32 - logits_t_f32.max(dim=-1, keepdim=True)[0].detach()
                ste_token_dist = torch.nn.functional.gumbel_softmax(
                    logits_t_f32,
                    tau=max(gumbel_tau, 0.1),  # Prevent extremely low tau
                    hard=True  # Keep using STE as in original
                ).to(logits_t_scaled.dtype)
            
            # Use input embeddings for autoregressive feedback
            emb_t_input = ste_token_dist @ input_emb_table  # (B, d_model)
            
            # Use output embeddings for the encoder (or reuse input if tied)
            if embeddings_tied:
                emb_t_output = emb_t_input
            else:
                emb_t_output = ste_token_dist @ output_emb_table  # (B, d_model)
            
            # Store hard token IDs derived from the STE output
            hard_ids = ste_token_dist.argmax(dim=-1)
            
            return logits_t, emb_t_input, emb_t_output, hard_ids

        # Main generation loop with checkpointing
        for step in range(max_length):
            # Checkpoint all tokens except the first few (need some non-checkpointed for gradients)
            # When checkpoint_every_n_tokens=1, checkpoint all but first token
            # When checkpoint_every_n_tokens>1, checkpoint every N tokens
            should_checkpoint = (
                checkpoint_every_n_tokens == 1 and step > 0  # For every-token mode
            ) or (
                checkpoint_every_n_tokens > 1 and step % checkpoint_every_n_tokens == 0 and step > 0
            )
            
            if should_checkpoint:
                # Use gradient checkpointing
                logits_t, emb_t_input, emb_t_output, hard_ids = checkpoint(
                    generation_step, seq_embs, step, 
                    main_base, main_out,
                    input_emb_table, output_emb_table, embeddings_tied,
                    activation_input, use_projection, prompt_left_emb,
                    B, device, self.config, self,
                    gumbel_tau,
                    use_reentrant=False
                )
            else:
                # Regular forward pass
                logits_t, emb_t_input, emb_t_output, hard_ids = generation_step(
                    seq_embs, step,
                    main_base, main_out,
                    input_emb_table, output_emb_table, embeddings_tied,
                    activation_input, use_projection, prompt_left_emb,
                    B, device, self.config, self,
                    gumbel_tau
                )
            
            # Feed input embedding back for next autoregressive step
            seq_embs = torch.cat([seq_embs, emb_t_input.unsqueeze(1)], dim=1)
            
            # Store outputs
            logits_list.append(logits_t)
            output_embs_list.append(emb_t_output)
            hard_ids_list.append(hard_ids)

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
    
    def generate_soft_kv_cached(
        self,
        activation_input: torch.Tensor,
        max_length: int,
        gumbel_tau: float,
        use_projection: bool = True,
        print_prompt: bool = False,
        hard_left_emb: list[int] = None,
        hard_right_emb: list[int] = None,
        override_model_base_and_out = None,
        do_patching: bool = True,
        special_token = None
    ) -> Generated:
        """Differentiable generation with KV caching for O(n) attention computation.
        
        This method produces identical results to generate_soft but avoids
        recomputing attention for past tokens by caching their key/value projections.
        Only works with GPT-2 architecture currently.
        
        Args:
            Same as generate_soft
            
        Returns:
            Same as generate_soft
        """
        # Using native transformer KV caching instead of custom implementation
        
        # Setup similar to generate_soft
        if print_prompt and hasattr(self, 'prompt_text'):
            print(f"Prompt template: {self.prompt_text}")

        # Ensure dtype matches linear layer
        # Parameters are automatically on the correct device when the module is moved

        if override_model_base_and_out is not None:
            main_model = override_model_base_and_out
            # Handle both OrigWrapper (has .model) and raw model cases
            if hasattr(main_model, 'model'):
                main_base = main_model.model
                main_out = main_model.model.lm_head if hasattr(main_model.model, 'lm_head') else main_model.model.get_output_embeddings()
            else:
                # Direct model passed (e.g., GPT2LMHeadModel)
                main_base = main_model
                main_out = main_model.lm_head if hasattr(main_model, 'lm_head') else main_model.get_output_embeddings()
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

        # Get dtype from projection layer
        if self.config.per_layer_projections:
            activation_input = activation_input.to(self.proj_weight.dtype)
        else:
            activation_input = activation_input.to(self.proj.weight.dtype)
        B, d_model = activation_input.shape
        device = activation_input.device
        
        # Get embedding tables
        input_emb_table = main_base.get_input_embeddings().weight
        output_emb_table = main_base.get_output_embeddings().weight
        embeddings_tied = (input_emb_table.data_ptr() == output_emb_table.data_ptr())
        
        # Prepare initial sequence
        parts = []
        if prompt_left_emb is not None:
            parts.append(prompt_left_emb.expand(B, -1, -1))
        
        # Always insert activation as a token
        if use_projection:
            if self.config.patch_all_layers and self.config.per_layer_projections:
                # Use first layer's projection
                a_proj = self._apply_projection(activation_input, layer_idx=0).unsqueeze(1)
            else:
                # Use single projection
                a_proj = self._apply_projection(activation_input).unsqueeze(1)
        else:
            # No projection
            a_proj = activation_input.unsqueeze(1)
        if do_patching: 
            parts.append(a_proj)
        else: 
            print("patching in special token")
            parts.append(main_base.get_input_embeddings().weight[special_token].clone().unsqueeze(0).unsqueeze(1))
            
        if prompt_right_emb is not None:
            parts.append(prompt_right_emb.expand(B, -1, -1))
            
        seq_embs = torch.cat(parts, dim=1)
        
        # Initialize storage
        logits_list = []
        hard_ids_list = []
        output_embs_list = []
        
        # Storage for past_key_values
        past_key_values = None
        
        # Get the transformer and detect architecture
        if hasattr(main_base, 'transformer'):
            # GPT-2 style model
            transformer = main_base.transformer
            layers = transformer.h
            final_norm = transformer.ln_f
        elif hasattr(main_base, 'model'):
            # LLaMA style model (model.layers)
            transformer = main_base.model
            layers = transformer.layers
            final_norm = transformer.norm
        else:
            raise ValueError(f"Unknown model architecture. Expected transformer or model attribute.")
        
        # Process initial sequence (prompt + activation)
        if self.config.patch_all_layers:
            # Custom implementation with patching
            hidden_states = seq_embs
            seq_length = hidden_states.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
            
            # Pre-compute single projection if not using per-layer projections
            if not self.config.per_layer_projections:
                if use_projection:
                    single_proj = self._apply_projection(activation_input)
                else:
                    single_proj = activation_input
            
            # Calculate embed position
            embed_pos = prompt_left_emb.size(0) if prompt_left_emb is not None else 0
            
            # Add position embeddings for GPT-2 (but not for LLaMA)
            if hasattr(main_base, 'transformer'):
                # GPT-2 needs position embeddings added before layers
                position_embeds = transformer.wpe(position_ids)
                hidden_states = transformer.drop(hidden_states + position_embeds)
            
            # Compute rotary embeddings for LLaMA if needed
            if hasattr(main_base, 'model') and hasattr(transformer, 'rotary_emb'):
                # LLaMA uses rotary embeddings
                cos, sin = transformer.rotary_emb(hidden_states, position_ids)
                position_embeddings = (cos, sin)
            else:
                position_embeddings = None
            
            # Process each layer with activation patching
            past_key_values = DynamicCache()
            for layer_idx, layer_module in enumerate(layers):
                # FIRST: Apply patching to the input of this layer
                input_to_this_layer = hidden_states.clone()
                if self.config.per_layer_projections and do_patching:
                    # For per-layer projections, skip layer 0 since it's already applied in seq_embs
                    if layer_idx > 0:
                        if use_projection:
                            a_proj_layer = self._apply_projection(activation_input, layer_idx=layer_idx)
                        else:
                            a_proj_layer = activation_input
                        #input_to_this_layer = hidden_states.clone()
                        input_to_this_layer[:, embed_pos] = a_proj_layer
                elif do_patching:
                    # For single projection, apply to all layers except layer 0 if already applied
                    # Check if layer 0 already has projection applied (it should in seq_embs)
                    #if layer_idx > 0:
                    #input_to_this_layer = hidden_states.clone()
                    input_to_this_layer[:, embed_pos] = single_proj

                # THEN: Process the patched input through the layer
                with self._maybe_disable_dropout():
                    if hasattr(main_base, 'transformer'):
                        # GPT-2 style - position embeddings already added
                        layer_outputs = layer_module(
                            input_to_this_layer,
                            use_cache=True,
                            past_key_value=past_key_values
                        )
                    else:
                        # LLaMA style - pass position_ids and position_embeddings
                        layer_outputs = layer_module(
                            input_to_this_layer,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings,
                            use_cache=True,
                            past_key_value=past_key_values
                        )
                    
                hidden_states = layer_outputs[0]
                #print("Layer", layer_idx, "has", len(past_key_values), "past_key_values")
                #print("layer_outputs", layer_outputs)
                
                # Collect past_key_values
                #if len(layer_outputs) > 1 and layer_outputs[1] is not None:
                #    past_key_values.append(layer_outputs[1])
                #else: 
                #    print("no kvs returned, why?")
                #    print(layer_module)
                #    print(dir(layer_module))
                #    raise ValueError("past_key_values is None")
            # Final layer norm
            hidden_states = final_norm(hidden_states)
            
            # Convert list to tuple for compatibility
            #past_key_values = tuple(past_key_values)#torch.cat(past_key_values, dim=-2)
        else:
            # Original behavior - use standard forward pass with caching
            with self._maybe_disable_dropout():
                outputs = main_base(
                    inputs_embeds=seq_embs,
                    use_cache=True,
                    output_hidden_states=True
                )
            hidden_states = outputs.hidden_states[-1]
            past_key_values = outputs.past_key_values
        
        # print("patchalllayers", self.config.patch_all_layers)
        # print("perlayerprojections", self.config.per_layer_projections)
        # print("len(layers)", len(layers))
        # print("len(past_key_values)", len(past_key_values))
        # Generate tokens
        current_position = seq_embs.size(1)
        
        for step in range(max_length):
            # Get logits for the last position
            logits_t = main_out(hidden_states[:, -1])  # (B, V)
            
            # Gumbel-Softmax sampling with numerical stability
            logits_t_scaled = logits_t / 1.0  # T_sampling = 1.0
            with torch.amp.autocast('cuda',enabled=False):
                logits_f32 = logits_t_scaled.float()
                # Subtract max for numerical stability (detached)
                logits_f32 = logits_f32 - logits_f32.max(dim=-1, keepdim=True)[0].detach()
                ste_token_dist = torch.nn.functional.gumbel_softmax(
                    logits_f32,
                    tau=max(gumbel_tau, 0.1),  # Prevent extremely low tau
                    hard=True
                ).to(logits_t.dtype)
            
            # Get embeddings
            emb_t_input = ste_token_dist @ input_emb_table
            if embeddings_tied:
                emb_t_output = emb_t_input
            else:
                emb_t_output = ste_token_dist @ output_emb_table
            
            # Store outputs
            logits_list.append(logits_t)
            output_embs_list.append(emb_t_output)
            hard_ids_list.append(ste_token_dist.argmax(dim=-1))
            
            # Process new token through transformer with cached K,V
            if step < max_length - 1:  # Don't process last token if we won't use it
                if self.config.patch_all_layers:
                    # Custom processing with patching at each layer
                    hidden_states = emb_t_input.unsqueeze(1)
                    # Original behavior - use native caching
                    with self._maybe_disable_dropout():
                        outputs = main_base(
                            inputs_embeds=emb_t_input.unsqueeze(1),
                            past_key_values=past_key_values,
                            use_cache=True,
                            output_hidden_states=True
                        )
                    hidden_states = outputs.hidden_states[-1]
                    past_key_values = outputs.past_key_values
                    current_position += 1
                    # position_ids = torch.arange(current_position, current_position + 1, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
                    
                    # # For incremental generation, we're processing new tokens only
                    # # The activation was already inserted and cached in the initial pass
                    # # So we just need to process normally through layers
                    
                    # # Add position embeddings for GPT-2 (but not for LLaMA)
                    # if hasattr(main_base, 'transformer'):
                    #     # GPT-2 needs position embeddings added before layers
                    #     position_embeds = transformer.wpe(position_ids)
                    #     hidden_states = transformer.drop(hidden_states + position_embeds)
                    
                    # # # Compute rotary embeddings for LLaMA if needed
                    # # if hasattr(main_base, 'model') and hasattr(transformer, 'rotary_emb'):
                    # #     # LLaMA uses rotary embeddings for the new position
                    # #     cos, sin = transformer.rotary_emb(hidden_states, position_ids)
                    # #     position_embeddings = (cos, sin)
                    # # else:
                    # #     position_embeddings = None
                    
                    # # Process through layers using native KV caching
                    # new_past_key_values = []
                    # for layer_idx, (layer_module, past_kv) in enumerate(zip(layers, past_key_values)):
                    #     if hasattr(main_base, 'transformer'):
                    #         # GPT-2 - position embeddings already added, don't pass position_ids
                    #         layer_outputs = layer_module(
                    #             hidden_states,
                    #             past_key_value=past_kv,
                    #             use_cache=True
                    #         )
                    #     else:
                    #         # LLaMA - needs position_ids for RoPE
                    #         layer_outputs = layer_module(
                    #             hidden_states,
                    #             past_key_value=past_kv,
                    #             position_ids=position_ids,
                    #             use_cache=True
                    #         )
                    #     hidden_states = layer_outputs[0]
                        
                    #     # Update past_key_values
                    #     #if len(layer_outputs) > 1 and layer_outputs[1] is not None:
                    #     # FORONE
                    #     new_past_key_values[layer_idx][0].append(layer_outputs[1][0])#layer, keyval, tokenindex? maybe?
                    #     new_past_key_values[layer_idx][1].append(layer_outputs[1][1])
                    #     print("HELP")
                    #     print(new_past_key_values[layer_idx][0])
                    #     print(new_past_key_values[layer_idx][1])
                    #     print(layer_outputs[1][0].shape)
                    #     print(layer_outputs[1][1].shape)
                    #     print(layer_outputs[1][0][0].shape)
                    #     print(layer_outputs[1][1][0].shape)
                    #     print(layer_outputs[1][0][0][0].shape)
                    #     print(layer_outputs[1][1][0][0].shape)
                    #     exit()
                    
                    # # Final layer norm
                    # hidden_states = final_norm(hidden_states)
                    
                    # # Update past_key_values for next iteration
                    # past_key_values = tuple(new_past_key_values)
                    current_position += 1
                else:
                    # Original behavior - use native caching
                    with self._maybe_disable_dropout():
                        outputs = main_base(
                            inputs_embeds=emb_t_input.unsqueeze(1),
                            past_key_values=past_key_values,
                            use_cache=True,
                            output_hidden_states=True
                        )
                    hidden_states = outputs.hidden_states[-1]
                    past_key_values = outputs.past_key_values
                    current_position += 1
                    #print("HELPHELPHELPHELP")
        
        # Stack outputs
        logits_seq = torch.stack(logits_list, dim=1)
        hard_ids = torch.stack(hard_ids_list, dim=1)
        text_embs = torch.stack(output_embs_list, dim=1)
        
        return Generated(text_embs, logits_seq, hard_ids)
    
    def generate_soft_kv_flash(
        self,
        activation_input,
        max_length=64,
        gumbel_tau=1.0,
        hard_left_emb=None,
        hard_right_emb=None,
        print_prompt=False,
        override_model_base_and_out=None,
    ):
        """Generate soft text using Flash Attention with KV caching.
        
        This method combines Flash Attention's optimized computation with
        KV caching for O(n) generation complexity.
        
        Args:
            Same as generate_soft
        
        Returns:
            Same as generate_soft
        """
        from lens.models.flash_kv_cache_v2 import FlashKVCache, compute_with_flash_kv_cache, FLASH_AVAILABLE
        
        if not FLASH_AVAILABLE:
            raise RuntimeError(
                "Flash Attention requested but not available. "
                "Install with: make flash-attention"
            )
        
        # Setup similar to generate_soft_kv_cached
        if print_prompt and hasattr(self, 'prompt_text'):
            print(f"Prompt template: {self.prompt_text}")

        if override_model_base_and_out is not None:
            main_model = override_model_base_and_out
            # Handle both OrigWrapper (has .model) and raw model cases
            if hasattr(main_model, 'model'):
                main_base = main_model.model
                main_out = main_model.model.lm_head if hasattr(main_model.model, 'lm_head') else main_model.model.get_output_embeddings()
            else:
                # Direct model passed (e.g., GPT2LMHeadModel)
                main_base = main_model
                main_out = main_model.lm_head if hasattr(main_model, 'lm_head') else main_model.get_output_embeddings()
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
        
        # Get embedding tables
        input_emb_table = main_base.get_input_embeddings().weight
        
        # Project activation to embedding space
        emb_a = self.proj(activation_input)
        
        # Build initial prompt: [left_prompt, projected_activation, right_prompt]
        left_prompt_embs = prompt_left_emb.unsqueeze(0).expand(B, -1, -1)
        right_prompt_embs = prompt_right_emb.unsqueeze(0).expand(B, -1, -1)
        prompt_embs = torch.cat([left_prompt_embs, emb_a.unsqueeze(1), right_prompt_embs], dim=1)
        
        # Get transformer backbone
        transformer = main_base.transformer if hasattr(main_base, 'transformer') else main_base
        
        # Process initial prompt through transformer with Flash Attention
        hidden_states, kv_cache = compute_with_flash_kv_cache(
            transformer, 
            prompt_embs,
            use_cache=True
        )
        
        # Get logits for last position
        logits = main_out(hidden_states[:, -1])
        
        # Start autoregressive generation
        current_position = prompt_embs.size(1)
        logits_list = []
        hard_ids_list = []
        output_embs_list = []
        
        for step in range(max_length):
            # Apply Gumbel-Softmax
            if gumbel_tau > 0:
                # For Flash Attention, avoid casting to float32 as it only supports fp16/bf16
                # Subtract max for numerical stability (detached)
                logits_stable = logits - logits.max(dim=-1, keepdim=True)[0].detach()
                ste_token_dist = torch.nn.functional.gumbel_softmax(
                    logits_stable, tau=max(gumbel_tau, 0.1), hard=True
                )
            else:
                # Straight-through estimator with hard argmax
                # For Flash Attention, avoid casting to float32
                # Subtract max for numerical stability (detached)
                logits_stable = logits - logits.max(dim=-1, keepdim=True)[0].detach()
                probs = torch.nn.functional.softmax(logits_stable, dim=-1)
                hard_indices = probs.argmax(dim=-1)
                ste_token_dist = torch.zeros_like(probs)
                ste_token_dist.scatter_(-1, hard_indices.unsqueeze(-1), 1.0)
                ste_token_dist = ste_token_dist - probs.detach() + probs
            
            # Get embedding using Gumbel weights
            emb_t_input = ste_token_dist @ input_emb_table
            
            # Store outputs
            logits_list.append(logits)
            output_embs_list.append(emb_t_input)
            hard_ids_list.append(ste_token_dist.argmax(dim=-1))
            
            # Process new token through transformer with Flash KV cache
            if step < max_length - 1:  # Don't process last token if we won't use it
                hidden_states, kv_cache = compute_with_flash_kv_cache(
                    transformer,
                    emb_t_input.unsqueeze(1), 
                    kv_cache,
                    position_offset=current_position
                )
                current_position += 1
                
                # Get logits for next token
                logits = main_out(hidden_states[:, -1])
        
        # Stack outputs
        logits_seq = torch.stack(logits_list, dim=1)
        hard_ids = torch.stack(hard_ids_list, dim=1)
        text_embs = torch.stack(output_embs_list, dim=1)
        
        return Generated(text_embs, logits_seq, hard_ids)
