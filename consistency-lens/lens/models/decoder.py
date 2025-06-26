"""Decoder with Gumbel-Softmax generation over projection-injected activation."""

from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, Tuple, Dict
from contextlib import contextmanager
from contextlib import nullcontext
import copy

import torch
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.cache_utils import DynamicCache, Cache
from transformers.cache_utils import HybridCache
import logging
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
__all__ = ["Decoder", "Generated"]

log = logging.getLogger(__name__)
class Generated(NamedTuple):
    generated_text_embeddings: torch.Tensor
    raw_lm_logits: torch.Tensor
    hard_token_ids: torch.Tensor

    def detach(self):
        return Generated(
            self.generated_text_embeddings.detach(),
            self.raw_lm_logits.detach(),
            self.hard_token_ids.detach()
        )

    def cpu(self):
        return Generated(
            self.generated_text_embeddings.cpu(),
            self.raw_lm_logits.cpu(),
            self.hard_token_ids.cpu()
        )

    def clone(self):
        return Generated(
            self.generated_text_embeddings.clone(),
            self.raw_lm_logits.clone(),
            self.hard_token_ids.clone()
        )

@dataclass
class DecoderConfig:
    model_name: str
    n_prompt_tokens: int = 8
    base_model: bool = False         # YAML `base_model`
    projection_layer: bool = True    # YAML `projection_layer`
    output_head: bool = True         # YAML `output_head`
    embedding_head: bool = False     # YAML `embedding_head`
    pos_embeddings: bool = False     # YAML `pos_embeddings`
    eye_init: bool = True            # YAML `eye_init`
    trainable_prompts: bool = True   # YAML `trainable_prompts`
    use_checkpointing: bool = False # YAML `use_checkpointing`
    checkpoint_every_n_tokens: int = 4 # YAML `checkpoint_every_n_tokens`
    use_kv_cache: bool = False  # YAML `use_kv_cache`
    use_flash_attention: bool = False  # YAML `use_flash_attention`
    use_gumbel_for_LMorig: bool = False # YAML `use_gumbel_for_LMorig`
    patch_all_layers: Any = False   # YAML `patch_all_layers` - patch activation at all layers (True) or N layers (int)
    per_layer_projections: bool = False  # YAML `per_layer_projections` - use separate projection for each layer
    use_dropout: bool = True         # YAML `use_dropout` - whether to use dropout during training (False = deterministic)
    detach_after_each_sample: bool = False
    end_to_end: bool = True
    projection_init_method: str = "default"  # YAML `projection_init_method` - how to initialize the projection layer
    projection_init_tl_checkpoint: str = None  # e.g., "path/to/tuned_lens_checkpoint" or "username/model-tuned-lens"
    projection_init_tl_model_name: str = None
    projection_init_tl_comp_src_layer: int = None  # e.g., 5
    projection_init_tl_comp_tgt_layer: int = None  # e.g., 1
    projection_init_tl_comp_use_composed_bias: bool = True
    subtract_add_pos_embeddings: bool = False
    extra_pos_embeddings: bool = True
    clamp_entropy: float = 0.8
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_prompt_tokens < 0:
            raise ValueError(f"n_prompt_tokens must be non-negative, got {self.n_prompt_tokens}")
        if self.per_layer_projections and not self.patch_all_layers:
            raise ValueError("per_layer_projections requires patch_all_layers to be True or a positive integer")
        if isinstance(self.patch_all_layers, int) and self.patch_all_layers < 0:
            raise ValueError("If patch_all_layers is an integer, it must be non-negative.")


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

    def __init__(self, cfg: DecoderConfig, base_to_use: Optional[PreTrainedModel] = None) -> None:
        super().__init__()
        
        # Use provided base model or load a new one
        if base_to_use is not None and not cfg.base_model:
            # If we're not training the base model and one is provided, use it
            self.base = base_to_use
            log.info("Using shared base model for Decoder (memory efficient)")
        else:
            # Load our own copy
            self.base: PreTrainedModel = AutoModelForCausalLM.from_pretrained(cfg.model_name)
            
        # Configure trainability of the base model
        for p in self.base.parameters():
            p.requires_grad_(cfg.base_model)

        # Set requires_grad for positional embeddings if present in base model
        if hasattr(self.base, "transformer") and hasattr(self.base.transformer, "wpe"):
            self.base.transformer.wpe.weight.requires_grad_(cfg.pos_embeddings)
            log.info("Set requires_grad for positional embeddings in base model (wpe)")
        elif hasattr(self.base, "model") and hasattr(self.base.model, "embed_positions"):
            self.base.model.embed_positions.weight.requires_grad_(cfg.pos_embeddings)
            log.info("Set requires_grad for positional embeddings in base model (embed_positions)")

        self.config = cfg
        d_model = self.base.config.hidden_size
        n_layers_base = self.base.config.num_hidden_layers
        
        # Create projection layer(s)
        if cfg.per_layer_projections:
            if isinstance(cfg.patch_all_layers, bool):
                n_proj_layers = n_layers_base if cfg.patch_all_layers else 0
            else: # is an int
                n_proj_layers = cfg.patch_all_layers

            if n_proj_layers > 0:
                # Create a 3D parameter tensor for per-layer projections
                # Shape: (n_proj_layers, d_model, d_model)
                self.proj_weight = nn.Parameter(torch.empty(n_proj_layers, d_model, d_model))
                self.proj_bias = nn.Parameter(torch.empty(n_proj_layers, d_model))
                
                # Initialize as identity matrices
                if cfg.eye_init:
                    for i in range(n_proj_layers):
                        nn.init.eye_(self.proj_weight[i])
                    nn.init.zeros_(self.proj_bias)
                    log.info(f"Initialized {n_proj_layers} per-layer projection matrices as identity (decoder)")
                else:
                    # Default initialization
                    for i in range(n_proj_layers):
                        nn.init.xavier_uniform_(self.proj_weight[i])
                    nn.init.zeros_(self.proj_bias)
                    log.info(f"Initialized {n_proj_layers} per-layer projection matrices as random (decoder)")
                
                # Configure trainability
                self.proj_weight.requires_grad_(cfg.projection_layer)
                self.proj_bias.requires_grad_(cfg.projection_layer)
            else:
                self.proj_weight = None
                self.proj_bias = None
            
            # For compatibility, keep self.proj as None
            self.proj = None
        else:
            # Single projection layer (original behavior)
            self.proj = nn.Linear(d_model, d_model, bias=True)
            # Initialize as identity matrix
            if cfg.eye_init:
                nn.init.eye_(self.proj.weight)
                nn.init.zeros_(self.proj.bias)
                log.info("Initialized projection layer as identity matrix (decoder)")
            # Configure trainability of the input projection layer
            for p in self.proj.parameters():
                p.requires_grad_(cfg.projection_layer)

        # The output head (self.out) maps hidden states to vocabulary logits.
        # This is essential for the Decoder to generate text.
        # If self.base is frozen, self.out allows the Decoder to adapt the
        # (frozen) base model\'s representations for the explanation generation task.
        # If self.base is trainable, self.out is trained along with it.
        # Initialise `self.out` with a copy of the original unembedding matrix
        # (i.e. the tied output embedding / LM head from the base model).
        if cfg.output_head:
            raise ValueError("output_head is not supported for now")
                #self.out = nn.Linear(d_model, self.base.config.vocab_size, bias=False)
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
        

        input_emb_table = self.base.get_input_embeddings().weight
        output_emb_table = self.base.get_output_embeddings().weight
        embeddings_tied = (input_emb_table.data_ptr() == output_emb_table.data_ptr())

        if not embeddings_tied:
            try:
                output_embeddings = self.base.get_output_embeddings()
                if output_embeddings is not None:
                    for p in output_embeddings.parameters():
                        p.requires_grad_(cfg.output_embedding_head)
            except AttributeError:
                # Fallback for models that expose `lm_head`
                if hasattr(self.base, 'lm_head'):
                    for p in self.base.lm_head.parameters():
                        p.requires_grad_(cfg.embedding_head)
                else:
                    log.warning("Could not access output embeddings for freezing control")
            
        #if not cfg.embedding_head and not cfg.output_head:
        #    embed_prod_out = self.out.weight @ self.base.get_output_embeddings().weight.T
        
        # --- Prompt placeholders -------------------------------------------------
        self.prompt_left_emb = None      # type: nn.Parameter | None
        self.prompt_right_emb = None     # type: nn.Parameter | None
        self.prompt_len = 0
        self.prompt_text = []
        # keep prompt_ids only for logging/debug convenience
        self.register_buffer("prompt_ids", torch.empty(0, dtype=torch.long))

        # Activation-specific positional embedder
        self.activation_pos_embedder = None
        if self.config.subtract_add_pos_embeddings:
            d_model = self.base.config.hidden_size
            num_pos_embeddings = self.base.config.max_position_embeddings
            self.activation_pos_embedder = nn.Embedding(num_pos_embeddings, d_model)
            # Initialize weights (e.g., small random values or from base model's pos embeddings)
            self.activation_pos_embedder.weight.data = self.base.transformer.wpe.weight.data.clone()
            self.activation_pos_embedder.weight.requires_grad_(cfg.extra_pos_embeddings) # Ensure trainable
            log.info(f"Initialized activation positional embedder for Decoder with {num_pos_embeddings} embeddings.")


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
        n_layers_base = self.base.config.num_hidden_layers
        
        # if False: self.out.out_features != vocab_size:
        #     pass
        #     #raise ValueError("trainable output_head is not supported for now")
        #     self.out = nn.Linear(d_model, vocab_size, bias=False).to(old_device)
        #     # Initialize with new model's output embeddings
        #     with torch.no_grad():
        #         try:
        #             orig_out_w = self.base.get_output_embeddings().weight
        #         except AttributeError:
        #             orig_out_w = self.base.lm_head.weight
        #         if orig_out_w.shape == self.out.weight.shape:
        #             self.out.weight.copy_(orig_out_w)
            
        #     for p in self.out.parameters():
        #         p.requires_grad_(self.config.output_head)
        
        # Restore or reinitialize projection weights
        if self.config.per_layer_projections:
            if isinstance(self.config.patch_all_layers, bool):
                n_proj_layers = n_layers_base if self.config.patch_all_layers else 0
            else:  # is an int
                n_proj_layers = self.config.patch_all_layers

            if keep_projection and self.proj_weight is not None and old_proj_weight.shape == (n_proj_layers, d_model, d_model):
                self.proj_weight.data = old_proj_weight.to(old_dtype)
                self.proj_bias.data = old_proj_bias.to(old_dtype)
            elif n_proj_layers > 0:
                # Reinitialize if dimensions changed
                self.proj_weight = nn.Parameter(torch.empty(n_proj_layers, d_model, d_model).to(old_device))
                self.proj_bias = nn.Parameter(torch.empty(n_proj_layers, d_model).to(old_device))
                if self.config.eye_init:
                    for i in range(n_proj_layers):
                        nn.init.eye_(self.proj_weight[i])
                    nn.init.zeros_(self.proj_bias)
                self.proj_weight.requires_grad_(self.config.projection_layer)
                self.proj_bias.requires_grad_(self.config.projection_layer)
            else:
                self.proj_weight = None
                self.proj_bias = None
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
    def _patched_forward(
        self,
        main_base,
        seq_embs: torch.Tensor,
        activation_input_modified: torch.Tensor,
        use_projection: bool,
        do_patching: bool,
        prompt_left_emb: Optional[torch.Tensor],
        past_key_values: Optional[Any] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        max_total_len_for_cache: Optional[int] = None,
        return_past_key_values: bool = False,
    ) -> tuple[torch.Tensor, Optional[Any]]:
        """
        Performs a forward pass through the model, with activation patching at each layer.
        This is used when `self.config.patch_all_layers` is True.
        Supports KV caching for GPT-2, LLaMA, and Gemma-2 style models.
        """
        B = seq_embs.shape[0]
        device = seq_embs.device
        # Use a consistent and robust check for Gemma-2
        is_gemma2 = (hasattr(main_base.config, 'model_type') and 
                     isinstance(main_base.config.model_type, str) and 
                     "gemma2" in main_base.config.model_type.lower()) # Match Gemma-2 name, e.g. "gemma2" or "gemma-2-9b"

        # Pre-compute single projection if not using per-layer projections
        if not self.config.per_layer_projections:
            if use_projection:
                single_proj = self._apply_projection(activation_input_modified)
            else:
                single_proj = activation_input_modified
        
        # Calculate embed position (where activation should be patched)
        embed_pos = prompt_left_emb.size(0) if prompt_left_emb is not None else 0
        
        hidden_states = seq_embs
        seq_length = hidden_states.size(1)
        
        # Prepare position IDs, using cache_position if available for KV caching
        if cache_position is not None:
            position_ids = cache_position.unsqueeze(0)
        else:
            # If cache_position is not provided, this implies it's the first pass or caching is not used.
            # The past_seen_tokens for mask creation will be 0.
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)

        # --- Model-specific forward pass ---

        if hasattr(main_base, "transformer") and main_base.config.model_type != "gemma2": # Check it's not Gemma2 disguised as GPT-2
            # --- GPT-2 Style ---
            transformer = main_base.transformer # This is GPT2Model
            layers      = transformer.h
            final_norm  = transformer.ln_f
            # ... (GPT-2 specific cache, mask, and layer loop as previously corrected) ...
            # ---- KV-cache + causal mask (GPT-2) ------------------------------------
            if use_cache and past_key_values is None:
                past_key_values = DynamicCache()

            # For GPT-2, cache_position for _update_causal_mask is distinct from position_ids for wpe
            gpt2_cache_position_for_mask = cache_position
            if gpt2_cache_position_for_mask is None: # Should align with how position_ids was set if cache_position was None
                past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
                gpt2_cache_position_for_mask = torch.arange(past_seen, past_seen + seq_length, device=device, dtype=torch.long)


            # Use the _update_causal_mask from the GPT2Model instance (main_base.transformer)
            causal_mask = transformer._update_causal_mask(
                attention_mask, # User-provided 2D mask or None
                hidden_states,  # Current hidden_states for shape/dtype
                gpt2_cache_position_for_mask, # Absolute positions for mask generation
                past_key_values,
                output_attentions=False, # Assuming False for internal processing
            )
            # ----------------------------------------------------------------------

            position_embeds = transformer.wpe(position_ids) # Use the originally computed position_ids
            hidden_states   = transformer.drop(hidden_states + position_embeds)

            for layer_idx, layer_module in enumerate(layers):
                input_to_this_layer = hidden_states.clone()
                # ... existing patch-in-place logic ...
                if do_patching:
                    # Determine if we should patch this layer
                    should_patch_this_layer = False
                    if isinstance(self.config.patch_all_layers, bool) and self.config.patch_all_layers:
                        should_patch_this_layer = True
                    elif isinstance(self.config.patch_all_layers, int) and layer_idx < self.config.patch_all_layers:
                        should_patch_this_layer = True

                    if should_patch_this_layer:
                        if self.config.per_layer_projections:
                            if use_projection:
                                a_proj_layer = self._apply_projection(activation_input_modified, layer_idx=layer_idx)
                            else:
                                a_proj_layer = activation_input_modified
                            input_to_this_layer[:, embed_pos] = a_proj_layer
                        else:
                            input_to_this_layer[:, embed_pos] = single_proj
                
                with self._maybe_disable_dropout():
                    layer_outputs = layer_module(
                        input_to_this_layer,
                        past_key_value=past_key_values,
                        cache_position=gpt2_cache_position_for_mask, # Pass the cache_position for KV updates
                        attention_mask=causal_mask,
                        use_cache=use_cache,
                        head_mask=None,
                    )
                hidden_states = layer_outputs[0]


        elif is_gemma2: # Matches "gemma2" in main_base.config.model_type
            # --- Gemma-2 Style ---
            # main_base is likely Gemma2ForCausalLM, so main_base.model is Gemma2Model
            gemma2_model = main_base.model 
            layers = gemma2_model.layers
            final_norm = gemma2_model.norm

            if use_cache and past_key_values is None:
                # Gemma2 uses HybridCache, but can also work with DynamicCache if that's what's passed.
                # Standard Gemma2Model.forward initializes DynamicCache if past_key_values is None and use_cache.
                # If _patched_forward is expected to create a Gemma2-specific HybridCache, that logic
                # needs to be here. For now, let's assume it can accept a DynamicCache or HybridCache.
                # The generate_soft_kv_cached already initializes HybridCache for Gemma2 if needed.
                if isinstance(past_key_values, HybridCache): # type: ignore
                     pass # Already a HybridCache
                elif past_key_values is None: # Create DynamicCache if nothing is passed and HybridCache not made by caller
                    past_key_values = DynamicCache()


            # For Gemma-2, cache_position for mask creation is crucial.
            # If not provided by caller, compute it based on past_seen_tokens.
            gemma2_cache_position = cache_position
            if gemma2_cache_position is None:
                 past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                 gemma2_cache_position = torch.arange(
                     past_seen_tokens, past_seen_tokens + seq_length, device=device, dtype=torch.long
                 )
            
            # Prepare mask arguments for Gemma-2 utilities
            mask_creation_kwargs = {
                "config": gemma2_model.config, # Use Gemma2Model's config
                "input_embeds": hidden_states,
                "attention_mask": attention_mask, # User-provided 2D mask or None
                "cache_position": gemma2_cache_position,
                "past_key_values": past_key_values,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_creation_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_creation_kwargs),
            }

            position_embeddings = gemma2_model.rotary_emb(hidden_states, position_ids) # Use originally computed position_ids

            normalizer = torch.tensor(gemma2_model.config.hidden_size**0.5, dtype=hidden_states.dtype)
            hidden_states = hidden_states * normalizer

            for layer_idx, layer_module in enumerate(layers):
                input_to_this_layer = hidden_states.clone()
                # ... existing patching logic ...
                if do_patching:
                    # Determine if we should patch this layer
                    should_patch_this_layer = False
                    if isinstance(self.config.patch_all_layers, bool) and self.config.patch_all_layers:
                        should_patch_this_layer = True
                    elif isinstance(self.config.patch_all_layers, int) and layer_idx < self.config.patch_all_layers:
                        should_patch_this_layer = True

                    if should_patch_this_layer:
                        if self.config.per_layer_projections:
                            if use_projection:
                                a_proj_layer = self._apply_projection(activation_input_modified, layer_idx=layer_idx)
                            else:
                                a_proj_layer = activation_input_modified
                            input_to_this_layer[:, embed_pos] = a_proj_layer
                        else:
                            input_to_this_layer[:, embed_pos] = single_proj

                # Select the correct mask based on the layer's attention type
                current_attention_mask = causal_mask_mapping[layer_module.attention_type]

                with self._maybe_disable_dropout():
                    layer_outputs = layer_module(
                        input_to_this_layer,
                        position_embeddings=position_embeddings,
                        attention_mask=current_attention_mask, # Pass the selected 4D mask
                        position_ids=position_ids, # Pass original position_ids for RoPE in attention
                        past_key_value=past_key_values,
                        use_cache=use_cache,
                        cache_position=gemma2_cache_position, # Pass the absolute cache_position
                    )
                hidden_states = layer_outputs[0]
        
        elif hasattr(main_base, 'model'): # Fallback for LLaMA-style if not Gemma2
            # --- LLaMA Style ---
            # ... (LLaMA specific cache, mask, and layer loop, similar to Gemma2 but using LlamaModel's _update_causal_mask)
            llama_model = main_base.model # This is LlamaModel
            layers = llama_model.layers
            final_norm = llama_model.norm

            if use_cache and past_key_values is None:
                past_key_values = DynamicCache()
            
            llama_cache_position = cache_position
            if llama_cache_position is None:
                past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
                llama_cache_position = torch.arange(past_seen, past_seen + seq_length, device=device, dtype=torch.long)

            # Use LlamaModel's _update_causal_mask
            processed_attention_mask = llama_model._update_causal_mask(
                attention_mask=attention_mask,
                input_tensor=hidden_states,
                cache_position=llama_cache_position,
                past_key_values=past_key_values,
                output_attentions=False 
            )
            
            # LLaMA uses rotary embeddings passed explicitly to layers
            # position_ids are computed at the start of _patched_forward
            #cos, sin = llama_model.rotary_emb(hidden_states, llama_cache_position) # RoPE uses cache_position
            #llama_position_embeddings = (cos, sin)


            for layer_idx, layer_module in enumerate(layers):
                input_to_this_layer = hidden_states.clone()
                # ... existing patching logic ...
                if do_patching and (layer_idx > 0 or not self.config.per_layer_projections):
                    if self.config.per_layer_projections:
                        if use_projection:
                            a_proj_layer = self._apply_projection(activation_input_modified, layer_idx=layer_idx)
                        else:
                            a_proj_layer = activation_input_modified
                        input_to_this_layer[:, embed_pos] = a_proj_layer
                    else:
                         input_to_this_layer[:, embed_pos] = single_proj
                
                with self._maybe_disable_dropout():
                    layer_outputs = layer_module(
                        input_to_this_layer,
                        attention_mask=processed_attention_mask, # Pass the processed 4D mask
                        position_ids=position_ids, # Pass original position_ids
                        past_key_value=past_key_values,
                        use_cache=use_cache,
                        cache_position=llama_cache_position, # Pass absolute cache_position
                        # LLaMA attention layer takes position_embeddings directly
                        # but it's computed from hidden_states and position_ids (via cache_position) by the LlamaAttention module itself
                        # The standard LlamaDecoderLayer doesn't take position_embeddings as a direct kwarg.
                        # Instead, LlamaAttention computes it using self.rotary_emb(hidden_states, position_ids=position_ids)
                        # So we pass position_ids, and the layer handles RoPE internally.
                        # For Llama, rotary_emb is usually applied *inside* the attention module.
                        # The (cos, sin) computed above are for the *entire sequence* if done like Gemma2.
                        # Let's stick to passing position_ids and let the LlamaAttention module apply RoPE.
                        # If LlamaDecoderLayer expects (cos, sin) tuple, then:
                        # position_embeddings=llama_position_embeddings 
                    )
                hidden_states = layer_outputs[0]

        # Final layer norm
        hidden_states = final_norm(hidden_states)
        if return_past_key_values:
            return hidden_states, past_key_values
        else:
            return hidden_states

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
        special_token = None,
        original_token_pos: Optional[torch.Tensor] = None,
        add_tokens: list[int] = None,
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
            main_out = self.base.get_output_embeddings()

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
        
        # Subtract positional embedding from activation_input if configured
        activation_input_modified = activation_input
        if self.config.subtract_add_pos_embeddings and original_token_pos is not None and self.activation_pos_embedder is not None:
            original_token_pos = original_token_pos.to(device=device, dtype=torch.long)
            original_token_pos = original_token_pos.squeeze() if original_token_pos.dim() == 2 else original_token_pos
            
            pos_embeds_to_subtract = self.activation_pos_embedder(original_token_pos)
            
            if pos_embeds_to_subtract.shape != activation_input.shape:
                raise ValueError(
                    f"Shape mismatch for positional embedding subtraction: "
                    f"activation_input shape {activation_input.shape}, "
                    f"pos_embeds_to_subtract shape {pos_embeds_to_subtract.shape}"
                )
            activation_input_modified = activation_input - pos_embeds_to_subtract
        
        # Get both input and output embedding tables
        input_emb_table = main_base.get_input_embeddings().weight
        output_emb_table = main_base.get_output_embeddings().weight
        embeddings_tied = (input_emb_table.data_ptr() == output_emb_table.data_ptr())
        
        # Check if embeddings are tied (same memory location)
        if not embeddings_tied or self.config.output_head:
            raise ValueError("Embeddings are not tied or output head is trainable - this is not supported")
        

        # 0) prepend textual prompt (pre-computed at set_prompt)
        parts = []
        if prompt_left_emb is not None:
            parts.append(prompt_left_emb.expand(B, -1, -1))
        
        # Always insert activation as a token
        if use_projection:
            if self.config.patch_all_layers and self.config.per_layer_projections:
                # Use first layer's projection
                a_proj = self._apply_projection(activation_input_modified, layer_idx=0).unsqueeze(1)
            else:
                # Use single projection
                a_proj = self._apply_projection(activation_input_modified).unsqueeze(1)
        else:
            # No projection
            a_proj = activation_input_modified.unsqueeze(1)
        if do_patching: 
            parts.append(a_proj)
        else: 
            print("patching in special token")
            parts.append(main_base.get_input_embeddings().weight[special_token.squeeze().item() if isinstance(special_token, torch.Tensor) else special_token].clone().unsqueeze(0).repeat(B,1,1))
            
        if prompt_right_emb is not None:
            parts.append(prompt_right_emb.expand(B, -1, -1))
        if add_tokens is not None:
            token_ids_tensor = torch.tensor(
                add_tokens, dtype=torch.long, device=input_emb_table.device
            )
            added_token_embeddings = input_emb_table[token_ids_tensor].unsqueeze(0).expand(B, -1, -1)
            parts.append(added_token_embeddings)
            
        seq_embs = torch.cat(parts, dim=1)

        logits_list = []
        hard_ids_list = []
        output_embs_list = []  # Store embeddings for encoder

        for _ in range(max_length):
            if self.config.patch_all_layers:
                # Custom forward pass with activation patching at all layers
                h_last = self._patched_forward(
                    main_base=main_base,
                    seq_embs=seq_embs,
                    activation_input_modified=activation_input_modified,
                    use_projection=use_projection,
                    do_patching=do_patching,
                    prompt_left_emb=prompt_left_emb,
                    return_past_key_values=False,
                    use_cache=False,
                )
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
    
    def check_for_nans(self) -> bool:
        """Check decoder parameters for NaN values. Returns True if NaN found."""
        nan_found = False
        
        # Check projection weights
        if self.config.per_layer_projections:
            if torch.isnan(self.proj_weight).any():
                log.error(f"NaN detected in per-layer projection weights")
                nan_found = True
            if torch.isnan(self.proj_bias).any():
                log.error(f"NaN detected in per-layer projection bias")
                nan_found = True
        elif self.proj is not None:
            if torch.isnan(self.proj.weight).any():
                log.error(f"NaN detected in projection weights")
                nan_found = True
            if torch.isnan(self.proj.bias).any():
                log.error(f"NaN detected in projection bias")
                nan_found = True
        
        # Check output head
        if torch.isnan(self.base.get_output_embeddings().weight).any():
            log.error(f"NaN detected in output head weights")
            nan_found = True
        
        # Check prompt embeddings
        if hasattr(self, 'prompt_left_emb') and self.prompt_left_emb is not None:
            if torch.isnan(self.prompt_left_emb).any():
                log.error(f"NaN detected in prompt_left_emb")
                nan_found = True
        if hasattr(self, 'prompt_right_emb') and self.prompt_right_emb is not None:
            if torch.isnan(self.prompt_right_emb).any():
                log.error(f"NaN detected in prompt_right_emb")
                nan_found = True
        
        # Check activation positional embedder if present
        if self.activation_pos_embedder is not None:
            if torch.isnan(self.activation_pos_embedder.weight).any():
                log.error(f"NaN detected in activation_pos_embedder weights")
                nan_found = True
        
        # Check base model input embeddings
        try:
            input_emb_weight = self.base.get_input_embeddings().weight
            if torch.isnan(input_emb_weight).any():
                log.error(f"NaN detected in base model input embeddings")
                nan_found = True
        except Exception as e:
            log.warning(f"Could not check base model input embeddings: {e}")
        
        return nan_found
    
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
        special_token = None,
        original_token_pos: Optional[torch.Tensor] = None,
    ) -> Generated:
        """Differentiable generation with KV caching for O(n) attention computation.
        
        This method produces identical results to generate_soft but avoids
        recomputing attention for past tokens by caching their key/value projections.
        Works with GPT-2, LLaMA, and Gemma2 architectures.
        
        Args:
            Same as generate_soft
            
        Returns:
            Same as generate_soft
        """

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
            main_out = self.base.get_output_embeddings()

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
        
        # Detect if this is a Gemma2 model
        is_gemma2 = (hasattr(main_base.config, 'model_type') and 
                     isinstance(main_base.config.model_type, str) and 
                     "gemma2" in main_base.config.model_type.lower())
        
        # Subtract positional embedding from activation_input if configured
        activation_input_modified = activation_input
        if self.config.subtract_add_pos_embeddings and original_token_pos is not None and self.activation_pos_embedder is not None:
            original_token_pos = original_token_pos.to(device=device, dtype=torch.long)
            original_token_pos = original_token_pos.squeeze() if original_token_pos.dim() == 2 else original_token_pos
            
            pos_embeds_to_subtract = self.activation_pos_embedder(original_token_pos)
            
            if pos_embeds_to_subtract.shape != activation_input.shape:
                if pos_embeds_to_subtract.shape[0] == 1 and activation_input.shape[0] > 1:
                    pos_embeds_to_subtract = pos_embeds_to_subtract.expand_as(activation_input)
                else:
                     raise ValueError(
                        f"Shape mismatch for positional embedding subtraction: "
                        f"activation_input shape {activation_input.shape}, "
                        f"pos_embeds_to_subtract shape {pos_embeds_to_subtract.shape}"
                    )
            activation_input_modified = activation_input - pos_embeds_to_subtract
        
        # Get embedding tables
        input_emb_table = main_base.get_input_embeddings().weight
        output_emb_table = main_base.get_output_embeddings().weight
        embeddings_tied = (input_emb_table.data_ptr() == output_emb_table.data_ptr())
        #  if not embeddings_tied or self.config.output_head:
        #     raise ValueError("Embeddings are not tied or output head is trainable - this is not supported")

        
        # Prepare initial sequence
        parts = []
        if prompt_left_emb is not None:
            parts.append(prompt_left_emb.expand(B, -1, -1))
        
        # Always insert activation as a token
        if use_projection:
            if self.config.patch_all_layers and self.config.per_layer_projections:
                # Use first layer's projection
                a_proj = self._apply_projection(activation_input_modified, layer_idx=0).unsqueeze(1)
            else:
                # Use single projection
                a_proj = self._apply_projection(activation_input_modified).unsqueeze(1)
        else:
            # No projection
            a_proj = activation_input_modified.unsqueeze(1)
        if do_patching: 
            parts.append(a_proj)
        else: 
            print("patching in special token")
            parts.append(main_base.get_input_embeddings().weight[special_token.squeeze().item() if isinstance(special_token, torch.Tensor) else special_token].clone().unsqueeze(0).repeat(B,1,1))
            
        if prompt_right_emb is not None:
            parts.append(prompt_right_emb.expand(B, -1, -1))
            
        seq_embs = torch.cat(parts, dim=1)
        
        # Initialize storage
        logits_list = []
        hard_ids_list = []
        output_embs_list = []
        
        # Storage for past_key_values
        past_key_values = None
        
        # Initialize cache_position for Gemma2
        if is_gemma2:
            cache_position = torch.arange(seq_embs.size(1), device=device, dtype=torch.long)
        
        # Get the transformer and detect architecture
        if hasattr(main_base, 'transformer'):
            # GPT-2 style model
            transformer = main_base.transformer
            layers = transformer.h
            final_norm = transformer.ln_f
        elif hasattr(main_base, 'model'):
            # LLaMA/Gemma style model (model.layers)
            transformer = main_base.model
            layers = transformer.layers
            final_norm = transformer.norm
        else:
            raise ValueError(f"Unknown model architecture. Expected transformer or model attribute.")
        
        # Process initial sequence (prompt + activation)
        if self.config.patch_all_layers:
            # Use _patched_forward for patching logic
            # Determine max_total_len for cache, needed if _patched_forward creates it
            max_total_length_for_cache = seq_embs.size(1) + max_length

            hidden_states, past_key_values = self._patched_forward(
                main_base=main_base,
                seq_embs=seq_embs,
                activation_input_modified=activation_input_modified,
                use_projection=use_projection,
                do_patching=do_patching,
                prompt_left_emb=prompt_left_emb,
                past_key_values=past_key_values, # Initially None
                use_cache=True, # KV caching is the point of this function
                attention_mask=None, # Assuming causal mask is handled internally by model
                #cache_position=cache_position if is_gemma2 else None,
                max_total_len_for_cache=max_total_length_for_cache if is_gemma2 else None, # Pass for Gemma2
                return_past_key_values=True,
            )
        else:
            # Original behavior - use standard forward pass with caching
            if is_gemma2 and HybridCache is not None:
                # Create HybridCache for Gemma2
                past_key_values = HybridCache(
                    config=main_base.config,
                    max_batch_size=B,
                    max_cache_len=seq_embs.size(1) + max_length,
                    device=device,
                    dtype=seq_embs.dtype
                )
                
                with self._maybe_disable_dropout():
                    outputs = transformer(
                        inputs_embeds=seq_embs,
                        #cache_position=cache_position,
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
            else:
                # Standard forward for GPT2/LLaMA
                with self._maybe_disable_dropout():
                    outputs = transformer(
                        inputs_embeds=seq_embs,
                        use_cache=True,
                    )
            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

        
        # print("patchalllayers", self.config.patch_all_layers)
        # print("perlayerprojections", self.config.per_layer_projections)
        # print("len(layers)", len(layers))
        # print("len(past_key_values)", len(past_key_values))
        # Generate tokens
        current_position = seq_embs.size(1)
        
        # # Convert to DynamicCacheEnableDisable if needed (but not for Gemma2)
        # if not is_gemma2 and not isinstance(past_key_values, DynamicCacheEnableDisable):
        #     # Convert regular tuple/DynamicCache to our custom cache
        #     new_cache = DynamicCacheEnableDisable()
        #     if hasattr(past_key_values, 'key_cache'):
        #         # It's already a DynamicCache, just copy
        #         for i in range(len(past_key_values.key_cache)):
        #             new_cache.key_cache.append(past_key_values.key_cache[i])
        #             new_cache.value_cache.append(past_key_values.value_cache[i])
        #         new_cache._seen_tokens = past_key_values._seen_tokens
        #     else:
        #         # It's a tuple of (key, value) pairs per layer
        #         for layer_past in past_key_values:
        #             if layer_past is not None:
        #                 new_cache.key_cache.append(layer_past[0])
        #                 new_cache.value_cache.append(layer_past[1])
        #         new_cache._seen_tokens = seq_embs.size(1)
        #     past_key_values = new_cache
        
        if self.config.detach_after_each_sample and not is_gemma2:
            past_key_values.disable_new_grads()# after each sample, detach gradients! radically reduce memory usage
            
        if not self.config.end_to_end:
            ctxt = torch.no_grad()  
            if is_gemma2:
                copy_of_kvs = past_key_values  # Gemma2 doesn't support our custom detaching
            else:
                copy_of_kvs = clone_dynamic_cache_detach(past_key_values)
            #copy_of_hidden_states_with_grads = hidden_states.clone()
            copy_of_last_of_last_hidden_state=(main_out(hidden_states[:,-1])@input_emb_table).clone()
        elif self.config.end_to_end:
            ctxt = nullcontext()        
            copy_of_kvs = past_key_values
        with ctxt:
            gumbel_noise_list = [] # Store gumbel noise if not end-to-end
            for step in range(max_length):
                # Get logits for the last position
                logits_t = main_out(hidden_states[:, -1])  # (B, V)

                # Gumbel-Softmax sampling with numerical stability
                logits_t_scaled = logits_t / 1.0  # T_sampling = 1.0
                
                # Debug: Check logits before Gumbel-Softmax
                # if torch.isnan(logits_t).any():
                #     log.error(f"NaN detected in logits_t at generation step {step}")
                # if torch.isinf(logits_t).any():
                #     log.error(f"Inf detected in logits_t at generation step {step}")
                # if (torch.abs(logits_t) > 1e4).any():
                #     log.warning(f"Large logits at step {step}: max={logits_t.max().item():.2e}, min={logits_t.min().item():.2e}")
                
                with torch.amp.autocast('cuda',enabled=False):
                    logits_f32 = logits_t_scaled.float()
                    # Subtract max for numerical stability (detached)
                    logits_f32 = logits_f32 - logits_f32.max(dim=-1, keepdim=True)[0].detach()
                    if self.config.end_to_end:
                        ste_token_dist = torch.nn.functional.gumbel_softmax(
                            logits_f32,
                            tau=max(gumbel_tau, 0.1),  # Prevent extremely low tau
                            hard=True
                        ).to(logits_t.dtype)
                    else:
                        # In the non-end-to-end case, we sample with Gumbel noise
                        # but detach the computation graph for the forward pass.
                        # The noise is stored for use in the backward pass recomputation.
                        gumbels = (
                            -torch.empty_like(logits_f32, memory_format=torch.legacy_contiguous_format)
                            .exponential_()
                            .log()
                        )
                        gumbel_noise_list.append(gumbels)

                        # Sample using Gumbel-max trick (argmax of logits + Gumbel noise)
                        gumbel_logits = logits_f32 + gumbels
                        hard_token_ids = gumbel_logits.argmax(dim=-1)
                        
                        ste_token_dist = torch.nn.functional.one_hot(
                            hard_token_ids, num_classes=logits_f32.shape[-1]
                        ).to(logits_t.dtype)

                # Get embeddings
                emb_t_input = ste_token_dist @ input_emb_table
                if embeddings_tied:
                    emb_t_output = emb_t_input
                else:
                    output_embs = ste_token_dist @ output_emb_table

                # Store outputs
                logits_list.append(logits_t)
                if self.config.end_to_end:
                    output_embs_list.append(emb_t_output)
                hard_ids_list.append(ste_token_dist.argmax(dim=-1))
                #if not self.config.end_to_end and not embeddings_tied:
                #    input_embs_list.append(emb_t_input)

                # Process new token through transformer with cached K,V
                if step < max_length - 1:  # Don't process last token if we won't use it
                    # Update cache_position for Gemma2
                    # if is_gemma2:
                    #     cache_position = cache_position[-1:] + 1
                    
                    # Original behavior - use native caching
                    with self._maybe_disable_dropout():
                        if is_gemma2:
                            outputs = transformer(
                                inputs_embeds=emb_t_input.unsqueeze(1),
                                #cache_position=cache_position,
                                past_key_values=copy_of_kvs,
                                use_cache=True,
                            )
                        else:
                            outputs = transformer(
                                inputs_embeds=emb_t_input.unsqueeze(1),
                                past_key_values=copy_of_kvs,
                                use_cache=True,
                            )
                    hidden_states = outputs.last_hidden_state
                    if self.config.detach_after_each_sample:
                        hidden_states = hidden_states.detach()
                    copy_of_kvs = outputs.past_key_values
                    current_position += 1

        if not self.config.end_to_end:
            # RL = style - here is where the gradients flow??
            # allow the gradient to flow here through copy of hidden states with grads - why not.
            # Concatenate the initial hidden state with the embeddings of the selected hard token ids (excluding the last token)
            hard_ids = torch.stack(hard_ids_list, dim=1)  # (B, T)
            inputs_embeds = torch.cat( #want the logits for these guys - the next tokens
                [copy_of_last_of_last_hidden_state.unsqueeze(1), input_emb_table[hard_ids[:, :-1]]],
                dim=1
            )
            
            # For Gemma2, we need cache_position for the final forward pass
            if is_gemma2:
                final_cache_position = torch.arange(
                    current_position - max_length, current_position,
                    device=device, dtype=torch.long
                )
                outputs = main_base(
                    inputs_embeds=inputs_embeds, 
                    #cache_position=final_cache_position,
                    past_key_values=past_key_values, 
                    use_cache=True, 
                    output_hidden_states=False
                )
            else:
                outputs = main_base(
                    inputs_embeds=inputs_embeds, 
                    past_key_values=past_key_values, 
                    use_cache=True, 
                    output_hidden_states=False
                )
                
            with torch.amp.autocast('cuda',enabled=False):
                logits_f32 = outputs.logits.float()
                # The graph from outputs.logits is now attached to logits_f32.
                # We can free the original outputs object and its graph.
                del outputs

                logits_f32 = logits_f32 - logits_f32.max(dim=-1, keepdim=True)[0].detach()
                
                # For the reparameterization trick to be valid, we must use the same Gumbel noise
                # that was used in the forward sampling pass.
                gumbel_noise = torch.stack(gumbel_noise_list, dim=1)  # (B, T, V)
                del gumbel_noise_list
                
                gumbel_logits = (logits_f32 + gumbel_noise) / max(gumbel_tau, 0.1)
                del gumbel_noise
                
                y_soft = gumbel_logits.softmax(dim=-1)
                del gumbel_logits

                y_hard = torch.nn.functional.one_hot(hard_ids, num_classes=logits_f32.shape[-1]).to(y_soft.dtype)
                del logits_f32

                # Straight-through estimator, computed in the smaller embedding space
                # to avoid creating a massive (B, T, V) intermediate tensor.
                y_soft_embs = y_soft @ output_emb_table
                del y_soft

                # y_hard is a one-hot (B, T, V) tensor. Multiplying it by the embedding table
                # is equivalent to a much more efficient indexing operation.
                y_hard_embs = output_emb_table[hard_ids]
                del y_hard

                # The STE trick, now applied to embeddings. The gradient path is preserved
                # through y_soft_embs.
                output_embs = y_hard_embs - y_soft_embs.detach() + y_soft_embs

            text_embs = output_embs
        else:
            hard_ids = torch.stack(hard_ids_list, dim=1)
            text_embs = torch.stack(output_embs_list, dim=1)
        
        # Stack outputs
        logits_seq = torch.stack(logits_list, dim=1)
        #hard_ids = torch.stack(hard_ids_list, dim=1)
        
        #text_embs = torch.stack(output_embs_list, dim=1)
        # should be embedded with enc's input, but it's the same so ok for now! 
        return Generated(text_embs, logits_seq, hard_ids)

    def generate_soft_kv_cached_nondiff(
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
        special_token = None,
        original_token_pos: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        add_tokens: list[int] = None,
    ) -> Generated:
        """Differentiable generation with KV caching for O(n) attention computation.
        
        This method produces identical results to generate_soft but avoids
        recomputing attention for past tokens by caching their key/value projections.
        Works with GPT-2, LLaMA, and Gemma2 architectures.
        
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
            main_out = self.base.get_output_embeddings()

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
        
        # Detect if this is a Gemma2 model
        is_gemma2 = (hasattr(main_base.config, 'model_type') and 
                     main_base.config.model_type == 'gemma2')
        
        # Subtract positional embedding from activation_input if configured
        activation_input_modified = activation_input
        if self.config.subtract_add_pos_embeddings and original_token_pos is not None and self.activation_pos_embedder is not None:
            original_token_pos = original_token_pos.to(device=device, dtype=torch.long)
            original_token_pos = original_token_pos.squeeze() if original_token_pos.dim() == 2 else original_token_pos
            
            pos_embeds_to_subtract = self.activation_pos_embedder(original_token_pos)
            
            if pos_embeds_to_subtract.shape != activation_input.shape:
                if pos_embeds_to_subtract.shape[0] == 1 and activation_input.shape[0] > 1:
                    pos_embeds_to_subtract = pos_embeds_to_subtract.expand_as(activation_input)
                else:
                     raise ValueError(
                        f"Shape mismatch for positional embedding subtraction: "
                        f"activation_input shape {activation_input.shape}, "
                        f"pos_embeds_to_subtract shape {pos_embeds_to_subtract.shape}"
                    )
            activation_input_modified = activation_input - pos_embeds_to_subtract
        
        # Get embedding tables
        input_emb_table = main_base.get_input_embeddings().weight
        output_emb_table = main_base.get_output_embeddings().weight
        embeddings_tied = (input_emb_table.data_ptr() == output_emb_table.data_ptr())
        #  if not embeddings_tied or self.config.output_head:
        #     raise ValueError("Embeddings are not tied or output head is trainable - this is not supported")

        
        # Prepare initial sequence
        parts = []
        if prompt_left_emb is not None:
            parts.append(prompt_left_emb.expand(B, -1, -1))
        
        # Always insert activation as a token
        if use_projection:
            if self.config.patch_all_layers and self.config.per_layer_projections:
                # Use first layer's projection
                a_proj = self._apply_projection(activation_input_modified, layer_idx=0).unsqueeze(1)
            else:
                # Use single projection
                a_proj = self._apply_projection(activation_input_modified).unsqueeze(1)
        else:
            # No projection
            a_proj = activation_input_modified.unsqueeze(1)
        if do_patching: 
            parts.append(a_proj)
        else: 
            print("patching in special token")
            parts.append(input_emb_table[special_token.squeeze().item() if isinstance(special_token, torch.Tensor) else special_token].clone().unsqueeze(0).repeat(B,1,1))
            
        if prompt_right_emb is not None:
            parts.append(prompt_right_emb.expand(B, -1, -1))
        
        if add_tokens is not None:
            # Convert the list of token IDs to a tensor on the correct device and with the correct dtype.
            token_ids_tensor = torch.tensor(
                add_tokens, dtype=torch.long, device=input_emb_table.device
            )
            # Get embeddings for these tokens, then unsqueeze to add a batch dimension,
            # and expand to match the batch size B.
            # Using expand is generally preferred over repeat for memory efficiency here.
            added_token_embeddings = input_emb_table[token_ids_tensor].unsqueeze(0).expand(B, -1, -1)
            parts.append(added_token_embeddings)
            print(f"new length: {len(parts)} (added {len(add_tokens)} tokens)")
        seq_embs = torch.cat(parts, dim=1)
        
        # Initialize storage
        logits_list = []
        hard_ids_list = []
        output_embs_list = []
        
        # Storage for past_key_values
        past_key_values = None
        
        # # Initialize cache_position for Gemma2
        # if is_gemma2:
        #     cache_position = torch.arange(seq_embs.size(1), device=device, dtype=torch.long)
        
        # Get the transformer and detect architecture
        if hasattr(main_base, 'transformer'):
            # GPT-2 style model
            transformer = main_base.transformer
            layers = transformer.h
            final_norm = transformer.ln_f
        elif hasattr(main_base, 'model'):
            # LLaMA/Gemma style model (model.layers)
            transformer = main_base.model
            layers = transformer.layers
            final_norm = transformer.norm
        else:
            raise ValueError(f"Unknown model architecture. Expected transformer or model attribute.")
        
        # Process initial sequence (prompt + activation)
        if self.config.patch_all_layers:
            # Use _patched_forward for patching logic
            # Determine max_total_len for cache, needed if _patched_forward creates it
            max_total_length_for_cache = seq_embs.size(1) + max_length

            hidden_states, past_key_values = self._patched_forward(
                main_base=main_base,
                seq_embs=seq_embs,
                activation_input_modified=activation_input_modified,
                use_projection=use_projection,
                do_patching=do_patching,
                prompt_left_emb=prompt_left_emb,
                past_key_values=past_key_values, # Initially None
                use_cache=True, # KV caching is the point of this function
                attention_mask=None, # Assuming causal mask is handled internally by model
                #cache_position=cache_position if is_gemma2 else None,
                max_total_len_for_cache=max_total_length_for_cache if is_gemma2 else None, # Pass for Gemma2
                return_past_key_values=True,
            )
        else:
            # Original behavior - use standard forward pass with caching
            if is_gemma2 and HybridCache is not None:
                # Create HybridCache for Gemma2
                past_key_values = HybridCache(
                    config=main_base.config,
                    max_batch_size=B,
                    max_cache_len=seq_embs.size(1) + max_length,
                    device=device,
                    dtype=seq_embs.dtype
                )
                
                with self._maybe_disable_dropout():
                    outputs = transformer(
                        inputs_embeds=seq_embs,
                        #cache_position=cache_position,
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
            else:
                # Standard forward for GPT2/LLaMA
                with self._maybe_disable_dropout():
                    outputs = transformer(
                        inputs_embeds=seq_embs,
                        use_cache=True,
                    )
            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

        current_position = seq_embs.size(1)
        
        if self.config.detach_after_each_sample and not is_gemma2:
            past_key_values.disable_new_grads()# after each sample, detach gradients! radically reduce memory usage
            
        if not self.config.end_to_end:
            ctxt = torch.no_grad()  
            if is_gemma2:
                copy_of_kvs = past_key_values  # Gemma2 doesn't support our custom detaching
            else:
                copy_of_kvs = clone_dynamic_cache_detach(past_key_values)
            #copy_of_hidden_states_with_grads = hidden_states.clone()
            copy_of_last_of_last_hidden_state=(main_out(hidden_states[:,-1])@input_emb_table).clone()
        elif self.config.end_to_end:
            ctxt = nullcontext()        
            copy_of_kvs = past_key_values
        with ctxt:
            gumbel_noise_list = [] # Store gumbel noise if not end-to-end
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
                if return_logits:
                    logits_list.append(logits_t)

                # Get embeddings
                emb_t_input = ste_token_dist @ input_emb_table
                if embeddings_tied:
                    emb_t_output = emb_t_input
                else:
                    output_embs = ste_token_dist @ output_emb_table

                # Store outputs
                if self.config.end_to_end:
                    output_embs_list.append(emb_t_output)
                hard_ids_list.append(ste_token_dist.argmax(dim=-1))
                #if not self.config.end_to_end and not embeddings_tied:
                #    input_embs_list.append(emb_t_input)

                # Process new token through transformer with cached K,V
                if step < max_length - 1:  # Don't process last token if we won't use it
                    # Update cache_position for Gemma2
                    # if is_gemma2:
                    #     cache_position = cache_position[-1:] + 1
                    
                    # Original behavior - use native caching
                    with self._maybe_disable_dropout():
                        if is_gemma2:
                            outputs = transformer(
                                inputs_embeds=emb_t_input.unsqueeze(1),
                                #cache_position=cache_position,
                                past_key_values=copy_of_kvs,
                                use_cache=True,
                            )
                        else:
                            outputs = transformer(
                                inputs_embeds=emb_t_input.unsqueeze(1),
                                past_key_values=copy_of_kvs,
                                use_cache=True,
                            )
                    hidden_states = outputs.last_hidden_state
                    if self.config.detach_after_each_sample:
                        hidden_states = hidden_states.detach()
                    copy_of_kvs = outputs.past_key_values
                    current_position += 1

        hard_ids = torch.stack(hard_ids_list, dim=1)
        text_embs = torch.stack(output_embs_list, dim=1)
        if return_logits:
            logits = torch.stack(logits_list, dim=1)
        else:
            logits = None
        
        # Stack outputs
        #hard_ids = torch.stack(hard_ids_list, dim=1)
        
        #text_embs = torch.stack(output_embs_list, dim=1)
        # should be embedded with enc's input, but it's the same so ok for now! 
        return Generated(text_embs,logits, hard_ids)

    
    def fwd_tokens(
        self,
        activation_input: torch.Tensor,
        use_projection: bool = True,
        original_token_pos: Optional[torch.Tensor] = None,
        input_tokens: Optional[torch.Tensor] = None,
        detach_entropy: bool = True,
        calculate_entropy: bool = True,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs a single forward pass to get probabilities and entropies for a given sequence.

        This method constructs a sequence from a prompt and `input_tokens`,
        with a projected activation inserted in the prompt. It then performs a
        single forward pass through the model to get the logits for predicting
        each token in `input_tokens`.

        Args:
            activation_input: The activation to be projected and inserted into the prompt.
            use_projection: Whether to apply the projection to the activation.
            original_token_pos: Positional info for the activation, for pos embedding subtraction.
            input_tokens: The sequence of token IDs to evaluate. Must be provided.

        Returns:
            A tuple containing:
            - probs_of_interest (torch.Tensor): The model's predicted probability for each
              token in `input_tokens`. Shape: (batch_size, seq_len).
            - entropies (torch.Tensor): The detached entropy of the predictive distribution
              for each token position in `input_tokens`. Shape: (batch_size, seq_len).
        """
        if input_tokens is None:
            raise ValueError("`fwd_tokens` requires `input_tokens` to be provided.")

        # Setup similar to generate()
        main_base = self.base
        main_out = self.base.get_output_embeddings()

        prompt_left_emb = self.prompt_left_emb
        prompt_right_emb = self.prompt_right_emb

        # Get dtype from projection layer
        if self.config.per_layer_projections:
            if self.proj_weight is not None:
                activation_input = activation_input.to(self.proj_weight.dtype)
            else:
                activation_input = activation_input.to(self.proj.weight.dtype)
        else:
            activation_input = activation_input.to(self.proj.weight.dtype)
        B, d_model = activation_input.shape
        device = activation_input.device
        
        # Subtract positional embedding from activation_input if configured
        activation_input_modified = activation_input
        if self.config.subtract_add_pos_embeddings and original_token_pos is not None and self.activation_pos_embedder is not None:
            original_token_pos = original_token_pos.to(device=device, dtype=torch.long)
            original_token_pos = original_token_pos.squeeze() if original_token_pos.dim() == 2 else original_token_pos
            
            pos_embeds_to_subtract = self.activation_pos_embedder(original_token_pos)
            
            if pos_embeds_to_subtract.shape != activation_input.shape:
                if pos_embeds_to_subtract.shape[0] == 1 and activation_input.shape[0] > 1:
                    pos_embeds_to_subtract = pos_embeds_to_subtract.expand_as(activation_input)
                else:
                     raise ValueError(
                        f"Shape mismatch for positional embedding subtraction: "
                        f"activation_input shape {activation_input.shape}, "
                        f"pos_embeds_to_subtract shape {pos_embeds_to_subtract.shape}"
                    )
            activation_input_modified = activation_input - pos_embeds_to_subtract
        
        # Get embedding tables
        input_emb_table = main_base.get_input_embeddings().weight
        output_emb_table = main_base.get_output_embeddings().weight
        embeddings_tied = (input_emb_table.data_ptr() == output_emb_table.data_ptr())
        if not embeddings_tied:
            raise ValueError("Embeddings are not tied - this is not supported")

        
        # Prepare initial sequence
        parts = []
        prompt_len_left = 0
        if prompt_left_emb is not None:
            parts.append(prompt_left_emb.expand(B, -1, -1))
            prompt_len_left = prompt_left_emb.shape[0]
        
        # Always insert activation as a token
        if use_projection:
            if self.config.patch_all_layers and self.config.per_layer_projections:
                # Use first layer's projection
                a_proj = self._apply_projection(activation_input_modified, layer_idx=0).unsqueeze(1)
            else:
                # Use single projection
                a_proj = self._apply_projection(activation_input_modified).unsqueeze(1)
        else:
            # No projection
            a_proj = activation_input_modified.unsqueeze(1)
        parts.append(a_proj)
            
        prompt_len_right = 0
        if prompt_right_emb is not None:
            parts.append(prompt_right_emb.expand(B, -1, -1))
            prompt_len_right = prompt_right_emb.shape[0]
            
        # Ensure input_tokens is broadcastable to batch size B
        if input_tokens.dim() == 1:
            input_tokens = input_tokens.unsqueeze(0).expand(B, -1)
        parts.append(input_emb_table[input_tokens])
            
        seq_embs = torch.cat(parts, dim=1)

        # A single forward pass through the model.
        if self.config.patch_all_layers:
            # Custom forward pass with activation patching at all layers
            hidden_states = self._patched_forward(
                main_base=main_base,
                seq_embs=seq_embs,
                activation_input_modified=activation_input_modified,
                use_projection=use_projection,
                do_patching=True,  # Always patch when calling fwd_tokens
                prompt_left_emb=prompt_left_emb,
                return_past_key_values=False,
                use_cache=False,
            )
            logits = main_out(hidden_states)
        else:
            # Standard forward pass - base model already includes LM head
            with self._maybe_disable_dropout():
                outputs = main_base(inputs_embeds=seq_embs)
            # Get logits directly from the model output
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        
        prompt_len = prompt_len_left + 1 + prompt_len_right
        
        # Logits for predicting input_tokens.
        # The logit at index `t` is for predicting token `t+1`.
        # To predict `input_tokens`, we need logits from `prompt_len - 1` up to the one before last.
        logits_for_input_tokens = logits[:, prompt_len - 1 : -1, :]
        if logits_for_input_tokens.shape[1] != input_tokens.shape[1]:
            raise ValueError(f"Logits for input tokens have shape {logits_for_input_tokens.shape} but input tokens have shape {input_tokens.shape}, originally  logits shape {logits.shape}")
        
        # Calculate probabilities over the vocabulary
        probs = torch.nn.functional.softmax(logits_for_input_tokens, dim=-1)
        
        # Gather the probabilities of the actual `input_tokens` that occurred.
        probs_of_interest = probs.gather(dim=2, index=input_tokens.unsqueeze(-1)).squeeze(-1)
        
        # Calculate entropy for each token's predictive distribution.
        # Using log_softmax for better numerical stability.
        if calculate_entropy:
            context = torch.no_grad() if detach_entropy else nullcontext()
            with context:
                log_probs = torch.nn.functional.log_softmax(logits_for_input_tokens, dim=-1)
                entropies = (-probs * log_probs).sum(dim=-1)
        else:
            entropies = None

        if return_logits:
            return probs_of_interest, entropies#, logits_for_input_tokens
        else:   
            return probs_of_interest, entropies
    
class DynamicCacheEnableDisable(DynamicCache):
    """
    A DynamicCache that can enable/disable gradients for new key-value states.
    When gradients are disabled, new key-value states are detached before being added to the cache.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.grads_enabled: bool = True

    def enable_new_grads(self) -> None:
        """Enable gradients for new additions to the KV cache."""
        self.grads_enabled = True

    def disable_new_grads(self) -> None:
        """Disable gradients for new additions to the KV cache."""
        self.grads_enabled = False

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        If `self.grads_enabled` is `False`, the new states are detached before being passed to the superclass.
        """
        if not self.grads_enabled and key_states is not None:
            key_states = key_states.detach()
            value_states = value_states.detach()

        return super().update(key_states, value_states, layer_idx, cache_kwargs)

#from torch.nn.functional import
import warnings
from torch.overrides import (
    handle_torch_function,
    has_torch_function,
    has_torch_function_unary,
    has_torch_function_variadic,
)

def gumbel_softmax_fixed_choice(
    logits: torch.Tensor,
    tau: float = 1,
    hard: bool = False,
    eps: float = 1e-10,
    dim: int = -1,
    selected_indices = None,
) -> torch.Tensor:
    r"""
    Sample from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretize.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """
    if has_torch_function_unary(logits):
        return handle_torch_function(
            gumbel_softmax_fixed_choice, (logits,), logits, tau=tau, hard=hard, eps=eps, dim=dim, selected_indices=selected_indices
        )
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        if selected_indices is not None:
            index = selected_indices
        else:
            index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

def clone_dynamic_cache_detach(original_cache):
    # Create new cache
    new_cache = DynamicCacheEnableDisable()

    # Copy each layer's key/value tensors
    for i in range(len(original_cache.key_cache)):
        if original_cache.key_cache[i] is not None:
            new_cache.key_cache.append(original_cache.key_cache[i].clone().detach())
            new_cache.value_cache.append(original_cache.value_cache[i].clone().detach())

    # Copy the _seen_tokens attribute
    new_cache._seen_tokens = original_cache._seen_tokens

    return new_cache

















#DO NNOT DELETE THE BELOW CODE!

    # #@DeprecationWarning("use generate_soft_kv_cached instead")
    # # TODO FIX THIS implementaiton - do not use as is
    # def generate_soft_chkpt(
    #     self,
    #     activation_input: torch.Tensor,
    #     max_length: int,
    #     gumbel_tau: float,
    #     use_projection: bool = True,
    #     print_prompt: bool = False,
    #     hard_left_emb: list[int] = None,
    #     hard_right_emb: list[int] = None,
    #     override_model_base_and_out = None,
    #     checkpoint_every_n_tokens: int = 4,
    # ) -> Generated:
    #     """Differentiable autoregressive generation with gradient checkpointing.
        
    #     This version maintains full differentiability through all generated tokens
    #     while using gradient checkpointing to reduce memory usage.
        
    #     Args:
    #         Same as generate_soft, plus:
    #         checkpoint_every_n_tokens: How often to checkpoint (default: every 4 tokens)
    #     """
    #     from torch.utils.checkpoint import checkpoint
        
    #     if print_prompt and hasattr(self, 'prompt_text'):
    #         print(f"Prompt template: {self.prompt_text}")

    #     # Ensure dtype matches linear layer to avoid Half/Float mismatch during eval.
    #     # Parameters are automatically on the correct device when the module is moved

    #     if override_model_base_and_out is not None:
    #         main_model = override_model_base_and_out
    #         # Handle both OrigWrapper (has .model) and raw model cases
    #         if hasattr(main_model, 'model'):
    #             main_base = main_model.model
    #             main_out = main_model.model.lm_head if hasattr(main_model.model, 'lm_head') else main_model.model.get_output_embeddings()
    #         else:
    #             # Direct model passed (e.g., GPT2LMHeadModel)
    #             main_base = main_model
    #             main_out = main_model.lm_head if hasattr(main_model, 'lm_head') else main_model.get_output_embeddings()
    #     else:   
    #         main_model = self
    #         main_base = self.base
    #         main_out = self.out

    #     if hard_left_emb is not None:
    #         prompt_left_emb = main_base.get_input_embeddings().weight[hard_left_emb].clone()
    #     else:
    #         prompt_left_emb = self.prompt_left_emb
    #     if hard_right_emb is not None:
    #         prompt_right_emb = main_base.get_input_embeddings().weight[hard_right_emb].clone()
    #     else:
    #         prompt_right_emb = self.prompt_right_emb

    #     # Get dtype from projection layer
    #     if self.config.per_layer_projections:
    #         activation_input = activation_input.to(self.proj_weight.dtype)
    #     else:
    #         activation_input = activation_input.to(self.proj.weight.dtype)

    #     B, d_model = activation_input.shape
    #     device = activation_input.device
        
    #     # Get both input and output embedding tables
    #     input_emb_table = main_base.get_input_embeddings().weight  # (V, d_model)
    #     output_emb_table = main_base.get_output_embeddings().weight  # (V, d_model)
        
    #     # Check if embeddings are tied (same memory location)
    #     embeddings_tied = (input_emb_table.data_ptr() == output_emb_table.data_ptr())

    #     # 0) prepend textual prompt (pre-computed at set_prompt)
    #     parts = []
    #     if prompt_left_emb is not None:
    #         parts.append(prompt_left_emb.expand(B, -1, -1))
        
    #     # Always insert activation as a token (will be replaced at each layer if patch_all_layers=True)
    #     if use_projection and not self.config.patch_all_layers:
    #         # Only apply projection here if not patching all layers
    #         a_proj = self._apply_projection(activation_input).unsqueeze(1)
    #     else:
    #         # Use unprojected activation as placeholder
    #         a_proj = activation_input.unsqueeze(1)
    #     parts.append(a_proj)
            
    #     if prompt_right_emb is not None:
    #         parts.append(prompt_right_emb.expand(B, -1, -1))
            
    #     seq_embs = torch.cat(parts, dim=1)

    #     logits_list = []
    #     hard_ids_list = []
    #     output_embs_list = []  # Store embeddings for encoder
        
    #     # Define single step function for checkpointing as a method
    #     def generation_step(seq_embs_input, step_idx, 
    #                       main_base, main_out, 
    #                       input_emb_table, output_emb_table, embeddings_tied,
    #                       activation_input, use_projection, prompt_left_emb,
    #                       B, device, config, decoder_module,
    #                       gumbel_tau, max_gumbel_tau=0.1):
    #         """Single generation step that can be checkpointed."""
    #         if config.patch_all_layers:
    #             # Custom forward pass with activation patching at all layers
    #             # Get transformer module and detect architecture
    #             if hasattr(main_base, 'transformer'):
    #                 # GPT-2 style model
    #                 transformer = main_base.transformer
    #                 layers = transformer.h
    #                 final_norm = transformer.ln_f
    #             elif hasattr(main_base, 'model'):
    #                 # LLaMA style model (model.layers)
    #                 transformer = main_base.model
    #                 layers = transformer.layers
    #                 final_norm = transformer.norm
    #             else:
    #                 raise ValueError(f"Unknown model architecture. Expected transformer or model attribute.")
                
    #             # Embedding layer
    #             hidden_states = seq_embs_input
                
    #             # Get position IDs
    #             seq_length = hidden_states.size(1)
    #             position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
                
    #             # Pre-compute single projection if not using per-layer projections
    #             if not config.per_layer_projections:
    #                 if use_projection:
    #                     single_proj = decoder_module._apply_projection(activation_input)
    #                 else:
    #                     single_proj = activation_input
                
    #             # Calculate embed position (where activation should be patched)
    #             embed_pos = prompt_left_emb.size(0) if prompt_left_emb is not None else 0
                
    #             # Compute rotary embeddings for LLaMA if needed
    #             if hasattr(main_base, 'model') and hasattr(transformer, 'rotary_emb'):
    #                 # LLaMA uses rotary embeddings
    #                 cos, sin = transformer.rotary_emb(hidden_states, position_ids)
    #                 position_embeddings = (cos, sin)
    #             else:
    #                 position_embeddings = None
                
    #             # Run through transformer layers with activation patching
    #             for layer_idx, layer_module in enumerate(layers):
    #                 # Apply layer
    #                 if hasattr(main_base, 'model'):
    #                     # LLaMA style - pass position embeddings
    #                     layer_outputs = layer_module(
    #                         hidden_states,
    #                         position_ids=position_ids,
    #                         position_embeddings=position_embeddings,
    #                     )
    #                 else:
    #                     # GPT-2 style
    #                     layer_outputs = layer_module(hidden_states, position_ids=position_ids)
    #                 hidden_states = layer_outputs[0]
                    
    #                 # Replace activation at the embed position for this layer
    #                 # Skip layer 0 for per-layer projections since it's already applied
    #                 if layer_idx > 0 or not config.per_layer_projections:
    #                     if config.per_layer_projections:
    #                         # Use layer-specific projection
    #                         if use_projection:
    #                             a_proj_layer = decoder_module._apply_projection(activation_input, layer_idx=layer_idx)
    #                         else:
    #                             a_proj_layer = activation_input
    #                         hidden_states[:, embed_pos] = a_proj_layer
    #                     else:
    #                         # Use pre-computed single projection
    #                         hidden_states[:, embed_pos] = single_proj
                
    #             # Final layer norm
    #             hidden_states = final_norm(hidden_states)
                
    #             # Get logits
    #             h_last = hidden_states
    #             logits_t = main_out(h_last[:, -1])  # (B, V)
    #         else:
    #             # Original behavior
    #             # Model forward pass
    #             out = main_base(inputs_embeds=seq_embs_input, output_hidden_states=True)
    #             h_last = out.last_hidden_state if hasattr(out, "last_hidden_state") else out.hidden_states[-1]
    #             logits_t = main_out(h_last[:, -1])  # (B, V)

    #         # 1. Apply forward sampling temperature
    #         current_T_sampling = 1.0  # TODO: could add schedule
    #         logits_t_scaled = logits_t / current_T_sampling

    #         # 2. Apply Gumbel-Softmax with STE temperature (hard=True)
    #         # Add numerical stability for low tau values
    #         with torch.amp.autocast('cuda',enabled=False):
    #             logits_t_f32 = logits_t_scaled.float()
    #             # Subtract max for numerical stability (detached)
    #             logits_t_f32 = logits_t_f32 - logits_t_f32.max(dim=-1, keepdim=True)[0].detach()
    #             ste_token_dist = torch.nn.functional.gumbel_softmax(
    #                 logits_t_f32,
    #                 tau=max(gumbel_tau, 0.1),  # Prevent extremely low tau
    #                 hard=True  # Keep using STE as in original
    #             ).to(logits_t_scaled.dtype)
            
    #         # Use input embeddings for autoregressive feedback
    #         emb_t_input = ste_token_dist @ input_emb_table  # (B, d_model)
            
    #         # Use output embeddings for the encoder (or reuse input if tied)
    #         if embeddings_tied:
    #             emb_t_output = emb_t_input
    #         else:
    #             emb_t_output = ste_token_dist @ output_emb_table  # (B, d_model)
            
    #         # Store hard token IDs derived from the STE output
    #         hard_ids = ste_token_dist.argmax(dim=-1)
            
    #         return logits_t, emb_t_input, emb_t_output, hard_ids

    #     # Main generation loop with checkpointing
    #     for step in range(max_length):
    #         # Checkpoint all tokens except the first few (need some non-checkpointed for gradients)
    #         # When checkpoint_every_n_tokens=1, checkpoint all but first token
    #         # When checkpoint_every_n_tokens>1, checkpoint every N tokens
    #         should_checkpoint = (
    #             checkpoint_every_n_tokens == 1 and step > 0  # For every-token mode
    #         ) or (
    #             checkpoint_every_n_tokens > 1 and step % checkpoint_every_n_tokens == 0 and step > 0
    #         )
            
    #         if should_checkpoint:
    #             # Use gradient checkpointing
    #             logits_t, emb_t_input, emb_t_output, hard_ids = checkpoint(
    #                 generation_step, seq_embs, step, 
    #                 main_base, main_out,
    #                 input_emb_table, output_emb_table, embeddings_tied,
    #                 activation_input, use_projection, prompt_left_emb,
    #                 B, device, self.config, self,
    #                 gumbel_tau,
    #                 use_reentrant=False
    #             )
    #         else:
    #             # Regular forward pass
    #             logits_t, emb_t_input, emb_t_output, hard_ids = generation_step(
    #                 seq_embs, step,
    #                 main_base, main_out,
    #                 input_emb_table, output_emb_table, embeddings_tied,
    #                 activation_input, use_projection, prompt_left_emb,
    #                 B, device, self.config, self,
    #                 gumbel_tau
    #             )
            
    #         # Feed input embedding back for next autoregressive step
    #         seq_embs = torch.cat([seq_embs, emb_t_input.unsqueeze(1)], dim=1)
            
    #         # Store outputs
    #         logits_list.append(logits_t)
    #         output_embs_list.append(emb_t_output)
    #         hard_ids_list.append(hard_ids)

    #     # Stack along time dim (B, T, ...)
    #     logits_seq = torch.stack(logits_list, dim=1)
    #     hard_ids = torch.stack(hard_ids_list, dim=1)
        
    #     # Stack output embeddings for encoder
    #     text_embs = torch.stack(output_embs_list, dim=1)  # (B, T, d_model)

    #     return Generated(text_embs, logits_seq, hard_ids)

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------



    # def generate_soft_kv_cached_nondiff(
    #     self,
    #     activation_input: torch.Tensor,
    #     max_length: int,
    #     gumbel_tau: float,
    #     use_projection: bool = True,
    #     print_prompt: bool = False,
    #     hard_left_emb: list[int] = None,
    #     hard_right_emb: list[int] = None,
    #     override_model_base_and_out = None,
    #     do_patching: bool = True,
    #     special_token = None,
    #     original_token_pos: Optional[torch.Tensor] = None,
    # ) -> Generated:
    #     """Differentiable generation with KV caching for O(n) attention computation.
        
    #     This method produces identical results to generate_soft but avoids
    #     recomputing attention for past tokens by caching their key/value projections.
    #     Only works with GPT-2 architecture currently.
        
    #     Args:
    #         Same as generate_soft
            
    #     Returns:
    #         Same as generate_soft
    #     """
    #     # Using native transformer KV caching instead of custom implementation
        
    #     # Setup similar to generate_soft
    #     if print_prompt and hasattr(self, 'prompt_text'):
    #         print(f"Prompt template: {self.prompt_text}")

    #     # Ensure dtype matches linear layer
    #     # Parameters are automatically on the correct device when the module is moved

    #     if override_model_base_and_out is not None:
    #         main_model = override_model_base_and_out
    #         # Handle both OrigWrapper (has .model) and raw model cases
    #         if hasattr(main_model, 'model'):
    #             main_base = main_model.model
    #             main_out = main_model.model.lm_head if hasattr(main_model.model, 'lm_head') else main_model.model.get_output_embeddings()
    #         else:
    #             # Direct model passed (e.g., GPT2LMHeadModel)
    #             main_base = main_model
    #             main_out = main_model.lm_head if hasattr(main_model, 'lm_head') else main_model.get_output_embeddings()
    #     else:   
    #         main_model = self
    #         main_base = self.base
    #         main_out = self.out

    #     if hard_left_emb is not None:
    #         prompt_left_emb = main_base.get_input_embeddings().weight[hard_left_emb].clone()
    #     else:
    #         prompt_left_emb = self.prompt_left_emb
    #     if hard_right_emb is not None:
    #         prompt_right_emb = main_base.get_input_embeddings().weight[hard_right_emb].clone()
    #     else:
    #         prompt_right_emb = self.prompt_right_emb

    #     # Get dtype from projection layer
    #     if self.config.per_layer_projections:
    #         activation_input = activation_input.to(self.proj_weight.dtype)
    #     else:
    #         activation_input = activation_input.to(self.proj.weight.dtype)
    #     B, d_model = activation_input.shape
    #     device = activation_input.device
        
    #     # Subtract positional embedding from activation_input if configured
    #     activation_input_modified = activation_input
    #     if self.config.subtract_add_pos_embeddings and original_token_pos is not None and self.activation_pos_embedder is not None:
    #         original_token_pos = original_token_pos.to(device=device, dtype=torch.long)
    #         original_token_pos = original_token_pos.squeeze() if original_token_pos.dim() == 2 else original_token_pos
            
    #         pos_embeds_to_subtract = self.activation_pos_embedder(original_token_pos)
            
    #         if pos_embeds_to_subtract.shape != activation_input.shape:
    #             if pos_embeds_to_subtract.shape[0] == 1 and activation_input.shape[0] > 1:
    #                 pos_embeds_to_subtract = pos_embeds_to_subtract.expand_as(activation_input)
    #             else:
    #                  raise ValueError(
    #                     f"Shape mismatch for positional embedding subtraction: "
    #                     f"activation_input shape {activation_input.shape}, "
    #                     f"pos_embeds_to_subtract shape {pos_embeds_to_subtract.shape}"
    #                 )
    #         activation_input_modified = activation_input - pos_embeds_to_subtract
        
    #     # Get embedding tables
    #     input_emb_table = main_base.get_input_embeddings().weight
    #     output_emb_table = main_base.get_output_embeddings().weight
    #     embeddings_tied = (input_emb_table.data_ptr() == output_emb_table.data_ptr())
    #     #  if not embeddings_tied or self.config.output_head:
    #     #     raise ValueError("Embeddings are not tied or output head is trainable - this is not supported")

        
    #     # Prepare initial sequence
    #     parts = []
    #     if prompt_left_emb is not None:
    #         parts.append(prompt_left_emb.expand(B, -1, -1))
        
    #     # Always insert activation as a token
    #     if use_projection:
    #         if self.config.patch_all_layers and self.config.per_layer_projections:
    #             # Use first layer's projection
    #             a_proj = self._apply_projection(activation_input_modified, layer_idx=0).unsqueeze(1)
    #         else:
    #             # Use single projection
    #             a_proj = self._apply_projection(activation_input_modified).unsqueeze(1)
    #     else:
    #         # No projection
    #         a_proj = activation_input_modified.unsqueeze(1)
    #     if do_patching: 
    #         parts.append(a_proj)
    #     else: 
    #         print("patching in special token")
    #         parts.append(input_emb_table[special_token.squeeze().item() if isinstance(special_token, torch.Tensor) else special_token].clone().unsqueeze(0).repeat(B,1,1))
            
    #     if prompt_right_emb is not None:
    #         parts.append(prompt_right_emb.expand(B, -1, -1))
            
    #     seq_embs = torch.cat(parts, dim=1)
        
    #     # Initialize storage
    #     logits_list = []
    #     hard_ids_list = []
    #     output_embs_list = []
        
    #     # Storage for past_key_values
    #     past_key_values = None
        
    #     # Get the transformer and detect architecture
    #     if hasattr(main_base, 'transformer'):
    #         # GPT-2 style model
    #         transformer = main_base.transformer
    #         layers = transformer.h
    #         final_norm = transformer.ln_f
    #     elif hasattr(main_base, 'model'):
    #         # LLaMA style model (model.layers)
    #         transformer = main_base.model
    #         layers = transformer.layers
    #         final_norm = transformer.norm
    #     else:
    #         raise ValueError(f"Unknown model architecture. Expected transformer or model attribute.")
        
    #     # Process initial sequence (prompt + activation)
    #     if self.config.patch_all_layers:
    #         # Custom implementation with patching
    #         hidden_states = seq_embs
    #         seq_length = hidden_states.size(1)
    #         position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
            
    #         # Pre-compute single projection if not using per-layer projections
    #         if not self.config.per_layer_projections:
    #             if use_projection:
    #                 single_proj = self._apply_projection(activation_input_modified)
    #             else:
    #                 single_proj = activation_input_modified
            
    #         # Calculate embed position
    #         embed_pos = prompt_left_emb.size(0) if prompt_left_emb is not None else 0
            
    #         # Add position embeddings for GPT-2 (but not for LLaMA)
    #         if hasattr(main_base, 'transformer'):
    #             # GPT-2 needs position embeddings added before layers
    #             position_embeds = transformer.wpe(position_ids)
    #             hidden_states = transformer.drop(hidden_states + position_embeds)
            
    #         # Compute rotary embeddings for LLaMA if needed
    #         if hasattr(main_base, 'model') and hasattr(transformer, 'rotary_emb'):
    #             # LLaMA uses rotary embeddings
    #             cos, sin = transformer.rotary_emb(hidden_states, position_ids)
    #             position_embeddings = (cos, sin)
    #         else:
    #             position_embeddings = None
            
    #         # Process each layer with activation patching
    #         past_key_values = DynamicCacheEnableDisable()

    #         for layer_idx, layer_module in enumerate(layers):
    #             # FIRST: Apply patching to the input of this layer
    #             input_to_this_layer = hidden_states.clone()
    #             if self.config.per_layer_projections and do_patching:
    #                 # For per-layer projections, skip layer 0 since it's already applied in seq_embs
    #                 if layer_idx > 0:
    #                     if use_projection:
    #                         a_proj_layer = self._apply_projection(activation_input_modified, layer_idx=layer_idx)
    #                     else:
    #                         a_proj_layer = activation_input_modified
    #                     #input_to_this_layer = hidden_states.clone()
    #                     input_to_this_layer[:, embed_pos] = a_proj_layer
    #             elif do_patching:
    #                      input_to_this_layer[:, embed_pos] = single_proj


    #             # THEN: Process the patched input through the layer
    #             with self._maybe_disable_dropout():
    #                 if hasattr(main_base, 'transformer'):
    #                     # GPT-2 style - position embeddings already added
    #                     layer_outputs = layer_module(
    #                         input_to_this_layer,
    #                         use_cache=True,
    #                         past_key_value=past_key_values
    #                     )
    #                 else:
    #                     # LLaMA style - pass position_ids and position_embeddings
    #                     layer_outputs = layer_module(
    #                         input_to_this_layer,
    #                         position_ids=position_ids,
    #                         position_embeddings=position_embeddings,
    #                         use_cache=True,
    #                         past_key_value=past_key_values
    #                     )
                    
    #             hidden_states = layer_outputs[0]
    #         hidden_states = final_norm(hidden_states)
    #     else:
    #         # Original behavior - use standard forward pass with caching
    #         with self._maybe_disable_dropout():
    #             outputs = transformer(
    #                 inputs_embeds=seq_embs,
    #                 use_cache=True,
    #             )
    #         hidden_states = outputs.last_hidden_state
    #         past_key_values = outputs.past_key_values

    #     current_position = seq_embs.size(1)
        
    #     if self.config.detach_after_each_sample:
    #         past_key_values.disable_new_grads()# after each sample, detach gradients! radically reduce memory usage
            
    #     if not self.config.end_to_end:
    #         ctxt = torch.no_grad()  
    #         copy_of_kvs = clone_dynamic_cache_detach(past_key_values)
    #         #copy_of_hidden_states_with_grads = hidden_states.clone()
    #         copy_of_last_of_last_hidden_state=(main_out(hidden_states[:,-1])@input_emb_table).clone()
    #     elif self.config.end_to_end:
    #         ctxt = nullcontext()        
    #         copy_of_kvs = past_key_values
    #     with ctxt:
    #         gumbel_noise_list = [] # Store gumbel noise if not end-to-end
    #         for step in range(max_length):
    #             # Get logits for the last position
    #             logits_t = main_out(hidden_states[:, -1])  # (B, V)

    #             # Gumbel-Softmax sampling with numerical stability
    #             logits_t_scaled = logits_t / 1.0  # T_sampling = 1.0
    #             with torch.amp.autocast('cuda',enabled=False):
    #                 logits_f32 = logits_t_scaled.float()
    #                 # Subtract max for numerical stability (detached)
    #                 logits_f32 = logits_f32 - logits_f32.max(dim=-1, keepdim=True)[0].detach()
    #                 ste_token_dist = torch.nn.functional.gumbel_softmax(
    #                     logits_f32,
    #                     tau=max(gumbel_tau, 0.1),  # Prevent extremely low tau
    #                     hard=True
    #                 ).to(logits_t.dtype)

    #             # Get embeddings
    #             emb_t_input = ste_token_dist @ input_emb_table
    #             if embeddings_tied:
    #                 emb_t_output = emb_t_input
    #             else:
    #                 output_embs = ste_token_dist @ output_emb_table

    #             # Store outputs
    #             if self.config.end_to_end:
    #                 output_embs_list.append(emb_t_output)
    #             hard_ids_list.append(ste_token_dist.argmax(dim=-1))
    #             #if not self.config.end_to_end and not embeddings_tied:
    #             #    input_embs_list.append(emb_t_input)

    #             # Process new token through transformer with cached K,V
    #             if step < max_length - 1:  # Don't process last token if we won't use it
    #                 # Original behavior - use native caching
    #                 with self._maybe_disable_dropout():
    #                     outputs = transformer(
    #                         inputs_embeds=emb_t_input.unsqueeze(1),
    #                         past_key_values=copy_of_kvs,
    #                         use_cache=True,
    #                     )
    #                 hidden_states = outputs.last_hidden_state
    #                 if self.config.detach_after_each_sample:
    #                     hidden_states = hidden_states.detach()
    #                 copy_of_kvs = outputs.past_key_values
    #                 current_position += 1

    #     hard_ids = torch.stack(hard_ids_list, dim=1)
    #     text_embs = torch.stack(output_embs_list, dim=1)
        
    #     # Stack outputs
    #     #hard_ids = torch.stack(hard_ids_list, dim=1)
        
    #     #text_embs = torch.stack(output_embs_list, dim=1)
    #     # should be embedded with enc's input, but it's the same so ok for now! 
    #     return Generated(text_embs,None, hard_ids)

    # #@DeprecationWarning("use generate_soft_kv_cached instead")
    # # TODO FIX THIS implementaiton - do not use as is
    # def generate_soft_kv_flash(
    #     self,
    #     activation_input,
    #     max_length=64,
    #     gumbel_tau=1.0,
    #     hard_left_emb=None,
    #     hard_right_emb=None,
    #     print_prompt=False,
    #     override_model_base_and_out=None,
    # ):
    #     """Generate soft text using Flash Attention with KV caching.
        
    #     This method combines Flash Attention's optimized computation with
    #     KV caching for O(n) generation complexity.
        
    #     Args:
    #         Same as generate_soft
        
    #     Returns:
    #         Same as generate_soft
    #     """
    #     from lens.models.flash_kv_cache_v2 import FlashKVCache, compute_with_flash_kv_cache, FLASH_AVAILABLE
        
    #     if not FLASH_AVAILABLE:
    #         raise RuntimeError(
    #             "Flash Attention requested but not available. "
    #             "Install with: make flash-attention"
    #         )
        
    #     # Setup similar to generate_soft_kv_cached
    #     if print_prompt and hasattr(self, 'prompt_text'):
    #         print(f"Prompt template: {self.prompt_text}")

    #     if override_model_base_and_out is not None:
    #         main_model = override_model_base_and_out
    #         # Handle both OrigWrapper (has .model) and raw model cases
    #         if hasattr(main_model, 'model'):
    #             main_base = main_model.model
    #             main_out = main_model.model.lm_head if hasattr(main_model.model, 'lm_head') else main_model.model.get_output_embeddings()
    #         else:
    #             # Direct model passed (e.g., GPT2LMHeadModel)
    #             main_base = main_model
    #             main_out = main_model.lm_head if hasattr(main_model, 'lm_head') else main_model.get_output_embeddings()
    #     else:   
    #         main_model = self
    #         main_base = self.base
    #         main_out = self.out

    #     if hard_left_emb is not None:
    #         prompt_left_emb = main_base.get_input_embeddings().weight[hard_left_emb].clone()
    #     else:
    #         prompt_left_emb = self.prompt_left_emb
    #     if hard_right_emb is not None:
    #         prompt_right_emb = main_base.get_input_embeddings().weight[hard_right_emb].clone()
    #     else:
    #         prompt_right_emb = self.prompt_right_emb

    #     activation_input = activation_input.to(self.proj.weight.dtype)
    #     B, d_model = activation_input.shape
    #     device = activation_input.device
        
    #     # Get embedding tables
    #     input_emb_table = main_base.get_input_embeddings().weight
        
    #     # Project activation to embedding space
    #     emb_a = self.proj(activation_input)
        
    #     # Build initial prompt: [left_prompt, projected_activation, right_prompt]
    #     left_prompt_embs = prompt_left_emb.unsqueeze(0).expand(B, -1, -1)
    #     right_prompt_embs = prompt_right_emb.unsqueeze(0).expand(B, -1, -1)
    #     prompt_embs = torch.cat([left_prompt_embs, emb_a.unsqueeze(1), right_prompt_embs], dim=1)
        
    #     # Get transformer backbone
    #     transformer = main_base.transformer if hasattr(main_base, 'transformer') else main_base
        
    #     # Process initial prompt through transformer with Flash Attention
    #     hidden_states, kv_cache = compute_with_flash_kv_cache(
    #         transformer, 
    #         prompt_embs,
    #         use_cache=True
    #     )
        
    #     # Get logits for last position
    #     logits = main_out(hidden_states[:, -1])
        
    #     # Start autoregressive generation
    #     current_position = prompt_embs.size(1)
    #     logits_list = []
    #     hard_ids_list = []
    #     output_embs_list = []
        
    #     for step in range(max_length):
    #         # Apply Gumbel-Softmax
    #         if gumbel_tau > 0:
    #             # For Flash Attention, avoid casting to float32 as it only supports fp16/bf16
    #             # Subtract max for numerical stability (detached)
    #             logits_stable = logits - logits.max(dim=-1, keepdim=True)[0].detach()
    #             ste_token_dist = torch.nn.functional.gumbel_softmax(
    #                 logits_stable, tau=max(gumbel_tau, 0.1), hard=True
    #             )
    #         else:
    #             # Straight-through estimator with hard argmax
    #             # For Flash Attention, avoid casting to float32
    #             # Subtract max for numerical stability (detached)
    #             logits_stable = logits - logits.max(dim=-1, keepdim=True)[0].detach()
    #             probs = torch.nn.functional.softmax(logits_stable, dim=-1)
    #             hard_indices = probs.argmax(dim=-1)
    #             ste_token_dist = torch.zeros_like(probs)
    #             ste_token_dist.scatter_(-1, hard_indices.unsqueeze(-1), 1.0)
    #             ste_token_dist = ste_token_dist - probs.detach() + probs
            
    #         # Get embedding using Gumbel weights
    #         emb_t_input = ste_token_dist @ input_emb_table
            
    #         # Store outputs
    #         logits_list.append(logits)
    #         output_embs_list.append(emb_t_input)
    #         hard_ids_list.append(ste_token_dist.argmax(dim=-1))
            
    #         # Process new token through transformer with Flash KV cache
    #         if step < max_length - 1:  # Don't process last token if we won't use it
    #             hidden_states, kv_cache = compute_with_flash_kv_cache(
    #                 transformer,
    #                 emb_t_input.unsqueeze(1), 
    #                 kv_cache,
    #                 position_offset=current_position
    #             )
    #             current_position += 1
                
    #             # Get logits for next token
    #             logits = main_out(hidden_states[:, -1])
        
    #     # Stack outputs
    #     logits_seq = torch.stack(logits_list, dim=1)
    #     hard_ids = torch.stack(hard_ids_list, dim=1)
    #     text_embs = torch.stack(output_embs_list, dim=1)
        
    #     return Generated(text_embs, logits_seq, hard_ids)
