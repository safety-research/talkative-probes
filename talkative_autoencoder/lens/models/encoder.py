"""Encoder that reconstructs activation from decoder-generated text."""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from transformers import AutoModelForCausalLM, PreTrainedModel

__all__ = ["Encoder", "EncoderConfig"]

log = logging.getLogger(__name__)


@dataclass
class EncoderConfig:
    model_name: str
    base_model: bool = True  # YAML `base_model`
    pos_embeddings: bool = True  # YAML `pos_embeddings` - whether to train positional embeddings
    projection_layer: bool = True  # YAML `projection_layer`
    use_base_model: bool = True  # YAML `use_base_model`
    use_projection_layer: bool = True  # YAML `use_projection_layer`
    embedding_head: bool = False  # YAML `embedding_head`
    eye_init: bool = True  # YAML `eye_init`
    stop_grad_aprime: bool = False  # YAML `stop_grad_aprime`
    soft_prompt_length: int = 0  # YAML `soft_prompt_length` - number of trainable soft prompt tokens
    trainable_soft_prompt: bool = True  # YAML `trainable_soft_prompt` - whether soft prompt is trainable
    soft_prompt_init_std: float = 0.1  # YAML `soft_prompt_init_std` - standard deviation for random initialization
    soft_prompt_init_text: str | None = (
        None  # YAML `soft_prompt_init_text` - text to initialize from (overrides random init)
    )
    output_layer: int = -1  # YAML `output_layer` - which layer to extract activations from (-1=last, -2=pre-first-layer embeddings, 0..N-1 for specific layers)
    use_dropout: bool = True  # YAML `use_dropout` - whether to use dropout during training (False = deterministic)
    projection_init_method: str = "default"  # YAML `projection_init_method` - how to initialize the projection layer
    subtract_add_pos_embeddings: bool = False  # YAML `subtract_add_pos_embeddings`
    extra_pos_embeddings: bool = (
        False  # YAML `extra_pos_embeddings` - whether to add extra positional embeddings to the activation
    )
    add_current_token: bool = (
        False  # YAML `add_current_token` - whether to add the current token to the input to the encoder
    )
    special_last_token_vector: bool = (
        False  # YAML `special_last_token_vector` - whether to use a special last token vector
    )
    attn_implementation: str = None

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.soft_prompt_length < 0:
            raise ValueError(f"soft_prompt_length must be non-negative, got {self.soft_prompt_length}")

        if self.soft_prompt_init_std <= 0:
            raise ValueError(f"soft_prompt_init_std must be positive, got {self.soft_prompt_init_std}")

        # If text initialization is specified, soft_prompt_length is ignored (text determines length)
        if self.soft_prompt_init_text is not None and self.soft_prompt_length > 0:
            log.warning(
                f"soft_prompt_length={self.soft_prompt_length} will be ignored since "
                f"soft_prompt_init_text is specified (length will be determined by text)"
            )

        # Can't train a base model if we're not using it
        if not self.use_base_model and self.base_model:
            raise ValueError("base_model=True requires use_base_model=True (can't train a model that isn't being used)")

        # Can't train embedding heads if we're not using the base model
        if not self.use_base_model and self.embedding_head:
            raise ValueError(
                "embedding_head=True requires use_base_model=True (can't train embeddings of a model that isn't being used)"
            )

        # Validate output_layer - can only extract from specific layers if using base model
        if not self.use_base_model and self.output_layer != -1:
            raise ValueError(
                f"output_layer={self.output_layer} requires use_base_model=True (can't extract from specific layers without base model)"
            )


class Encoder(nn.Module):
    """Processes generated tokens and projects final hidden to activation dims."""

    def __init__(self, cfg: EncoderConfig, base_to_use: Optional[PreTrainedModel] = None):
        super().__init__()
        self.config = cfg

        # Use provided base model or load a new one
        if base_to_use is not None and not cfg.base_model and cfg.use_base_model:
            # If we're not training the base model and one is provided, use it
            self.base = base_to_use
            log.info("Using shared base model for Encoder (memory efficient)")
        else:
            # Load our own copy
            self.base: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                cfg.model_name, attn_implementation=cfg.attn_implementation
            )

        d_model = (
            self.base.config.hidden_size
            if not hasattr(self.base.config, "text_config")
            else self.base.config.text_config.hidden_size
        )
        self._use_base = cfg.use_base_model
        if self._use_base:
            # Configure trainability of the base model
            for p in self.base.parameters():
                p.requires_grad_(cfg.base_model)
        else:
            self.base = None

        self.is_gemma2 = hasattr(self.base.config, "model_type") and "gemma2" in self.base.config.model_type.lower()
        self.is_gemma3 = (
            hasattr(self.base.config, "model_type") and "gemma3" in self.base.config.model_type.lower()
        ) or (hasattr(self.base.config, "text_config") and "gemma3" in self.base.config.text_config.model_type.lower())
        log.info(f"model_name: {cfg.model_name}, is_gemma2: {self.is_gemma2}, is_gemma3: {self.is_gemma3}")
        self.is_gemma = self.is_gemma2 or self.is_gemma3
        self.normalizer = d_model**0.5

        if hasattr(self.base, "transformer") and hasattr(self.base.transformer, "wpe"):
            self.base.transformer.wpe.weight.requires_grad_(cfg.pos_embeddings)
            log.info("Set requires_grad for positional embeddings in base model (wpe)")
        elif hasattr(self.base, "model") and hasattr(self.base.model, "embed_positions"):
            self.base.model.embed_positions.weight.requires_grad_(cfg.pos_embeddings)
            log.info("Set requires_grad for positional embeddings in base model (embed_positions)")

        # if hasattr(self.base, "transformer") and hasattr(self.base.transformer, "wte"):
        #     self.base.transformer.wte.weight.requires_grad_(False)
        #     log.info("Set requires_grad for input embeddings in base model (wte) to False")
        #     log.info(f"Separately control 'input embeddings' trainability with 'embedding_head'")

        # This is Proj_E_hidden_to_A from the README
        if cfg.use_projection_layer:
            self.proj = nn.Linear(d_model, d_model, bias=True)
            # Initialize as identity matrix
            if cfg.eye_init:
                nn.init.eye_(self.proj.weight)
                nn.init.zeros_(self.proj.bias)  # Also initialize bias to zeros with eye_init
                log.info("Initialized projection layer as identity matrix (encoder)")
            # Configure trainability of the output projection layer
            for p in self.proj.parameters():
                p.requires_grad_(cfg.projection_layer)
        else:
            self.proj = None

        # Configure trainability of the embedding heads (input/output embeddings) if using base model
        if self._use_base:
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
                if hasattr(self.base, "lm_head"):
                    for p in self.base.lm_head.parameters():
                        p.requires_grad_(cfg.embedding_head)
                else:
                    log.warning("Could not access output embeddings for freezing control")

        # Initialize soft prompt tokens
        self.soft_prompt_embeddings = None
        self.soft_prompt_embeddings_postfix = None
        self.cfg = cfg  # Store config for later use in set_soft_prompt_from_text

        # If text initialization is specified, we'll set the length based on the text
        if cfg.soft_prompt_init_text is not None:
            # We'll initialize this later after models are created, just set a placeholder
            pass
        elif cfg.soft_prompt_length > 0:
            # Initialize with random values for the specified length
            d_model = (
                (
                    self.base.config.hidden_size
                    if not hasattr(self.base.config, "text_config")
                    else self.base.config.text_config.hidden_size
                )
                if self._use_base
                else self.proj.in_features
            )
            self.soft_prompt_embeddings = nn.Parameter(
                torch.randn(cfg.soft_prompt_length, d_model) * cfg.soft_prompt_init_std
            )
            self.soft_prompt_embeddings.requires_grad_(cfg.trainable_soft_prompt)
            self.soft_prompt_embeddings_postfix = None
            log.info(
                f"Initialized {cfg.soft_prompt_length} soft prompt tokens for encoder "
                f"(trainable: {cfg.trainable_soft_prompt}, init_std: {cfg.soft_prompt_init_std})"
            )

        # Activation-specific positional embedder
        self.activation_pos_embedder = None
        if self.config.subtract_add_pos_embeddings:
            num_pos_embeddings = self.base.config.max_position_embeddings
            self.activation_pos_embedder = nn.Embedding(num_pos_embeddings, d_model)
            # Initialize weights (e.g., small random values or from base model's pos embeddings)
            self.activation_pos_embedder.weight.data = self.base.transformer.wpe.weight.data.clone()
            self.activation_pos_embedder.weight.requires_grad_(cfg.extra_pos_embeddings)  # Ensure trainable
            log.info(f"Initialized activation positional embedder for Encoder with {num_pos_embeddings} embeddings.")
        if self.config.special_last_token_vector:
            self.special_last_token_vector = nn.Parameter(torch.zeros(d_model, dtype=self.base.dtype))
            self.special_last_token_vector.requires_grad_(True)
            log.info("Initialized special last token vector")

        # Lazy hook management for hidden state capture
        self._capture_hooks = {}  # layer_idx -> hook handle
        self._captured_states = {}  # layer_idx -> captured tensor
        self._capture_enabled = {}  # layer_idx -> bool

    def to(self, device: torch.device):
        super().to(device)
        if self.proj is not None:
            self.proj.to(device)
        if self.activation_pos_embedder is not None:
            self.activation_pos_embedder.to(device)
        if self.base is not None:
            self.base.to(device)
        return self

    def set_soft_prompt_from_text(self, string: str, tokenizer) -> None:
        """Initialize soft prompt embeddings from text string using tokenizer.
        When called, this will create or recreate the soft prompt with length matching the tokenized text.

        Args:
            text: Text string to convert to embeddings for soft prompt initialization
            tokenizer: Tokenizer to use for text conversion
        """
        if "<text>" in string:
            prefix, postfix = string.split("<text>")
        else:
            prefix = string
            postfix = None

        if postfix == "":
            postfix = None
        if prefix == "":
            prefix = None
        # Tokenize the text
        token_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)
        if postfix is not None:
            token_ids_postfix = tokenizer(postfix, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)
        text_length = len(token_ids)
        self.prefix_token_ids = token_ids

        if postfix is not None:
            self.postfix_token_ids = token_ids_postfix
        else:
            self.postfix_token_ids = None

        # Get model dimensions
        d_model = (
            (
                self.base.config.hidden_size
                if not hasattr(self.base.config, "text_config")
                else self.base.config.text_config.hidden_size
            )
            if self._use_base
            else self.proj.in_features
        )

        # Get device - use the model's device
        device = next(self.parameters()).device
        token_ids = token_ids.to(device)

        # Delete old soft prompt if it exists
        if hasattr(self, "soft_prompt_embeddings") and self.soft_prompt_embeddings is not None:
            delattr(self, "soft_prompt_embeddings")

        # Create new soft prompt with the exact length of the text
        self.soft_prompt_embeddings = nn.Parameter(torch.zeros(text_length, d_model, device=device, dtype=self.base.dtype))
        self.soft_prompt_embeddings.requires_grad_(self.cfg.trainable_soft_prompt)
        self.soft_prompt_embeddings_postfix = None
        if postfix is not None:
            self.soft_prompt_embeddings_postfix = nn.Parameter(
                torch.zeros(len(token_ids_postfix), d_model, device=device, dtype=self.base.dtype)
            )
            self.soft_prompt_embeddings_postfix.requires_grad_(self.cfg.trainable_soft_prompt)
            log.info(
                f"Initialized encoder soft prompt postfix with {len(token_ids_postfix)} tokens, requires grad {self.cfg.trainable_soft_prompt}",
            )

        # Get embeddings from the base model
        if self._use_base:
            emb_table = self.base.get_input_embeddings().weight
        else:
            # If not using base model, we need some embedding table - use the base model anyway
            emb_table = self.base.get_input_embeddings().weight

        # Initialize the soft prompt with text embeddings
        with torch.no_grad():
            text_embeddings = emb_table[token_ids]  # Shape: (text_length, d_model)
            # if self._use_base and self.is_gemma3:
            #     text_embeddings = text_embeddings * self.normalizer
            #     print("gemma3,s so multiplied by normalizer whem setting soft prompt")
            self.soft_prompt_embeddings.data.copy_(text_embeddings)
            if postfix is not None:
                self.soft_prompt_embeddings_postfix.data.copy_(emb_table[token_ids_postfix])
                # if self._use_base and self.is_gemma3:
                #     self.soft_prompt_embeddings_postfix.data = (
                #         self.soft_prompt_embeddings_postfix.data * self.normalizer
                #     )
                #     print("gemma3,s so multiplied by normalizer whem setting soft prompt postfix")
        # if 'gemma3' in self.base.config.model_type:
        # don't need normalizer as we are not using the base model....

        log.info(
            f"Initialized encoder soft prompt from text: '{prefix}' ({text_length} tokens, "
            + (
                f"postfix: '{postfix}' ({len(token_ids_postfix) if postfix is not None else 0} tokens, "
                if postfix is not None
                else "(no postfix) "
            )
            + f"   trainable: {self.cfg.trainable_soft_prompt})"
        )

        # Store flag for forward()

    @contextmanager
    def _maybe_disable_dropout(self):
        """Context manager to control dropout behavior based on config."""
        if not self.config.use_dropout and self._use_base:
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

    def _ensure_capture_hook_registered(self, layer_idx: int):
        """Register capture hook for a layer if not already registered."""
        if layer_idx in self._capture_hooks:
            return

        # Get the layers
        layers = self._get_layers(self.base)

        if layer_idx == -2:
            # Special case: capture input to first layer
            def pre_hook(module, args, kwargs):
                if self._capture_enabled.get(layer_idx, False):
                    if args:
                        self._captured_states[layer_idx] = args[0]
                    elif "hidden_states" in kwargs:
                        self._captured_states[layer_idx] = kwargs["hidden_states"]
                # Return None to not modify the input
                return None

            self._capture_hooks[layer_idx] = layers[0].register_forward_pre_hook(pre_hook, with_kwargs=True)
        else:
            # Regular layer output capture
            actual_idx = layer_idx if layer_idx >= 0 else len(layers) + layer_idx

            def post_hook(module, args, output):
                if self._capture_enabled.get(layer_idx, False):
                    self._captured_states[layer_idx] = output[0] if isinstance(output, tuple) else output
                return output

            self._capture_hooks[layer_idx] = layers[actual_idx].register_forward_hook(post_hook)

        self._capture_enabled[layer_idx] = False

    def _get_layers(self, model):
        """Helper to find the list of transformer layers in a model."""
        if hasattr(model, "model"):
            model = model.model
        if hasattr(model, "transformer"):
            model = model.transformer
        if hasattr(model, "layers"):
            return model.layers
        if hasattr(model, "h"):
            return model.h
        if hasattr(model, "blocks"):
            return model.blocks
        if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            return model.encoder.layer
        if hasattr(model, "decoder") and hasattr(model.decoder, "layer"):
            return model.decoder.layer
        raise AttributeError("Could not find transformer layers in the base model.")

    def forward(
        self,
        embeddings: torch.Tensor,
        original_token_pos: Optional[torch.Tensor] = None,
        current_token_ids: Optional[torch.Tensor] = None,
        add_special_last_token_vector: bool = True,
        use_hard_token_ids: bool = False,
    ) -> torch.Tensor:  # type: ignore[override]
        """Project final token embedding back to activation space."""
        # The Encoder base model processes the embeddings from the Decoder
        # to obtain hidden states.
        # We expect `embeddings` to be `decoder_output.generated_text_embeddings`
        # which are (B, t_text, d_model).

        # don't need a normalizer, as we are not using the full CausalLM
        # Prepend soft prompt tokens if configured
        if self.soft_prompt_embeddings is not None:
            B = embeddings.shape[0]
            # Expand soft prompt for batch size: (soft_prompt_length, d_model) -> (B, soft_prompt_length, d_model)
            soft_prompt_expanded = self.soft_prompt_embeddings.unsqueeze(0)
            if self.is_gemma3:
                soft_prompt_expanded = soft_prompt_expanded * self.normalizer
            soft_prompt_expanded = soft_prompt_expanded.expand(B, -1, -1)
            # if self._use_base and hasattr(self.base.config, "model_type") and 'gemma3' in self.base.config.model_type:#PRE SCALED!!!
            #     hidden_size = self.base.config.hidden_size if not hasattr(self.base.config, "text_config") else self.base.config.text_config.hidden_size
            #     soft_prompt_expanded = soft_prompt_expanded * (hidden_size ** 0.5)

            if self.soft_prompt_embeddings_postfix is not None:
                soft_prompt_expanded_postfix = self.soft_prompt_embeddings_postfix.unsqueeze(0)
                if self.is_gemma3:
                    soft_prompt_expanded_postfix = soft_prompt_expanded_postfix * self.normalizer
                soft_prompt_expanded_postfix = soft_prompt_expanded_postfix.expand(B, -1, -1)
                # if self._use_base and hasattr(self.base.config, "model_type") and 'gemma3' in self.base.config.model_type:
                #     hidden_size = self.base.config.hidden_size if not hasattr(self.base.config, "text_config") else self.base.config.text_config.hidden_size
                #     soft_prompt_expanded_postfix = soft_prompt_expanded_postfix * (hidden_size ** 0.5)####PRE SCALED!!!
                # Concatenate: [soft_prompt, decoder_generated_tokens]
                embeddings = torch.cat(
                    [soft_prompt_expanded, embeddings, soft_prompt_expanded_postfix], dim=1
                )  # (B, soft_prompt_length + t_text, d_model)
            else:
                # Concatenate: [soft_prompt, decoder_generated_tokens]
                embeddings = torch.cat(
                    [soft_prompt_expanded, embeddings], dim=1
                )  # (B, soft_prompt_length + t_text, d_model)

        if self.config.add_current_token and current_token_ids is not None:
            current_token_vectors = self.base.get_input_embeddings().weight[current_token_ids]
            if self._use_base and self.is_gemma3:
                current_token_vectors = current_token_vectors * self.normalizer
            embeddings = torch.cat([embeddings, current_token_vectors.unsqueeze(1)], dim=1)

        if self.config.special_last_token_vector and add_special_last_token_vector:
            if self._use_base and self.is_gemma3:
                special_last_token_vector = self.special_last_token_vector * self.normalizer
                special_last_token_vector = special_last_token_vector.unsqueeze(0).expand_as(embeddings[:, -1])
            else:
                special_last_token_vector = self.special_last_token_vector.unsqueeze(0).expand_as(embeddings[:, -1])
            embeddings[:, -1] += special_last_token_vector

        # If the Encoder's base model is meant to process the embeddings first:
        if self._use_base:
            with self._maybe_disable_dropout():
                # Ensure hook is registered for the layer we want
                self._ensure_capture_hook_registered(self.config.output_layer)

                # Enable capture for this forward pass
                self._capture_enabled[self.config.output_layer] = True
                self._captured_states[self.config.output_layer] = None

                try:
                    # Run forward pass
                    self.base(inputs_embeds=embeddings)

                    # Get captured output
                    captured_output = self._captured_states[self.config.output_layer]
                    if captured_output is None:
                        raise RuntimeError(f"Failed to capture hidden state at layer {self.config.output_layer}")

                    processed_embeddings = captured_output
                finally:
                    # Disable capture (but keep hook registered)
                    self._capture_enabled[self.config.output_layer] = False
                    self._captured_states[self.config.output_layer] = None

                # Take the embedding of the final token for projection
                last_emb_to_proj = processed_embeddings[:, -1]
        else:
            # Project the last token embedding directly if `self.base` is not in use.
            last_emb_to_proj = embeddings[:, -1]  # (B, d_model)

        if self.proj is not None:
            A_hat_intermediate = self.proj(last_emb_to_proj)
        else:
            A_hat_intermediate = last_emb_to_proj

        if (
            self.config.subtract_add_pos_embeddings
            and original_token_pos is not None
            and self.activation_pos_embedder is not None
        ):
            # Ensure original_token_pos is long type and on the correct device
            original_token_pos = original_token_pos.to(device=A_hat_intermediate.device, dtype=torch.long)
            original_token_pos = original_token_pos.squeeze() if original_token_pos.dim() == 2 else original_token_pos
            # Get positional embeddings
            # Handle cases where original_token_pos might be scalar for a batch of size 1
            # raise ValueError(f"original_token_pos has shape {original_token_pos.shape} but A_hat_intermediate has shape {A_hat_intermediate.shape}")

            pos_embeds_to_add = self.activation_pos_embedder(original_token_pos)  # (B, d_model)

            # # Ensure pos_embeds_to_add has the same shape as A_hat_intermediate
            # if pos_embeds_to_add.shape != A_hat_intermediate.shape:
            #     # This might happen if original_token_pos was scalar and then unsqueezed for a batch > 1
            #     # or if broadcasting rules didn't align. For safety, explicitly check.
            #     # Example: pos_embeds_to_add could be (1, d_model) if original_token_pos became (1,)
            #     # while A_hat_intermediate is (B, d_model).
            #     if pos_embeds_to_add.shape[0] == 1 and A_hat_intermediate.shape[0] > 1:
            #         pos_embeds_to_add = pos_embeds_to_add.expand_as(A_hat_intermediate)
            #     else:
            #         raise ValueError(
            #             f"Shape mismatch for positional embedding addition: "
            #             f"A_hat_intermediate shape {A_hat_intermediate.shape}, "
            #             f"pos_embeds_to_add shape {pos_embeds_to_add.shape}"
            #         )

            final_A_hat = A_hat_intermediate + pos_embeds_to_add
        else:
            final_A_hat = A_hat_intermediate

        return final_A_hat
