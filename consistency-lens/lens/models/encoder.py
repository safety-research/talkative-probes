"""Encoder that reconstructs activation from decoder-generated text."""

import torch
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel
from dataclasses import dataclass
import logging
__all__ = ["Encoder", "EncoderConfig"]


@dataclass
class EncoderConfig:
    model_name: str
    base_model: bool = True          # YAML `base_model`
    projection_layer: bool = True    # YAML `projection_layer`
    use_base_model: bool = False     # YAML `use_base_model`
    embedding_head: bool = False     # YAML `embedding_head`
    eye_init: bool = True            # YAML `eye_init`
    stop_grad_aprime: bool = False   # YAML `stop_grad_aprime`
    soft_prompt_length: int = 0      # YAML `soft_prompt_length` - number of trainable soft prompt tokens
    trainable_soft_prompt: bool = True  # YAML `trainable_soft_prompt` - whether soft prompt is trainable
    soft_prompt_init_std: float = 0.1   # YAML `soft_prompt_init_std` - standard deviation for random initialization
    soft_prompt_init_text: str | None = None  # YAML `soft_prompt_init_text` - text to initialize from (overrides random init)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.soft_prompt_length < 0:
            raise ValueError(f"soft_prompt_length must be non-negative, got {self.soft_prompt_length}")
        
        if self.soft_prompt_init_std <= 0:
            raise ValueError(f"soft_prompt_init_std must be positive, got {self.soft_prompt_init_std}")
        
        if self.soft_prompt_length == 0 and self.soft_prompt_init_text is not None:
            raise ValueError("soft_prompt_init_text specified but soft_prompt_length is 0")
            
        if not self.base_model and self.use_base_model:
            raise ValueError("use_base_model requires base_model to be True")
log = logging.getLogger(__name__)


class Encoder(nn.Module):
    """Processes generated tokens and projects final hidden to activation dims."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.config = cfg
        self.base: PreTrainedModel = AutoModelForCausalLM.from_pretrained(cfg.model_name)
        d_model = self.base.config.hidden_size
        self._use_base = cfg.use_base_model
        if self._use_base:
            # Configure trainability of the base model
            for p in self.base.parameters():
                p.requires_grad_(cfg.base_model)
        else:
            self.base = None

        # This is Proj_E_hidden_to_A from the README
        self.proj = nn.Linear(d_model, d_model, bias=False)
        # Initialize as identity matrix
        if cfg.eye_init:
            nn.init.eye_(self.proj.weight)
            log.info("Initialized projection layer as identity matrix")
        # Configure trainability of the output projection layer
        for p in self.proj.parameters():
            p.requires_grad_(cfg.projection_layer)

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
                if hasattr(self.base, 'lm_head'):
                    for p in self.base.lm_head.parameters():
                        p.requires_grad_(cfg.embedding_head)
                else:
                    log.warning("Could not access output embeddings for freezing control")

        # Initialize soft prompt tokens if specified
        self.soft_prompt_embeddings = None
        if cfg.soft_prompt_length > 0:
            d_model = self.base.config.hidden_size if self._use_base else self.proj.in_features
            # Initialize soft prompt embeddings with random values by default
            self.soft_prompt_embeddings = nn.Parameter(
                torch.randn(cfg.soft_prompt_length, d_model) * cfg.soft_prompt_init_std
            )
            self.soft_prompt_embeddings.requires_grad_(cfg.trainable_soft_prompt)
            log.info(f"Initialized {cfg.soft_prompt_length} soft prompt tokens for encoder "
                    f"(trainable: {cfg.trainable_soft_prompt}, init_std: {cfg.soft_prompt_init_std})")

    def set_soft_prompt_from_text(self, text: str, tokenizer) -> None:
        """Initialize soft prompt embeddings from text string using tokenizer.
        
        Args:
            text: Text string to convert to embeddings for soft prompt initialization
            tokenizer: Tokenizer to use for text conversion
        """
        if self.soft_prompt_embeddings is None:
            log.warning("No soft prompt embeddings to initialize (soft_prompt_length=0)")
            return
            
        # Tokenize the text
        token_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)
        
        # Get device from existing soft prompt embeddings
        device = self.soft_prompt_embeddings.device
        token_ids = token_ids.to(device)
        
        # Get embeddings from the base model
        if self._use_base:
            emb_table = self.base.get_input_embeddings().weight
        else:
            # If not using base model, we need some embedding table - use the base model anyway
            emb_table = self.base.get_input_embeddings().weight
            
        # Handle length mismatch
        target_length = self.soft_prompt_embeddings.shape[0]
        if len(token_ids) > target_length:
            # Truncate if text is too long
            token_ids = token_ids[:target_length]
            log.warning(f"Text too long for soft prompt, truncated to {target_length} tokens")
        elif len(token_ids) < target_length:
            # Repeat if text is too short
            repeats = (target_length + len(token_ids) - 1) // len(token_ids)  # Ceiling division
            token_ids = token_ids.repeat(repeats)[:target_length]
            log.info(f"Text too short for soft prompt, repeated to {target_length} tokens")
            
        # Get embeddings and initialize the soft prompt
        with torch.no_grad():
            text_embeddings = emb_table[token_ids]  # Shape: (target_length, d_model)
            self.soft_prompt_embeddings.data.copy_(text_embeddings)
            
        log.info(f"Initialized encoder soft prompt from text: '{text}' "
                f"({len(tokenizer(text, add_special_tokens=False).input_ids)} -> {target_length} tokens)")

        # Store flag for forward()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Project final token embedding back to activation space."""
        # The Encoder base model processes the embeddings from the Decoder
        # to obtain hidden states.
        # We expect `embeddings` to be `decoder_output.generated_text_embeddings`
        # which are (B, T_text, d_model).
        
        # Prepend soft prompt tokens if configured
        if self.soft_prompt_embeddings is not None:
            B = embeddings.shape[0]
            # Expand soft prompt for batch size: (soft_prompt_length, d_model) -> (B, soft_prompt_length, d_model)
            soft_prompt_expanded = self.soft_prompt_embeddings.unsqueeze(0).expand(B, -1, -1)
            # Concatenate: [soft_prompt, decoder_generated_tokens]
            embeddings = torch.cat([soft_prompt_expanded, embeddings], dim=1)  # (B, soft_prompt_length + T_text, d_model)
        
        # If the Encoder's base model is meant to process the embeddings first:
        if self._use_base:
            # Pass embeddings through the base model; request hidden_states
            outputs = self.base(inputs_embeds=embeddings, output_hidden_states=True)
            if outputs.hidden_states is None:
                raise RuntimeError("Model did not return hidden_states despite request.")
            processed_embeddings = outputs.hidden_states[-1]
            # Then take the embedding of the final token for projection
            last_emb_to_proj = processed_embeddings[:, -1] # (B, d_model)
        else:
            # Project the last token embedding directly if `self.base` is not in use.
            last_emb_to_proj = embeddings[:, -1]  # (B, d_model)
            
        return self.proj(last_emb_to_proj)
