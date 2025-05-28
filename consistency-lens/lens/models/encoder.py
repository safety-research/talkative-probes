"""Encoder that reconstructs activation from decoder-generated text."""

import torch
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel
from dataclasses import dataclass
import logging
__all__ = ["Encoder", "EncoderConfig"]

log = logging.getLogger(__name__)


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
    output_layer: int = -1           # YAML `output_layer` - which layer to extract activations from (-1 = last layer)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.soft_prompt_length < 0:
            raise ValueError(f"soft_prompt_length must be non-negative, got {self.soft_prompt_length}")
        
        if self.soft_prompt_init_std <= 0:
            raise ValueError(f"soft_prompt_init_std must be positive, got {self.soft_prompt_init_std}")
        
        # If text initialization is specified, soft_prompt_length is ignored (text determines length)
        if self.soft_prompt_init_text is not None and self.soft_prompt_length > 0:
            log.warning(f"soft_prompt_length={self.soft_prompt_length} will be ignored since "
                       f"soft_prompt_init_text is specified (length will be determined by text)")
            
        # Can't train a base model if we're not using it
        if not self.use_base_model and self.base_model:
            raise ValueError("base_model=True requires use_base_model=True (can't train a model that isn't being used)")
            
        # Can't train embedding heads if we're not using the base model
        if not self.use_base_model and self.embedding_head:
            raise ValueError("embedding_head=True requires use_base_model=True (can't train embeddings of a model that isn't being used)")
            
        # Validate output_layer - can only extract from specific layers if using base model
        if not self.use_base_model and self.output_layer != -1:
            raise ValueError(f"output_layer={self.output_layer} requires use_base_model=True (can't extract from specific layers without base model)")


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
        self.proj = nn.Linear(d_model, d_model, bias=True)
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

        # Initialize soft prompt tokens
        self.soft_prompt_embeddings = None
        self.cfg = cfg  # Store config for later use in set_soft_prompt_from_text
        
        # If text initialization is specified, we'll set the length based on the text
        if cfg.soft_prompt_init_text is not None:
            # We'll initialize this later after models are created, just set a placeholder
            pass
        elif cfg.soft_prompt_length > 0:
            # Initialize with random values for the specified length
            d_model = self.base.config.hidden_size if self._use_base else self.proj.in_features
            self.soft_prompt_embeddings = nn.Parameter(
                torch.randn(cfg.soft_prompt_length, d_model) * cfg.soft_prompt_init_std
            )
            self.soft_prompt_embeddings.requires_grad_(cfg.trainable_soft_prompt)
            log.info(f"Initialized {cfg.soft_prompt_length} soft prompt tokens for encoder "
                    f"(trainable: {cfg.trainable_soft_prompt}, init_std: {cfg.soft_prompt_init_std})")

    def set_soft_prompt_from_text(self, text: str, tokenizer) -> None:
        """Initialize soft prompt embeddings from text string using tokenizer.
        When called, this will create or recreate the soft prompt with length matching the tokenized text.
        
        Args:
            text: Text string to convert to embeddings for soft prompt initialization
            tokenizer: Tokenizer to use for text conversion
        """
        # Tokenize the text
        token_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)
        text_length = len(token_ids)
        
        # Get model dimensions
        d_model = self.base.config.hidden_size if self._use_base else self.proj.in_features
        
        # Get device - use the model's device
        device = next(self.parameters()).device
        token_ids = token_ids.to(device)
        
        # Delete old soft prompt if it exists
        if hasattr(self, 'soft_prompt_embeddings') and self.soft_prompt_embeddings is not None:
            delattr(self, 'soft_prompt_embeddings')
        
        # Create new soft prompt with the exact length of the text
        self.soft_prompt_embeddings = nn.Parameter(torch.zeros(text_length, d_model, device=device))
        self.soft_prompt_embeddings.requires_grad_(self.cfg.trainable_soft_prompt)
        
        # Get embeddings from the base model
        if self._use_base:
            emb_table = self.base.get_input_embeddings().weight
        else:
            # If not using base model, we need some embedding table - use the base model anyway
            emb_table = self.base.get_input_embeddings().weight
            
        # Initialize the soft prompt with text embeddings
        with torch.no_grad():
            text_embeddings = emb_table[token_ids]  # Shape: (text_length, d_model)
            self.soft_prompt_embeddings.data.copy_(text_embeddings)
            
        log.info(f"Initialized encoder soft prompt from text: '{text}' ({text_length} tokens, "
                f"trainable: {self.cfg.trainable_soft_prompt})")

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
            
            # Select the appropriate layer based on output_layer config
            if self.config.output_layer == -1:
                # Last layer (default behavior)
                processed_embeddings = outputs.hidden_states[-1]
            elif self.config.output_layer == 0:
                # Input embeddings (before any transformer layers)
                processed_embeddings = outputs.hidden_states[0]
            else:
                # Specific layer (0-indexed, so we shift by 1)
                # hidden_states[1] = output of first transformer layer 0
                # hidden_states[n] = output of nth transformer layer n-1
                if self.config.output_layer > len(outputs.hidden_states) - 2:
                    raise ValueError(f"Requested output_layer {self.config.output_layer} but model only has {len(outputs.hidden_states)-1} hidden states (0 to {len(outputs.hidden_states) - 2})")
                processed_embeddings = outputs.hidden_states[self.config.output_layer+1]
            
            # Then take the embedding of the final token for projection
            last_emb_to_proj = processed_embeddings[:, -1] # (B, d_model)
        else:
            # Project the last token embedding directly if `self.base` is not in use.
            last_emb_to_proj = embeddings[:, -1]  # (B, d_model)
            
        return self.proj(last_emb_to_proj)
