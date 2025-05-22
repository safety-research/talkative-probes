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
    eye_init: bool = True            # YAML `eye_init`
    stop_grad_aprime: bool = False   # YAML `stop_grad_aprime`
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

        # Store flag for forward()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Project final token embedding back to activation space."""
        # The Encoder base model processes the embeddings from the Decoder
        # to obtain hidden states.
        # We expect `embeddings` to be `decoder_output.generated_text_embeddings`
        # which are (B, T_text, d_model).
        # The base model processes these to get contextualized hidden states.
        # Note: The original implementation of Encoder didn't use self.base in the forward pass.
        # If self.base is intended to process the embeddings, it should be called here.
        # For now, assuming the projection is from the last embedding directly as per prior code.
        # If self.base *is* used, its output (e.g., last_hidden_state) would be passed to self.proj.
        
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
