"""Frozen original model with single-layer activation swap hook."""

import torch
from transformers import AutoModelForCausalLM

__all__ = ["OrigWrapper"]


class OrigWrapper:
    """Thin wrapper around HF CausalLM that supports `forward_with_replacement`."""

    def __init__(self, model_name: str, load_in_8bit: bool = False) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=load_in_8bit)
        self.model.eval()
        # Ensure base model parameters do not require gradients
        for param in self.model.parameters():
            param.requires_grad = False
        self._monkeypatch_reshape_inputs()

    def _validate_layer_idx(self, layer_idx: int) -> None:
        """Checks if the provided layer_idx is valid for the model."""
        try:
            num_hidden_layers = self.model.config.num_hidden_layers
            if not (0 <= layer_idx < num_hidden_layers):
                raise ValueError(
                    f"Invalid layer index: {layer_idx}. "
                    f"Model has {num_hidden_layers} layers (indices 0 to {num_hidden_layers - 1})."
                )
        except AttributeError as e:
            # This handles cases where model.config.num_hidden_layers might not exist
            raise ValueError(
                "Could not determine the number of hidden layers from model.config. "
                f"AttributeError when checking layer {layer_idx}: {e}"
            ) from e

    def to(self, device: torch.device) -> "OrigWrapper":
        self.model.to(device)
        return self

    def forward_with_replacement(
        self,
        input_ids: torch.Tensor,
        new_activation: torch.Tensor,
        layer_idx: int,
        token_pos: int,
        *,
        no_grad: bool = True,
    ) -> "transformers.modeling_outputs.CausalLMOutput":
        """Forward pass where hidden_state[layer_idx][:, token_pos] is replaced."""

        from contextlib import nullcontext
        self._validate_layer_idx(layer_idx)


        grad_ctx = torch.no_grad() if no_grad else nullcontext()
        with grad_ctx:
            def _swap_hook(_, __, output):  # noqa: ANN001
                hidden = output[0]
                if new_activation.dim() == 1:
                    hidden[:, token_pos] = new_activation.unsqueeze(0).to(hidden.dtype)
                else:
                    hidden[:, token_pos] = new_activation.to(hidden.dtype)  # type: ignore[index]
                return (hidden,) + output[1:]

            try:
                # Attempt to get the target block for hooking.
                # Try LLaMA-style path first (e.g., model.model.layers[idx])
                target_block = self.model.get_submodule(f"model.layers.{layer_idx}")
            except AttributeError:
                # Fallback to GPT-2-style paths if LLaMA-style fails
                try:
                    # Original primary attempt for GPT-2 (e.g., model.transformer.h[idx])
                    target_block = self.model.transformer.h[layer_idx]  # type: ignore[attr-defined]
                except AttributeError:
                    # Original fallback for GPT-2
                    target_block = self.model.get_submodule(f"transformer.h.{layer_idx}")
                    # If this also fails, get_submodule will raise an AttributeError,
                    # which is appropriate to signal an unsupported model structure.

            handle = target_block.register_forward_hook(_swap_hook)
            out = self.model(input_ids=input_ids)
            handle.remove()
            return out

    def forward_with_replacement_vectorized(
        self,
        input_ids: torch.Tensor,
        new_activations: torch.Tensor,
        layer_idx: int,
        token_positions: torch.Tensor,
        *,
        no_grad: bool = True,
    ) -> "transformers.modeling_outputs.CausalLMOutput":
        """Vectorized forward pass replacing activations at multiple positions in a single layer.
        
        Args:
            input_ids: Input token ids of shape (B, seq_len)
            new_activations: Activations to insert of shape (B, hidden_dim)
            layer_idx: Layer index to replace activations at (same for all samples)
            token_positions: Position indices of shape (B,) where activations are replaced
            no_grad: Whether to run in no_grad context. Set to False if gradients
                     w.r.t. new_activations are needed.
            
        Returns:
            Model output with activations replaced at specified positions
            
        Note: All samples must use the same layer_idx. For mixed layers, use multiple
        calls to forward_with_replacement() instead.
        """

        from contextlib import nullcontext
        self._validate_layer_idx(layer_idx) # Validate layer index at the beginning

        grad_ctx = torch.no_grad() if no_grad else nullcontext()
        with grad_ctx:
            def _vectorized_swap_hook(_, __, output):  # noqa: ANN001
                hidden = output[0]  # (B, seq_len, dim)
                batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
                # Ensure token_positions is 1D - squeeze if needed
                token_pos = token_positions.squeeze() if token_positions.ndim > 1 else token_positions
                # Ensure dtype matches between source and destination
                hidden[batch_indices, token_pos] = new_activations.to(hidden.dtype)
                return (hidden,) + output[1:]

            try:
                # Attempt to get the target block for hooking.
                # Try LLaMA-style path first (e.g., model.model.layers[idx])
                target_block = self.model.get_submodule(f"model.layers.{layer_idx}")
            except AttributeError:
                # Fallback to GPT-2-style paths if LLaMA-style fails
                try:
                    # Original primary attempt for GPT-2 (e.g., model.transformer.h[idx])
                    target_block = self.model.transformer.h[layer_idx]  # type: ignore[attr-defined]
                except AttributeError:
                    # Original fallback for GPT-2
                    target_block = self.model.get_submodule(f"transformer.h.{layer_idx}")
                    # If this also fails, get_submodule will raise an AttributeError,
                    # which is appropriate to signal an unsupported model structure.

            handle = target_block.register_forward_hook(_vectorized_swap_hook)
            out = self.model(input_ids=input_ids)
            handle.remove()
            return out

    # ------------------------------------------------------------------
    # Convenience ----------------------------------------------------------------
    # ------------------------------------------------------------------

    def _monkeypatch_reshape_inputs(self) -> None:  # noqa: D401
        """HF tiny GPT-2 lacks ``reshape_inputs`` – tests expect it."""

        if not hasattr(self.model, "reshape_inputs"):
            self.model.reshape_inputs = lambda x, **__: {"input_ids": x}  # type: ignore[attr-defined]

    # Called in __init__
