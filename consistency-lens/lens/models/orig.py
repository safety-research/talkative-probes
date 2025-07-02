"""Frozen original model with single-layer activation swap hook."""

import torch
import transformers
from transformers import AutoModelForCausalLM
from typing import Optional, Tuple, Union
from contextlib import nullcontext

__all__ = ["OrigWrapper"]


class OrigWrapper:
    """Thin wrapper around HF CausalLM that supports `forward_with_replacement`."""

    # Define the replacement function as a static method
    @staticmethod
    def _static_reshape_inputs(x, **__) -> dict:
        return {"input_ids": x}

    def __init__(self, model_name: str, load_in_8bit: bool = False, base_to_use = None) -> None:
        if base_to_use is None:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=load_in_8bit)
        else:
            self.model = base_to_use
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
        attention_mask: torch.Tensor | None = None,
        no_grad: bool = True,
    ) -> transformers.modeling_outputs.CausalLMOutput:
        """Forward pass where hidden_state[layer_idx][:, token_pos] is replaced."""

        from contextlib import nullcontext
        self._validate_layer_idx(layer_idx)

        # If no attention mask is provided, create one from pad tokens
        if attention_mask is None and self.model.config.pad_token_id is not None:
            attention_mask = input_ids.ne(self.model.config.pad_token_id).long()

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
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            handle.remove()
            return out

    def forward_with_replacement_vectorized(
        self,
        input_ids: torch.Tensor,
        new_activations: torch.Tensor,
        layer_idx: int,
        token_positions: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        no_grad: bool = True,
    ) -> transformers.modeling_outputs.CausalLMOutput:
        """Vectorized forward pass replacing activations at multiple positions in a single layer.
        
        Args:
            input_ids: Input token ids of shape (B, seq_len)
            new_activations: Activations to insert of shape (B, hidden_dim)
            layer_idx: Layer index to replace activations at (same for all samples)
            token_positions: Position indices of shape (B,) where activations are replaced
            attention_mask: Optional attention mask. If None, created from pad tokens.
            no_grad: Whether to run in no_grad context. Set to False if gradients
                     w.r.t. new_activations are needed.
            
        Returns:
            Model output with activations replaced at specified positions
            
        Note: All samples must use the same layer_idx. For mixed layers, use multiple
        calls to forward_with_replacement() instead.
        """

        from contextlib import nullcontext
        self._validate_layer_idx(layer_idx) # Validate layer index at the beginning

        # If no attention mask is provided, create one from pad tokens
        if attention_mask is None and self.model.config.pad_token_id is not None:
            attention_mask = input_ids.ne(self.model.config.pad_token_id).long()

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
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            handle.remove()
            return out

    def get_activations_at_positions(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
        token_positions: Optional[Union[int, torch.Tensor]] = None,
        min_pos_to_select_from: Optional[int] = None,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        no_grad: bool = True,
        position_selection_strategy: str = 'midpoint',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts hidden state activations from a specified layer at given or calculated token positions
        using a forward hook.

        Args:
            input_ids: Input token ids, shape (B, seq_len) or (seq_len).
            layer_idx: 0-indexed layer to extract activations from.
            token_positions: Optional. Exact token positions (0-indexed) to extract activations from.
                             Can be an int for a single sequence or a 1D tensor of shape (B,) for a batch.
                             If provided, `min_pos_to_select_from` is ignored.
            min_pos_to_select_from: Optional. If `token_positions` is None, this is used to calculate
                                    the token positions. The position is calculated as the middle
                                    of the non-padding tokens, starting from at least this index.
                                    Defaults to 0 if not specified and calculating positions.
            attention_mask: Optional attention mask, shape (B, seq_len). If None and pad_token_id
                            is configured, it's created from pad tokens.
            no_grad: Whether to run in no_grad context.
            position_selection_strategy: How to select the token position if not provided.
                                         'midpoint' (default) or 'random'.

        Returns:
            A tuple containing:
            - selected_activations: Tensor of shape (B, hidden_dim) or (hidden_dim)
                                    containing the activations.
            - calculated_token_positions: Tensor of shape (B,) or scalar tensor
                                          containing the token positions used.
        """
        self._validate_layer_idx(layer_idx)

        if token_positions is None and min_pos_to_select_from is None:
            min_pos_to_select_from = 0
        elif token_positions is not None and min_pos_to_select_from is not None:
            pass

        was_1d_input = False
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
            was_1d_input = True
            if isinstance(token_positions, int):
                token_positions = torch.tensor([token_positions], device=input_ids.device, dtype=torch.long)
        
        if not isinstance(input_ids, torch.Tensor):
             raise TypeError(f"input_ids must be a torch.Tensor, got {type(input_ids)}")

        batch_size = input_ids.shape[0]

        if attention_mask is None and self.model.config.pad_token_id is not None:
            attention_mask = input_ids.ne(self.model.config.pad_token_id).long()

        calculated_positions: torch.Tensor
        if token_positions is not None:
            if isinstance(token_positions, int):
                calculated_positions = torch.tensor([token_positions] * batch_size, device=input_ids.device, dtype=torch.long)
            elif isinstance(token_positions, torch.Tensor):
                if token_positions.ndim == 0:
                    calculated_positions = token_positions.repeat(batch_size).to(input_ids.device, dtype=torch.long)
                elif token_positions.ndim == 1 and token_positions.shape[0] == 1 and batch_size > 1:
                     calculated_positions = token_positions.repeat(batch_size).to(input_ids.device, dtype=torch.long)
                elif token_positions.ndim == 1 and token_positions.shape[0] == batch_size:
                    calculated_positions = token_positions.to(input_ids.device, dtype=torch.long)
                else:
                    raise ValueError(
                        f"token_positions tensor shape {token_positions.shape} "
                        f"is incompatible with batch size {batch_size}."
                    )
            else:
                raise TypeError(f"token_positions must be int, torch.Tensor, or None, got {type(token_positions)}")
        elif min_pos_to_select_from is not None:
            pad_token_id = self.model.config.pad_token_id
            
            if pad_token_id is None:
                # If no pad token, sequence is the full length
                start_indices = torch.zeros(batch_size, device=input_ids.device, dtype=torch.long)
                end_indices = torch.full((batch_size,), input_ids.shape[1] - 1, device=input_ids.device, dtype=torch.long)
            else:
                is_not_pad = input_ids.ne(pad_token_id)
                
                # Find first non-pad token (handles left-padding)
                start_indices = torch.argmax(is_not_pad.int(), dim=1)
                
                # Find last non-pad token
                end_indices = input_ids.shape[1] - 1 - torch.argmax(torch.flip(is_not_pad, dims=[1]).int(), dim=1)

            # Interpret min_pos_to_select_from as a relative offset from the start of content.
            lower_bounds = start_indices + min_pos_to_select_from
            
            # Ensure the lower bound does not exceed the upper bound.
            # If it does (sequence too short), this will select the last token.
            effective_lower_bounds = torch.minimum(lower_bounds, end_indices)
            upper_bounds = end_indices
            
            if position_selection_strategy == 'midpoint':
                calculated_positions = (effective_lower_bounds + upper_bounds) // 2
            elif position_selection_strategy == 'random':
                # Generate random floats in [0, 1) and scale them to the valid range
                rand_floats = torch.rand(batch_size, device=input_ids.device)
                range_size = (upper_bounds - effective_lower_bounds + 1).clamp(min=1)
                calculated_positions = effective_lower_bounds + (rand_floats * range_size).long()
            else:
                raise ValueError(f"Unknown position_selection_strategy: '{position_selection_strategy}'")

            # Final safety clamp, although the logic above should prevent out-of-bounds.
            seq_len_minus_1 = max(input_ids.shape[1] - 1, 0)
            calculated_positions = torch.clamp(calculated_positions, min=0, max=seq_len_minus_1)
        else:
            raise ValueError("Either `token_positions` or `min_pos_to_select_from` must be provided.")

        # --- Hook-based activation extraction ---
        captured_activations = None

        def _capture_hook(_, __, output): # noqa: ANN001
            nonlocal captured_activations
            # Output of a transformer block is usually a tuple (hidden_state, present_key_value, ...)
            # or just hidden_state. We are interested in the first element.
            if isinstance(output, tuple):
                captured_activations = output[0].detach()
            else:
                captured_activations = output.detach()
            return output # Pass through the output

        try:
            target_block = self.model.get_submodule(f"model.layers.{layer_idx}")
        except AttributeError:
            try:
                target_block = self.model.transformer.h[layer_idx] # type: ignore[attr-defined]
            except AttributeError:
                target_block = self.model.get_submodule(f"transformer.h.{layer_idx}")
        
        handle = target_block.register_forward_hook(_capture_hook)
        
        grad_ctx = torch.no_grad() if no_grad else nullcontext()
        with grad_ctx:
            _ = self.model( # We don't need the model's final output here
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False, # Not needed as we use a hook
            )
        
        handle.remove()

        if captured_activations is None:
            raise RuntimeError(
                f"Hook did not capture activations from layer {layer_idx}. "
                "This might indicate an issue with the model structure or hook registration."
            )
        
        # captured_activations is (B, seq_len, dim)
        batch_indices = torch.arange(batch_size, device=captured_activations.device)
        selected_activations = captured_activations[batch_indices, calculated_positions] # (B, dim)
        # --- End hook-based activation extraction ---

        if was_1d_input:
            selected_activations = selected_activations.squeeze(0)
            # calculated_positions = calculated_positions.squeeze(0) # Already (1,) or scalar

        return selected_activations, calculated_positions

    def get_all_activations_at_layer(
        self,
        input_ids: torch.Tensor, # Expected shape (seq_len,) or (1, seq_len)
        layer_idx: int,
        *,
        attention_mask: Optional[torch.Tensor] = None, # Expected shape (seq_len,) or (1, seq_len)
        no_grad: bool = True,
    ) -> torch.Tensor:
        """
        Extracts all hidden state activations from a specified layer for a single sequence
        using a forward hook.

        Args:
            input_ids: Input token ids, shape (seq_len,) or (1, seq_len).
            layer_idx: 0-indexed layer to extract activations from.
            attention_mask: Optional attention mask, shape (seq_len,) or (1, seq_len).
                            If None and pad_token_id is configured, it's created.
            no_grad: Whether to run in no_grad context.

        Returns:
            torch.Tensor: Hidden states for the layer, shape (seq_len, hidden_dim).
        """
        self._validate_layer_idx(layer_idx)

        if not isinstance(input_ids, torch.Tensor):
             raise TypeError(f"input_ids must be a torch.Tensor, got {type(input_ids)}")

        _processed_input_ids = input_ids
        _processed_attention_mask = attention_mask

        if input_ids.ndim == 1: # Shape (seq_len,)
            _processed_input_ids = input_ids.unsqueeze(0) # Convert to (1, seq_len)
            if attention_mask is not None and attention_mask.ndim == 1:
                if attention_mask.shape[0] != input_ids.shape[0]:
                    raise ValueError(
                        f"1D attention_mask length {attention_mask.shape[0]} "
                        f"does not match 1D input_ids length {input_ids.shape[0]}"
                    )
                _processed_attention_mask = attention_mask.unsqueeze(0)
        elif input_ids.ndim == 2: # Shape (possibly 1, seq_len)
            if input_ids.shape[0] != 1:
                raise ValueError(
                    f"For 2D input_ids, batch size must be 1. Got shape {input_ids.shape}"
                )
            if attention_mask is not None and attention_mask.ndim == 1:
                 raise ValueError(
                    f"Cannot use 1D attention_mask with 2D input_ids. Mask shape: {attention_mask.shape}, "
                    f"Input shape: {input_ids.shape}"
                 )
        else: # ndim is 0 or > 2
            raise ValueError(
                f"input_ids must be 1D (seq_len,) or 2D (1, seq_len). "
                f"Got {input_ids.ndim}D tensor with shape {input_ids.shape}"
            )
        
        # At this point, _processed_input_ids is (1, seq_len)

        if _processed_attention_mask is None and self.model.config.pad_token_id is not None:
            _processed_attention_mask = _processed_input_ids.ne(self.model.config.pad_token_id).long()
        elif _processed_attention_mask is not None:
            if not isinstance(_processed_attention_mask, torch.Tensor):
                 raise TypeError(f"attention_mask must be a torch.Tensor or None, got {type(_processed_attention_mask)}")
            
            is_valid_mask_shape = (
                _processed_attention_mask.ndim == 2 and
                _processed_attention_mask.shape[0] == 1 and
                _processed_attention_mask.shape[1] == _processed_input_ids.shape[1]
            )
            if not is_valid_mask_shape:
                raise ValueError(
                    f"Final attention_mask shape {_processed_attention_mask.shape} is incompatible with "
                    f"processed input_ids shape {_processed_input_ids.shape} (expected (1, {_processed_input_ids.shape[1]}))."
                )

        captured_layer_activations = None

        def _capture_all_hook(_, __, output):  # noqa: ANN001
            nonlocal captured_layer_activations
            if isinstance(output, tuple):
                captured_layer_activations = output[0].detach()
            else:
                captured_layer_activations = output.detach()
            return output 

        try:
            target_block = self.model.get_submodule(f"model.layers.{layer_idx}")
        except AttributeError:
            try:
                target_block = self.model.transformer.h[layer_idx]  # type: ignore[attr-defined]
            except AttributeError:
                target_block = self.model.get_submodule(f"transformer.h.{layer_idx}")
        
        handle = target_block.register_forward_hook(_capture_all_hook)
        
        grad_ctx = torch.no_grad() if no_grad else nullcontext()
        with grad_ctx:
            _ = self.model( 
                input_ids=_processed_input_ids,
                attention_mask=_processed_attention_mask,
                output_hidden_states=False, 
            )
        
        handle.remove()

        if captured_layer_activations is None:
            raise RuntimeError(
                f"Hook did not capture activations from layer {layer_idx}. "
                "This might indicate an issue with the model structure or hook registration."
            )
        
        return captured_layer_activations.squeeze(0)

    # ------------------------------------------------------------------
    # Convenience ----------------------------------------------------------------
    # ------------------------------------------------------------------

    def _monkeypatch_reshape_inputs(self) -> None:  # noqa: D401
        """HF tiny GPT-2 lacks ``reshape_inputs`` – tests expect it."""

        if not hasattr(self.model, "reshape_inputs"):
            # Assign the static method instead of a lambda
            self.model.reshape_inputs = OrigWrapper._static_reshape_inputs  # type: ignore[attr-defined]

    # Called in __init__
