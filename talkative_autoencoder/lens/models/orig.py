"""Frozen original model with single-layer activation swap hook."""

from contextlib import nullcontext
from typing import Optional, Tuple, Union

import torch

from transformers import AutoModelForCausalLM, Gemma3ForCausalLM

__all__ = ["OrigWrapper"]


class OrigWrapper:
    """Thin wrapper around HF CausalLM that supports `forward_with_replacement`."""

    # Define the replacement function as a static method
    @staticmethod
    def _static_reshape_inputs(x, **__) -> dict:
        return {"input_ids": x}

    def __init__(
        self, model_name: str, torch_dtype=None, load_in_8bit: bool = False, base_to_use=None, lora_name: str = None
    ) -> None:
        if base_to_use is None:
            # Load model with specified dtype
            if "gemma-3" in model_name:
                print(f"Loading Gemma3ModelForCausalLM for model '{model_name}'")
                self.model = Gemma3ForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    load_in_8bit=load_in_8bit,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    load_in_8bit=load_in_8bit,
                )
            self.name = model_name
            self.lora_name = None
            if hasattr(self.model, "peft_config"):
                self.lora_name = model_name
        else:
            self.model = base_to_use
            self.name = base_to_use.config._name_or_path
            if hasattr(base_to_use, "peft_config") and lora_name is None:
                self.lora_name = "????"
                self.name += "_" + "LORA" + "_" + self.lora_name
            elif hasattr(base_to_use, "peft_config") and lora_name is not None:
                self.lora_name = lora_name
                self.name += "_" + self.lora_name
            else:
                self.lora_name = None

        self.model.eval()

        self.name = model_name  # self.model.config._name_or_path if  not hasattr(self.model, 'active_adapter') else  self.model.config._name_or_path + "_LORA"
        # if hasattr(self.model, 'active_adapter'):
        #     self.lora_name = self.model.active_adapter
        # else:
        #     self.lora_name = None

        # if lora_name is not None:
        #     self.name += "_" + lora_name
        # elif
        # Ensure base model parameters do not require gradients
        for param in self.model.parameters():
            param.requires_grad = False
        self._monkeypatch_reshape_inputs()

        # Lazy hook management
        self._registered_hooks = {}  # layer_idx -> hook handle
        self._hook_enabled = {}  # layer_idx -> bool
        self._hook_data = {}  # layer_idx -> data dict
        self.num_hidden_layers = (
            self.model.config.num_hidden_layers
            if not hasattr(self.model.config, "text_config")
            else self.model.config.text_config.num_hidden_layers
        )

        # # Pre-register hooks for all layers
        # for layer_idx in range(self.model.config.num_hidden_layers):
        #     self._hook_enabled[layer_idx] = False
        #     self._hook_data[layer_idx] = None

        #     try:
        #         target_block = self.model.get_submodule(f"model.layers.{layer_idx}")
        #     except AttributeError:
        #         # Handle GPT-2 style models
        #         try:
        #             target_block = self.model.transformer.h[layer_idx]
        #         except AttributeError:
        #             target_block = self.model.get_submodule(f"transformer.h.{layer_idx}")

        #     # Create a layer-specific hook
        #     def make_hook(idx):
        #         def hook_fn(module, input, output):
        #             if self._hook_enabled[idx]:
        #                 return self._apply_swap(output, idx)
        #             return output
        #         return hook_fn

        #     self._registered_hooks[layer_idx] = target_block.register_forward_hook(make_hook(layer_idx))

    def _validate_layer_idx(self, layer_idx: int) -> None:
        """Checks if the provided layer_idx is valid for the model."""
        try:
            num_hidden_layers = self.num_hidden_layers
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

    def _apply_swap(self, output, layer_idx):
        """Apply the activation swap if enabled for this layer."""
        data = self._hook_data[layer_idx]
        if data is None:
            return output

        hidden = output[0]
        new_activation = data["new_activation"]
        token_pos = data["token_pos"]

        # Ensure dtype consistency
        if new_activation.dtype != hidden.dtype:
            new_activation = new_activation.to(hidden.dtype)

        if new_activation.dim() == 1:
            hidden[:, token_pos] = new_activation.unsqueeze(0)
        else:
            hidden[:, token_pos] = new_activation

        return (hidden,) + output[1:]

    def forward_with_replacement(
        self,
        input_ids: torch.Tensor,
        new_activation: torch.Tensor,
        layer_idx: int,
        token_pos: int,
        *,
        attention_mask: torch.Tensor | None = None,
        no_grad: bool = True,
    ):
        """Forward pass with activation replacement using lazy-loaded persistent hooks."""
        self._validate_layer_idx(layer_idx)

        # Ensure hook exists for this layer (lazy registration)
        self._ensure_hook_registered(layer_idx)

        # Set hook data
        self._hook_data[layer_idx] = {"new_activation": new_activation, "token_pos": token_pos}
        self._hook_enabled[layer_idx] = True

        grad_ctx = torch.no_grad() if no_grad else nullcontext()
        with grad_ctx:
            try:
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            finally:
                # Disable the hook after use (but keep it registered)
                self._hook_enabled[layer_idx] = False
                self._hook_data[layer_idx] = None

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
    ):
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

        self._validate_layer_idx(layer_idx)  # Validate layer index at the beginning

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
        position_selection_strategy: str = "random",
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
                calculated_positions = torch.tensor(
                    [token_positions] * batch_size, device=input_ids.device, dtype=torch.long
                )
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
            if attention_mask is None:
                # If no attention mask at all, assume the entire sequence is valid.
                start_indices = torch.zeros(batch_size, device=input_ids.device, dtype=torch.long)
                end_indices = torch.full(
                    (batch_size,), input_ids.shape[1] - 1, device=input_ids.device, dtype=torch.long
                )
            else:
                # Use the attention_mask to find the valid range of tokens.
                # `argmax` finds the first '1'.
                start_indices = torch.argmax(attention_mask.int(), dim=1)
                # Flip the mask to find the last '1' from the end.
                end_indices = input_ids.shape[1] - 1 - torch.argmax(torch.flip(attention_mask, dims=[1]).int(), dim=1)

                # --- Edge Case Fix ---
                # For sequences that are all padding, attention_mask is all zeros. `argmax` will return 0 for both start and end,
                # which incorrectly implies the whole sequence is valid. We correct this by setting start > end for such cases.
                has_no_valid_tokens = ~torch.any(attention_mask, dim=1)
                start_indices[has_no_valid_tokens] = 1
                end_indices[has_no_valid_tokens] = 0
                if has_no_valid_tokens.any():
                    print(f"Warning: {has_no_valid_tokens.sum()} sequences have no valid tokens.")

            # Interpret min_pos_to_select_from as a relative offset from the start of content.
            lower_bounds = start_indices + min_pos_to_select_from

            # Ensure the lower bound does not exceed the upper bound.
            # If it does (sequence too short), this will select the last token.
            effective_lower_bounds = torch.minimum(lower_bounds, end_indices)
            upper_bounds = end_indices

            if position_selection_strategy == "midpoint":
                calculated_positions = (effective_lower_bounds + upper_bounds) // 2
            elif position_selection_strategy == "random":
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

        def _capture_hook(_, __, output):  # noqa: ANN001
            nonlocal captured_activations
            # Output of a transformer block is usually a tuple (hidden_state, present_key_value, ...)
            # or just hidden_state. We are interested in the first element.
            if isinstance(output, tuple):
                captured_activations = output[0].detach()
            else:
                captured_activations = output.detach()
            return output  # Pass through the output

        try:
            target_block = self.model.get_submodule(f"model.layers.{layer_idx}")
        except AttributeError:
            try:
                target_block = self.model.transformer.h[layer_idx]  # type: ignore[attr-defined]
            except AttributeError:
                target_block = self.model.get_submodule(f"transformer.h.{layer_idx}")

        handle = target_block.register_forward_hook(_capture_hook)

        grad_ctx = torch.no_grad() if no_grad else nullcontext()
        with grad_ctx:
            _ = self.model(  # We don't need the model's final output here
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,  # Not needed as we use a hook
            )

        handle.remove()

        if captured_activations is None:
            raise RuntimeError(
                f"Hook did not capture activations from layer {layer_idx}. "
                "This might indicate an issue with the model structure or hook registration."
            )

        # captured_activations is (B, seq_len, dim)
        batch_indices = torch.arange(batch_size, device=captured_activations.device)
        selected_activations = captured_activations[batch_indices, calculated_positions]  # (B, dim)
        # --- End hook-based activation extraction ---

        if was_1d_input:
            selected_activations = selected_activations.squeeze(0)
            # calculated_positions = calculated_positions.squeeze(0) # Already (1,) or scalar

        return selected_activations, calculated_positions

    def get_activations_at_multiple_positions(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
        num_positions: int = 1,
        min_pos_to_select_from: Optional[int] = None,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        no_grad: bool = True,
        position_selection_strategy: str = "random",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts multiple hidden state activations per sequence from a specified layer in a single forward pass.

        Args:
            input_ids: Input token ids, shape (B, seq_len).
            layer_idx: 0-indexed layer to extract activations from.
            num_positions: Number of positions to extract per sequence.
            min_pos_to_select_from: Minimum position to consider for extraction.
            attention_mask: Optional attention mask, shape (B, seq_len).
            no_grad: Whether to run in no_grad context.
            position_selection_strategy: 'random' or 'midpoint'. For multiple positions,
                                        'random' selects different random positions,
                                        'midpoint' is treated as 'random' with a warning.

        Returns:
            Tuple of:
            - selected_activations: Tensor of shape (B, num_positions, hidden_dim)
            - calculated_positions: Tensor of shape (B, num_positions) with the positions used
        """
        self._validate_layer_idx(layer_idx)

        if min_pos_to_select_from is None:
            min_pos_to_select_from = 0

        if position_selection_strategy == "midpoint" and num_positions > 1:
            print(
                f"Warning: 'midpoint' strategy with num_positions={num_positions} > 1 will use random selection instead"
            )
            position_selection_strategy = "random"

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # Create attention mask if needed
        if attention_mask is None and self.model.config.pad_token_id is not None:
            attention_mask = input_ids.ne(self.model.config.pad_token_id).long()

        # Calculate valid range for each sequence
        if attention_mask is None:
            start_indices = torch.zeros(batch_size, device=input_ids.device, dtype=torch.long)
            end_indices = torch.full((batch_size,), seq_len - 1, device=input_ids.device, dtype=torch.long)
        else:
            start_indices = torch.argmax(attention_mask.int(), dim=1)
            end_indices = seq_len - 1 - torch.argmax(torch.flip(attention_mask, dims=[1]).int(), dim=1)

            # Handle sequences with no valid tokens
            has_no_valid_tokens = ~torch.any(attention_mask, dim=1)
            start_indices[has_no_valid_tokens] = 1
            end_indices[has_no_valid_tokens] = 0

        # Calculate effective bounds
        lower_bounds = torch.maximum(start_indices + min_pos_to_select_from, start_indices)
        upper_bounds = end_indices

        # Generate positions
        calculated_positions = torch.zeros((batch_size, num_positions), device=input_ids.device, dtype=torch.long)

        if position_selection_strategy == "random":
            for b in range(batch_size):
                lb = lower_bounds[b].item()
                ub = upper_bounds[b].item()

                if ub >= lb:
                    # Valid range exists
                    range_size = ub - lb + 1
                    if num_positions <= range_size:
                        # Sample without replacement if possible
                        sampled_positions = torch.randperm(range_size, device=input_ids.device)[:num_positions] + lb
                    else:
                        # Sample with replacement if we need more positions than available
                        sampled_positions = torch.randint(lb, ub + 1, (num_positions,), device=input_ids.device)
                    calculated_positions[b] = sampled_positions
                else:
                    # No valid range, use the last valid position
                    calculated_positions[b] = torch.clamp(end_indices[b], min=0, max=seq_len - 1)
        else:  # midpoint strategy (only for num_positions=1)
            midpoints = (lower_bounds + upper_bounds) // 2
            calculated_positions[:, 0] = midpoints

        # Hook-based activation extraction
        captured_activations = None

        def _capture_hook(_, __, output):
            with torch._dynamo.disable():
                nonlocal captured_activations
                if isinstance(output, tuple):
                    captured_activations = output[0].detach()
                else:
                    captured_activations = output.detach()
            return output

        # Get target block
        try:
            target_block = self.model.get_submodule(f"model.layers.{layer_idx}")
        except AttributeError:
            try:
                target_block = self.model.transformer.h[layer_idx]
            except AttributeError:
                target_block = self.model.get_submodule(f"transformer.h.{layer_idx}")

        handle = target_block.register_forward_hook(_capture_hook)

        grad_ctx = torch.no_grad() if no_grad else nullcontext()
        with grad_ctx:
            _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )

        handle.remove()

        if captured_activations is None:
            raise RuntimeError(
                f"Hook did not capture activations from layer {layer_idx}. "
                "This might indicate an issue with the model structure or hook registration."
            )

        # Extract activations at multiple positions
        # captured_activations is (B, seq_len, hidden_dim)
        batch_indices = (
            torch.arange(batch_size, device=captured_activations.device).unsqueeze(1).expand(-1, num_positions)
        )
        selected_activations = captured_activations[
            batch_indices, calculated_positions
        ]  # (B, num_positions, hidden_dim)

        return selected_activations, calculated_positions

    def get_all_activations_at_layer(
        self,
        input_ids: torch.Tensor,  # Expected shape (seq_len,) or (B, seq_len)
        layer_idx: int,
        *,
        attention_mask: Optional[torch.Tensor] = None,  # Expected shape (seq_len,) or (B, seq_len)
        no_grad: bool = True,
    ) -> torch.Tensor:
        """
        Extracts all hidden state activations from a specified layer for one or more sequences
        using a forward hook.

        Args:
            input_ids: Input token ids, shape (seq_len,) or (B, seq_len).
            layer_idx: 0-indexed layer to extract activations from.
            attention_mask: Optional attention mask, shape (seq_len,) or (B, seq_len).
                            If None and pad_token_id is configured, it's created.
            no_grad: Whether to run in no_grad context.

        Returns:
            torch.Tensor: Hidden states for the layer, shape (seq_len, hidden_dim) or (B, seq_len, hidden_dim).
        """
        self._validate_layer_idx(layer_idx)

        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"input_ids must be a torch.Tensor, got {type(input_ids)}")

        _processed_input_ids = input_ids
        _processed_attention_mask = attention_mask
        was_1d_input = False

        if input_ids.ndim == 1:  # Shape (seq_len,)
            _processed_input_ids = input_ids.unsqueeze(0)  # Convert to (1, seq_len)
            was_1d_input = True
            if attention_mask is not None and attention_mask.ndim == 1:
                if attention_mask.shape[0] != input_ids.shape[0]:
                    raise ValueError(
                        f"1D attention_mask length {attention_mask.shape[0]} "
                        f"does not match 1D input_ids length {input_ids.shape[0]}"
                    )
                _processed_attention_mask = attention_mask.unsqueeze(0)
        elif input_ids.ndim == 2:  # Shape (B, seq_len)
            if attention_mask is not None and attention_mask.ndim == 1:
                raise ValueError(
                    f"Cannot use 1D attention_mask with 2D input_ids. Mask shape: {attention_mask.shape}, "
                    f"Input shape: {input_ids.shape}"
                )
        else:  # ndim is 0 or > 2
            raise ValueError(
                f"input_ids must be 1D (seq_len,) or 2D (B, seq_len). "
                f"Got {input_ids.ndim}D tensor with shape {input_ids.shape}"
            )

        # At this point, _processed_input_ids is (B, seq_len)

        if _processed_attention_mask is None and self.model.config.pad_token_id is not None:
            _processed_attention_mask = _processed_input_ids.ne(self.model.config.pad_token_id).long()
        elif _processed_attention_mask is not None:
            if not isinstance(_processed_attention_mask, torch.Tensor):
                raise TypeError(f"attention_mask must be a torch.Tensor or None, got {type(_processed_attention_mask)}")

            is_valid_mask_shape = (
                _processed_attention_mask.ndim == 2
                and _processed_attention_mask.shape[0] == _processed_input_ids.shape[0]
                and _processed_attention_mask.shape[1] == _processed_input_ids.shape[1]
            )
            if not is_valid_mask_shape:
                raise ValueError(
                    f"Final attention_mask shape {_processed_attention_mask.shape} is incompatible with "
                    f"processed input_ids shape {_processed_input_ids.shape}."
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

        if was_1d_input:
            return captured_layer_activations.squeeze(0)
        else:
            return captured_layer_activations

    def get_all_layers_activations_at_position(
        self,
        input_ids: torch.Tensor,  # Expected shape (seq_len,) or (1, seq_len)
        position_to_analyze: int,
        *,
        attention_mask: Optional[torch.Tensor] = None,  # Expected shape (seq_len,) or (1, seq_len)
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
        # self._validate_layer_idx(layer_idx)

        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"input_ids must be a torch.Tensor, got {type(input_ids)}")

        _processed_input_ids = input_ids
        _processed_attention_mask = attention_mask

        if input_ids.ndim == 1:  # Shape (seq_len,)
            _processed_input_ids = input_ids.unsqueeze(0)  # Convert to (1, seq_len)
            if attention_mask is not None and attention_mask.ndim == 1:
                if attention_mask.shape[0] != input_ids.shape[0]:
                    raise ValueError(
                        f"1D attention_mask length {attention_mask.shape[0]} "
                        f"does not match 1D input_ids length {input_ids.shape[0]}"
                    )
                _processed_attention_mask = attention_mask.unsqueeze(0)
        elif input_ids.ndim == 2:  # Shape (possibly 1, seq_len)
            if input_ids.shape[0] != 1:
                raise ValueError(f"For 2D input_ids, batch size must be 1. Got shape {input_ids.shape}")
            if attention_mask is not None and attention_mask.ndim == 1:
                raise ValueError(
                    f"Cannot use 1D attention_mask with 2D input_ids. Mask shape: {attention_mask.shape}, "
                    f"Input shape: {input_ids.shape}"
                )
        else:  # ndim is 0 or > 2
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
                _processed_attention_mask.ndim == 2
                and _processed_attention_mask.shape[0] == 1
                and _processed_attention_mask.shape[1] == _processed_input_ids.shape[1]
            )
            if not is_valid_mask_shape:
                raise ValueError(
                    f"Final attention_mask shape {_processed_attention_mask.shape} is incompatible with "
                    f"processed input_ids shape {_processed_input_ids.shape} (expected (1, {_processed_input_ids.shape[1]}))."
                )

        # Capture the full stack of activations at a particular position across all layers
        activations_by_layer = []

        def make_hook():
            def _hook(_, __, output):
                if isinstance(output, tuple):
                    activations_by_layer.append(output[0].detach())
                else:
                    activations_by_layer.append(output.detach())
                return output

            return _hook

        handles = []
        num_layers = None
        # Try to infer number of layers from model
        try:
            num_layers = len(self.model.get_submodule("model.layers"))
            layer_module_getter = lambda idx: self.model.get_submodule(f"model.layers.{idx}")
        except Exception:
            try:
                num_layers = len(self.model.transformer.h)
                layer_module_getter = lambda idx: self.model.transformer.h[idx]
            except Exception:
                num_layers = len(self.model.get_submodule("transformer.h"))
                layer_module_getter = lambda idx: self.model.get_submodule(f"transformer.h.{idx}")

        for i in range(num_layers):
            block = layer_module_getter(i)
            handles.append(block.register_forward_hook(make_hook()))

        grad_ctx = torch.no_grad() if no_grad else nullcontext()
        with grad_ctx:
            _ = self.model(
                input_ids=_processed_input_ids,
                attention_mask=_processed_attention_mask,
                output_hidden_states=False,
            )

        for h in handles:
            h.remove()

        if not activations_by_layer or len(activations_by_layer) != num_layers:
            raise RuntimeError(
                f"Did not capture activations for all layers ({len(activations_by_layer)}/{num_layers})."
            )

        # activations_by_layer: list of tensors, each (1, seq_len, hidden_dim)
        # Extract the activations at the specified position
        # position_to_analyze is expected to be in scope as an argument
        position = position_to_analyze
        seq_len = activations_by_layer[0].shape[1]
        if not (0 <= position < seq_len):
            raise ValueError(f"position_to_analyze {position} is out of bounds for sequence length {seq_len}")

        # Stack to (num_layers, hidden_dim)
        stacked = torch.stack([a.squeeze(0)[position] for a in activations_by_layer], dim=0)  # (num_layers, hidden_dim)
        return stacked

    # ------------------------------------------------------------------
    # Convenience ----------------------------------------------------------------
    # ------------------------------------------------------------------

    def _monkeypatch_reshape_inputs(self) -> None:  # noqa: D401
        """HF tiny GPT-2 lacks ``reshape_inputs`` – tests expect it."""

        if not hasattr(self.model, "reshape_inputs"):
            # Assign the static method instead of a lambda
            self.model.reshape_inputs = OrigWrapper._static_reshape_inputs  # type: ignore[attr-defined]

    def _ensure_hook_registered(self, layer_idx: int):
        """Thread-safe lazy hook registration."""
        if layer_idx in self._registered_hooks:
            return

        # Double-check pattern
        if layer_idx in self._registered_hooks:
            return

        # Find the target layer
        try:
            target_block = self.model.get_submodule(f"model.layers.{layer_idx}")
        except AttributeError:
            try:
                target_block = self.model.transformer.h[layer_idx]
            except AttributeError:
                target_block = self.model.get_submodule(f"transformer.h.{layer_idx}")

        # Create a layer-specific hook
        def hook_fn(module, input, output):
            if self._hook_enabled.get(layer_idx, False):
                return self._apply_swap(output, layer_idx)
            return output

        # Register and store the handle
        self._registered_hooks[layer_idx] = target_block.register_forward_hook(hook_fn)
        self._hook_enabled[layer_idx] = False
        self._hook_data[layer_idx] = None

    # Called in __init__
