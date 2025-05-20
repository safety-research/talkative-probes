"""Frozen original model with single-layer activation swap hook."""

import torch
from transformers import AutoModelForCausalLM

__all__ = ["OrigWrapper"]


class OrigWrapper:
    """Thin wrapper around HF CausalLM that supports `forward_with_replacement`."""

    def __init__(self, model_name: str, load_in_8bit: bool = True) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=load_in_8bit)
        self.model.eval()
        self._monkeypatch_reshape_inputs()

    @torch.no_grad()
    def forward_with_replacement(
        self,
        input_ids: torch.Tensor,
        new_activation: torch.Tensor,
        layer_idx: int,
        token_pos: int,
    ) -> "transformers.modeling_outputs.CausalLMOutput":
        """Forward pass where hidden_state[layer_idx][:, token_pos] is replaced."""

        def _swap_hook(_, __, output):  # noqa: ANN001
            hidden = output[0]
            # Allow new_activation with shape (d_model,) or (B, d_model)
            if new_activation.dim() == 1:
                hidden[:, token_pos] = new_activation.unsqueeze(0)
            else:
                hidden[:, token_pos] = new_activation  # type: ignore[index]
            return (hidden,) + output[1:]

        # The exact module path varies across architectures.  We try the common
        # ``transformer.h`` fallback and, if missing, build the sub-module name
        # explicitly.
        try:
            target_block = self.model.transformer.h[layer_idx]  # type: ignore[attr-defined]
        except AttributeError:
            target_block = self.model.get_submodule(f"transformer.h.{layer_idx}")

        handle = target_block.register_forward_hook(_swap_hook)
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
