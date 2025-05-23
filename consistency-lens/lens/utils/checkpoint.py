"""Very small helper around ``torch.save``/``torch.load`` for single-file checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import torch

__all__ = ["save", "load"]


def save(path: str | Path, models: Dict[str, torch.nn.Module], optim: torch.optim.Optimizer | None, step: int, scheduler: Any | None = None, **extra) -> None:  # noqa: D401
    """Serialize *models* + *optim* state_dicts and training metadata to *path*."""

    ckpt = {"step": step, "models": {k: m.state_dict() for k, m in models.items()}}
    if optim is not None:
        ckpt["optim"] = optim.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    ckpt.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load(path: str | Path, models: Dict[str, torch.nn.Module], optim: torch.optim.Optimizer | None = None, map_location: str | torch.device = "cpu") -> Dict[str, Any]:  # noqa: D401
    """Load checkpoint *path* and restore weights into *models* / *optim*.

    Returns the checkpoint dictionary so caller can read ``step`` or any extra
    metadata stored during ``save``.
    """

    ckpt = torch.load(path, map_location=map_location)
    for k, m in models.items():
        if k in ckpt["models"]:
            state_dict = ckpt["models"][k]
            
            # Handle loading from non-compiled to compiled models
            if hasattr(m, '_orig_mod'):
                # Model is compiled (wrapped in OptimizedModule)
                # Check if state dict keys need _orig_mod prefix
                sample_key = next(iter(state_dict.keys()))
                if not sample_key.startswith('_orig_mod.'):
                    # Add _orig_mod prefix to all keys
                    state_dict = {f'_orig_mod.{k}': v for k, v in state_dict.items()}
            else:
                # Model is not compiled
                # Check if state dict has _orig_mod prefix that needs removal
                sample_key = next(iter(state_dict.keys())) if state_dict else ""
                if sample_key.startswith('_orig_mod.'):
                    # Remove _orig_mod prefix from all keys
                    state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
            
            m.load_state_dict(state_dict)
    if optim is not None and "optim" in ckpt:
        try:
            optim.load_state_dict(ckpt["optim"])
        except ValueError:
            # Param group mismatch (e.g., eval build uses flat param list). Skip.
            pass
    return ckpt
