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
    
    # Save random states for reproducibility
    import numpy as np
    ckpt["rng_states"] = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
    }
    
    ckpt.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load(path: str | Path, models: Dict[str, torch.nn.Module], optim: torch.optim.Optimizer | None = None, map_location: str | torch.device = "cpu", load_rng_state: bool = True, strict_load: bool = True) -> Dict[str, Any]:  # noqa: D401
    """Load checkpoint *path* and restore weights into *models* / *optim*.

    Returns the checkpoint dictionary so caller can read ``step`` or any extra
    metadata stored during ``save``.
    """

    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    
    # Restore random states if available and requested
    if load_rng_state and "rng_states" in ckpt:
        import numpy as np
        rng_states = ckpt["rng_states"]
        if "torch" in rng_states and rng_states["torch"] is not None:
            # Ensure the state is on CPU and is a ByteTensor
            state = rng_states["torch"]
            if isinstance(state, torch.Tensor):
                state = state.cpu()
                if state.dtype != torch.uint8:
                    state = state.to(torch.uint8)
            torch.set_rng_state(state)
        if "cuda" in rng_states and rng_states["cuda"] is not None and torch.cuda.is_available():
            # CUDA state should already be correct format
            state = rng_states["cuda"]
            if isinstance(state, torch.Tensor):
                state = state.cpu()
            torch.cuda.set_rng_state(state)
        if "numpy" in rng_states:
            np.random.set_state(rng_states["numpy"])
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
            
            # Debug: Check if prompt embeddings are in state dict
            if k == 'decoder':
                import logging
                log = logging.getLogger(__name__)
                if 'prompt_left_emb' in state_dict:
                    log.info(f"Loading decoder prompt_left_emb with norm {state_dict['prompt_left_emb'].norm().item():.4f}")
                if 'prompt_right_emb' in state_dict:
                    log.info(f"Loading decoder prompt_right_emb with norm {state_dict['prompt_right_emb'].norm().item():.4f}")
            
            try:
                if strict_load:
                    m.load_state_dict(state_dict, strict=strict_load)
                else:
                    missing, unexpected = m.load_state_dict(state_dict, strict=False)
                    if missing:
                        log.warning(f"Missing keys: {missing}")
                    if unexpected:
                        log.warning(f"Unexpected keys: {unexpected}")
            except RuntimeError as e:
                log = logging.getLogger(__name__)
                log.error(f"Failed to load state dict for {k}: {e}")
                # Try with strict=False to see what's missing/unexpected
                missing, unexpected = m.load_state_dict(state_dict, strict=False)
                if missing:
                    log.error(f"Missing keys: {missing}")
                if unexpected:
                    log.error(f"Unexpected keys: {unexpected}")
                raise
    if optim is not None and "optim" in ckpt:
        try:
            optim.load_state_dict(ckpt["optim"])
            # Ensure initial_lr is set for each param group (required by LR schedulers)
            for group in optim.param_groups:
                if 'initial_lr' not in group:
                    group['initial_lr'] = group['lr']
        except ValueError:
            # Param group mismatch (e.g., eval build uses flat param list). Skip.
            pass
    return ckpt
