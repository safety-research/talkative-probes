"""Tiny wrapper around Weights & Biases to avoid littering training code."""

from __future__ import annotations

from contextlib import suppress
from typing import Any, Dict

__all__ = ["init", "log"]


_wandb_run = None


def init(project: str = "consistency-lens", **kwargs: Any) -> str | None:  # noqa: D401
    """Initialise a W&B run if the module is available.

    If `wandb` is not installed, this becomes a no-op so that training still
    runs in minimal environments.
    
    Returns:
        The wandb run ID if wandb is available, None otherwise.
    """

    global _wandb_run

    with suppress(ImportError):
        import wandb  # type: ignore

        _wandb_run = wandb.init(project=project, **kwargs)
        return _wandb_run.id if _wandb_run else None
    
    return None


def log(metrics: Dict[str, float], step: int | None = None) -> None:  # noqa: D401
    """Log *metrics* to W&B if active, else ignore."""

    if _wandb_run is None:
        return

    _wandb_run.log(metrics, step=step)
