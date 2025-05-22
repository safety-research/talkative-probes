"""Checkpoint management utilities for training."""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

import torch
from lens.utils import checkpoint


class CheckpointManager:
    """Manages checkpoint saving, loading, and cleanup during training."""

    def __init__(self, config: dict, logger: logging.Logger):
        """Initialize checkpoint manager from config."""
        self.config = config.get('checkpoint', {})
        self.logger = logger
        self.enabled = self.config.get('enabled', True)
        
        if not self.enabled:
            return
            
        self.output_dir = Path(self.config.get('output_dir', 'outputs/checkpoints'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save strategies
        self.save_every_n_steps = self.config.get('save_every_n_steps', 0)
        self.save_every_n_epochs = self.config.get('save_every_n_epochs', 0)
        self.save_at_end = self.config.get('save_at_end', True)
        
        # Best checkpoint tracking
        self.track_best_n = self.config.get('track_best_n', 0)
        self.best_metric = self.config.get('best_metric', 'val_loss')
        self.best_mode = self.config.get('best_mode', 'min')
        self.best_checkpoints: List[Tuple[float, Path]] = []  # (metric_value, path)
        
        # Checkpoint management
        self.max_checkpoints = self.config.get('max_checkpoints', -1)
        self.delete_old = self.config.get('delete_old_checkpoints', True)
        self.all_checkpoints: List[Path] = []  # Track all saved checkpoints
        
        # Components to save
        self.save_components = self.config.get('save_components', {
            'models': True,
            'optimizer': True,
            'scheduler': False,
            'config': True,
            'metrics': True
        })
        
        self.name_pattern = self.config.get('name_pattern', 'checkpoint_step{step}_epoch{epoch}')

    def should_save_step(self, step: int) -> bool:
        """Check if checkpoint should be saved at this step."""
        if not self.enabled:
            return False
        return self.save_every_n_steps > 0 and step > 0 and step % self.save_every_n_steps == 0

    def should_save_epoch(self, epoch: int, epoch_just_finished: bool) -> bool:
        """Check if checkpoint should be saved at this epoch."""
        if not self.enabled:
            return False
        return (self.save_every_n_epochs > 0 and 
                epoch_just_finished and 
                epoch % self.save_every_n_epochs == 0)

    def format_checkpoint_name(self, step: int, epoch: int = 0, val_loss: Optional[float] = None) -> str:
        """Format checkpoint filename based on pattern."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = self.name_pattern.format(
            step=step,
            epoch=epoch,
            val_loss=f"{val_loss:.4f}" if val_loss is not None else "none",
            timestamp=timestamp
        )
        return f"{name}.pt"

    def save_checkpoint(
        self,
        step: int,
        epoch: int,
        models: Dict[str, torch.nn.Module],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[dict] = None,
        val_loss: Optional[float] = None,
        **kwargs
    ) -> Optional[Path]:
        """Save a checkpoint with the specified components."""
        if not self.enabled:
            return None
            
        filename = self.format_checkpoint_name(step, epoch, val_loss)
        filepath = self.output_dir / filename
        
        # Prepare checkpoint data
        checkpoint_data = {
            'step': step,
            'epoch': epoch,
            **kwargs  # Additional data like tau, alpha, etc.
        }
        
        # Add optional components
        if self.save_components.get('metrics', True) and metrics is not None:
            checkpoint_data['metrics'] = metrics
            
        if self.save_components.get('config', True) and config is not None:
            checkpoint_data['config'] = config
        
        # Save using the existing checkpoint utility
        checkpoint.save(
            path=str(filepath),
            models=models if self.save_components.get('models', True) else {},
            optim=optimizer if self.save_components.get('optimizer', True) else None,
            scheduler=scheduler if self.save_components.get('scheduler', False) else None,
            **checkpoint_data
        )
        
        self.logger.info(f"Saved checkpoint to {filepath}")
        
        # Track checkpoint
        self.all_checkpoints.append(filepath)
        
        # Handle best checkpoint tracking
        if self.track_best_n > 0 and val_loss is not None:
            self._update_best_checkpoints(val_loss, filepath)
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        return filepath

    def _update_best_checkpoints(self, metric_value: float, filepath: Path):
        """Update list of best checkpoints."""
        # Add new checkpoint
        self.best_checkpoints.append((metric_value, filepath))
        
        # Sort based on mode
        reverse = (self.best_mode == 'max')
        self.best_checkpoints.sort(key=lambda x: x[0], reverse=reverse)
        
        # Keep only top N
        if len(self.best_checkpoints) > self.track_best_n:
            # Remove worst checkpoints
            removed = self.best_checkpoints[self.track_best_n:]
            self.best_checkpoints = self.best_checkpoints[:self.track_best_n]
            
            # Delete removed checkpoints if they're not in all_checkpoints
            for _, path in removed:
                if path not in [p for _, p in self.best_checkpoints]:
                    self._safe_delete_checkpoint(path)
                    
        # Log current best
        if self.best_checkpoints:
            best_val, best_path = self.best_checkpoints[0]
            self.logger.info(f"Best checkpoint: {best_path.name} ({self.best_metric}={best_val:.4f})")

    def _cleanup_checkpoints(self):
        """Remove old checkpoints if max_checkpoints is set."""
        if not self.delete_old or self.max_checkpoints < 0:
            return
            
        # Get checkpoints that should be kept
        keep_paths = set()
        
        # Keep best checkpoints
        for _, path in self.best_checkpoints:
            keep_paths.add(path)
        
        # Keep most recent checkpoints up to max_checkpoints
        recent_checkpoints = sorted(self.all_checkpoints, key=lambda p: p.stat().st_mtime, reverse=True)
        for path in recent_checkpoints[:self.max_checkpoints]:
            keep_paths.add(path)
        
        # Delete checkpoints not in keep list
        for path in self.all_checkpoints[:]:
            if path not in keep_paths and path.exists():
                self._safe_delete_checkpoint(path)
                self.all_checkpoints.remove(path)

    def _safe_delete_checkpoint(self, filepath: Path):
        """Safely delete a checkpoint file."""
        try:
            if filepath.exists():
                filepath.unlink()
                self.logger.info(f"Deleted old checkpoint: {filepath.name}")
        except Exception as e:
            self.logger.warning(f"Failed to delete checkpoint {filepath}: {e}")

    def load_checkpoint(self, checkpoint_path: str, models: Dict[str, torch.nn.Module], 
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       map_location: Any = None) -> dict:
        """Load a checkpoint and return the metadata."""
        return checkpoint.load(checkpoint_path, models=models, optim=optimizer, map_location=map_location)

    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to the best checkpoint."""
        if self.best_checkpoints:
            return self.best_checkpoints[0][1]
        return None
    
    def get_latest_checkpoint_path(self) -> Optional[Path]:
        """Get path to the most recent checkpoint."""
        if self.all_checkpoints:
            return max(self.all_checkpoints, key=lambda p: p.stat().st_mtime)
        return None 