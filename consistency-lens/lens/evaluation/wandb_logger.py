"""WandB logging utilities for verbose samples."""

from __future__ import annotations

import dataclasses
from typing import List, Dict, Any, Optional, Union


@dataclasses.dataclass
class VerboseSamplesLogger:
    """Logger for verbose samples to WandB tables with workaround for table update issues."""
    
    def __init__(self):
        self.verbose_samples_table = None
        self.table_initialized = False
    
    def log_verbose_samples(self, samples_data: Union[str, List[Dict[str, Any]]], step: int, table_name: str = "verbose_samples") -> None:
        """Write verbose-sample information to W&B.

        The table now has only:
          • step   – training step  
          • details – a single long, raw-text blob (no markdown) that
                      concatenates the dumps for **all** samples logged
                      at this step.
                      
        Args:
            samples_data: Either a raw text string (preferred) or list of structured data dicts
            step: Training step
            table_name: Name of the table in wandb
        """
        try:
            import wandb
        except ImportError:
            return

        if not samples_data:
            return

        # Handle both raw text string and structured data
        if isinstance(samples_data, str):
            # Already formatted as raw text - use directly
            details_blob = samples_data
        else:
            # Legacy structured data format - convert to text
            blocks: list[str] = []
            for idx, sample in enumerate(samples_data):
                lines: list[str] = [f"--- Verbose sample {idx} ---"]
                for key, val in sample.items():
                    if isinstance(val, list):
                        lines.append(f"{key}:")
                        for item in val:
                            lines.append(f"  {item}")
                    else:
                        lines.append(f"{key}: {val}")
                blocks.append("\n".join(lines))
            details_blob: str = "\n\n".join(blocks)

        # ------------------------------------------------------------------
        # 2) Create / update the 2-column table
        # ------------------------------------------------------------------
        columns = ["step", "details"]
        if not self.table_initialized:
            self.verbose_samples_table = wandb.Table(columns=columns)
            self.table_initialized = True
        else:
            new_table = wandb.Table(columns=columns, data=self.verbose_samples_table.data)
            self.verbose_samples_table = new_table

        self.verbose_samples_table.add_data(step, details_blob)

        # Push to W&B
        wandb.log({f"samples/{table_name}": self.verbose_samples_table}, step=step)


# Global instance for easy access
verbose_samples_logger = VerboseSamplesLogger()