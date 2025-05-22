"""WandB logging utilities for verbose samples."""

from __future__ import annotations

import dataclasses
from typing import List, Dict, Any, Optional


@dataclasses.dataclass
class VerboseSamplesLogger:
    """Logger for verbose samples to WandB tables with workaround for table update issues."""
    
    def __init__(self):
        self.verbose_samples_table = None
        self.table_initialized = False
    
    def log_verbose_samples(self, samples_data: List[Dict[str, Any]], step: int, table_name: str = "verbose_samples") -> None:
        """Log verbose samples to wandb as a table.
        
        Args:
            samples_data: List of sample dictionaries with keys:
                - input_text: The input text with context
                - chosen_token: The analyzed token
                - position: Token position
                - decoded_text: Decoder output
                - top_predictions: List of (token, prob) tuples
                - continuation: Optional autoregressive continuation
                - original_logits: Top predictions from original model
                - reconstructed_logits: Top predictions from reconstructed model
            step: Current training step
            table_name: Name for the wandb table
        """
        try:
            import wandb
        except ImportError:
            return
        
        if not samples_data:
            return
        
        # Define columns based on what data is available
        columns = [
            "step",
            "sample_idx", 
            "input_text",
            "chosen_token",
            "position",
            "decoded_explanation",
        ]
        
        # Check if we have predictions data
        if samples_data[0].get("top_predictions"):
            columns.extend([
                "top1_token", "top1_prob",
                "top2_token", "top2_prob", 
                "top3_token", "top3_prob"
            ])
        
        # Check if we have original/reconstructed predictions
        if samples_data[0].get("original_logits"):
            columns.extend([
                "orig_top1", "orig_prob1",
                "recon_top1", "recon_prob1"
            ])
        
        # Check if we have continuation
        if samples_data[0].get("continuation"):
            columns.append("continuation")
        
        # Initialize or update table
        if not self.table_initialized:
            self.verbose_samples_table = wandb.Table(columns=columns)
            self.table_initialized = True
        else:
            # Workaround for wandb table update issue
            new_table = wandb.Table(columns=columns, data=self.verbose_samples_table.data)
            self.verbose_samples_table = new_table
        
        # Add rows for each sample
        for idx, sample in enumerate(samples_data):
            row_data = [step, idx]
            
            # Basic info
            row_data.extend([
                sample.get("input_text", ""),
                sample.get("chosen_token", ""),
                sample.get("position", -1),
                sample.get("decoded_text", ""),
            ])
            
            # Top predictions from decoder
            if sample.get("top_predictions"):
                preds = sample["top_predictions"]
                for i in range(3):
                    if i < len(preds):
                        row_data.extend([preds[i][0], f"{preds[i][1]:.3f}"])
                    else:
                        row_data.extend(["", ""])
            
            # Original vs reconstructed predictions
            if sample.get("original_logits"):
                orig = sample.get("original_logits", [])
                recon = sample.get("reconstructed_logits", [])
                
                if orig and len(orig) > 0:
                    row_data.extend([orig[0][0], f"{orig[0][1]:.3f}"])
                else:
                    row_data.extend(["", ""])
                    
                if recon and len(recon) > 0:
                    row_data.extend([recon[0][0], f"{recon[0][1]:.3f}"])
                else:
                    row_data.extend(["", ""])
            
            # Continuation
            if sample.get("continuation") is not None:
                row_data.append(sample["continuation"])
            
            self.verbose_samples_table.add_data(*row_data)
        
        # Log the table
        wandb.log({f"samples/{table_name}": self.verbose_samples_table}, step=step)


# Global instance for easy access
verbose_samples_logger = VerboseSamplesLogger()