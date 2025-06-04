#!/usr/bin/env python3
"""Analyze all tokens in a string through the consistency lens.

This script provides functions to analyze every token position in an input string,
showing the generated explanations and KL divergence for each position.

Usage in Python interactive session:
    >>> from analyze_all_tokens import LensAnalyzer
    >>> analyzer = LensAnalyzer("path/to/checkpoint.pt")
    >>> analyzer.analyze_all_positions("The cat sat on the mat")
"""

import torch
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.text import Text
from rich import print as rprint

# Add parent directory to path for imports
toadd = str(Path(__file__).parent)
print(toadd)
sys.path.append(toadd)


from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.orig import OrigWrapper
from lens.utils.embedding_remap import remap_embeddings
import torch.nn.functional as F

console = Console()


class LensAnalyzer:
    """Analyzer for consistency lens on all token positions."""
    
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        """Initialize the analyzer with a checkpoint.
        
        Args:
            checkpoint_path: Path to the consistency lens checkpoint
            device: Device to use ('cuda' or 'cpu'), auto-detected if None
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.console = Console()
        
        # Initialize models and config
        self.models = {}
        self.tokenizer = None
        self.orig_model = None
        self.config = {}
        self.layer_l = None
        
        self._load_checkpoint()
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
        print(f"  Model: {self.config.get('model_name', 'Unknown')}")
        print(f"  Layer: {self.layer_l}")
        print(f"  Device: {self.device}")
    
    def _load_checkpoint(self):
        """Load models and configuration from checkpoint."""
        # Load checkpoint data
        ckpt_data = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Extract configuration
        if 'config' in ckpt_data:
            self.config = ckpt_data['config']
            self.layer_l = self.config.get('layer_l', 5)
            model_name = self.config.get('model_name', 'gpt2')
            tokenizer_name = self.config.get('tokenizer_name', model_name)
        else:
            self.layer_l = 5
            model_name = 'gpt2'
            tokenizer_name = model_name
            
        # Build models
        dec_cfg = DecoderConfig(model_name=model_name)
        self.models['dec'] = Decoder(dec_cfg).to(self.device)
        
        enc_cfg = EncoderConfig(model_name=model_name)
        self.models['enc'] = Encoder(enc_cfg).to(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load original model
        self.orig_model = OrigWrapper(model_name, load_in_8bit=False)
        self.orig_model.model.to(self.device)
        
        # Handle tokenizer remapping if needed
        if tokenizer_name != model_name:
            base_tok = AutoTokenizer.from_pretrained(model_name)
            remap_embeddings(self.models['dec'].base, base_tok, self.tokenizer)
            if self.models['enc'].base is not None:
                remap_embeddings(self.models['enc'].base, base_tok, self.tokenizer)
            remap_embeddings(self.orig_model.model, base_tok, self.tokenizer)
        
        # Load model weights
        if "models" in ckpt_data:
            for model_name, model in self.models.items():
                if model_name in ckpt_data["models"]:
                    state_dict = ckpt_data["models"][model_name]
                    if model_name == "dec" and "prompt_ids" in state_dict:
                        state_dict.pop("prompt_ids")
                    model.load_state_dict(state_dict, strict=False)
        
        # Set decoder prompt
        decoder_prompt = self.config.get('decoder_prompt', 'Explain: ')
        self.models['dec'].set_prompt(decoder_prompt, self.tokenizer)
        
        # Set models to eval mode
        self.models['dec'].eval()
        self.models['enc'].eval()
    
    def compute_kl_divergence(self, A: torch.Tensor, A_hat: torch.Tensor, 
                            input_ids: torch.Tensor, position: int) -> float:
        """Compute KL divergence between original and reconstructed model outputs.
        
        This computes KL(P_orig || P_reconstructed) where:
        - P_orig: predictions from original activations
        - P_reconstructed: predictions from reconstructed activations
        
        Args:
            A: Original activation at layer L
            A_hat: Reconstructed activation
            input_ids: Input token IDs (full sequence)
            position: Position of the analyzed token
            
        Returns:
            KL divergence value
        """
        with torch.no_grad():
            # Get predictions with original activation
            logits_orig = self.orig_model.forward_with_replacement(
                input_ids=input_ids,
                new_activation=A,
                layer_idx=self.layer_l,
                token_pos=position,
                no_grad=True
            ).logits
            
            # Get predictions with reconstructed activation
            logits_recon = self.orig_model.forward_with_replacement(
                input_ids=input_ids,
                new_activation=A_hat,
                layer_idx=self.layer_l,
                token_pos=position,
                no_grad=True
            ).logits
            
            # Extract logits at the position (predicting next token)
            logits_orig_at_pos = logits_orig[:, position, :]
            logits_recon_at_pos = logits_recon[:, position, :]
            
            # Compute KL divergence: KL(P_orig || P_recon)
            # Using numerical stability best practices
            with torch.amp.autocast('cuda', enabled=False):
                logits_orig_f32 = logits_orig_at_pos.float()
                logits_recon_f32 = logits_recon_at_pos.float()
                
                # Numerical stability: subtract max before softmax
                logits_orig_f32 = logits_orig_f32 - logits_orig_f32.max(dim=-1, keepdim=True)[0].detach()
                logits_recon_f32 = logits_recon_f32 - logits_recon_f32.max(dim=-1, keepdim=True)[0].detach()
                
                # Convert to probabilities
                log_probs_orig = torch.log_softmax(logits_orig_f32, dim=-1)
                log_probs_recon = torch.log_softmax(logits_recon_f32, dim=-1)
                
                # KL(P_orig || P_recon) = sum(P_orig * (log P_orig - log P_recon))
                probs_orig = torch.exp(log_probs_orig)
                kl_div = (probs_orig * (log_probs_orig - log_probs_recon)).sum(dim=-1).mean().item()
            
            return kl_div
    
    def analyze_position(self, input_ids: torch.Tensor, position: int, 
                        hidden_states: List[torch.Tensor]) -> Dict[str, Any]:
        """Analyze a single token position.
        
        Args:
            input_ids: Input token IDs
            position: Position to analyze
            hidden_states: Pre-computed hidden states from the model
            
        Returns:
            Dictionary with analysis results
        """
        with torch.no_grad():
            # Extract activation at this position
            A = hidden_states[self.layer_l + 1][:, position, :]
            
            # Generate explanation
            tau = self.config.get("gumbel_tau_schedule", {}).get("end_value", 1.0)
            T_text = self.config.get("t_text", 5)
            
            if self.models["dec"].config.use_flash_attention:
                gen = self.models["dec"].generate_soft_kv_flash(A, max_length=T_text, gumbel_tau=tau)
            elif self.models["dec"].config.use_kv_cache:
                gen = self.models["dec"].generate_soft_kv_cached(A, max_length=T_text, gumbel_tau=tau)
            else:
                gen = self.models["dec"].generate_soft(A, max_length=T_text, gumbel_tau=tau)
            
            # Get reconstruction
            A_hat = self.models["enc"](gen.generated_text_embeddings)
            
            # Compute KL divergence
            kl_div = self.compute_kl_divergence(A, A_hat, input_ids, position)
            
            # Decode tokens
            generated_tokens = gen.hard_token_ids[0].cpu().tolist()
            explanation = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Get the token at this position
            token = self.tokenizer.decode([input_ids[0, position].item()])
            
            return {
                'position': position,
                'token': token,
                'explanation': explanation,
                'kl_divergence': kl_div,
                'generated_tokens': generated_tokens
            }
    
    def analyze_all_positions(self, text: str, show_progress: bool = True) -> List[Dict[str, Any]]:
        """Analyze all token positions in the input text.
        
        Args:
            text: Input text to analyze
            show_progress: Whether to show progress bar
            
        Returns:
            List of analysis results for each position
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(self.device)
        seq_len = input_ids.shape[1]
        
        # Get all hidden states at once
        with torch.no_grad():
            outputs = self.orig_model.model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states
        
        results = []
        
        # Analyze each position
        if show_progress:
            positions = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            )
            with positions:
                task = positions.add_task(f"Analyzing {seq_len} tokens", total=seq_len)
                for pos in range(seq_len):
                    result = self.analyze_position(input_ids, pos, hidden_states)
                    results.append(result)
                    positions.update(task, advance=1)
        else:
            for pos in range(seq_len):
                result = self.analyze_position(input_ids, pos, hidden_states)
                results.append(result)
        
        # Display results
        self._display_results(text, results)
        
        return results
    
    def _display_results(self, text: str, results: List[Dict[str, Any]]):
        """Display analysis results in a formatted table."""
        # Create header
        self.console.print(f"\n[bold cyan]Token-by-Token Analysis[/bold cyan]")
        self.console.print(f"Input: \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
        self.console.print(f"Total tokens: {len(results)}")
        
        # Create table
        table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 1))
        table.add_column("Pos", style="dim", width=4, justify="right")
        table.add_column("Token", width=15)
        table.add_column("Explanation", width=50)
        table.add_column("KL Div", justify="right", width=8)
        
        # Color scale for KL divergence (green = low, red = high)
        kl_values = [r['kl_divergence'] for r in results]
        if kl_values:
            min_kl = min(kl_values)
            max_kl = max(kl_values)
            kl_range = max_kl - min_kl if max_kl > min_kl else 1.0
        else:
            min_kl = max_kl = kl_range = 1.0
        
        for result in results:
            # Color code KL divergence
            kl = result['kl_divergence']
            normalized_kl = (kl - min_kl) / kl_range if kl_range > 0 else 0.5
            
            if normalized_kl < 0.33:
                kl_color = "green"
            elif normalized_kl < 0.67:
                kl_color = "yellow"
            else:
                kl_color = "red"
            
            # Truncate explanation if too long
            explanation = result['explanation']
            if len(explanation) > 47:
                explanation = explanation[:47] + "..."
            
            table.add_row(
                str(result['position']),
                repr(result['token']),
                explanation,
                f"[{kl_color}]{kl:.3f}[/{kl_color}]"
            )
        
        self.console.print(table)
        
        # Summary statistics
        avg_kl = np.mean(kl_values)
        std_kl = np.std(kl_values)
        self.console.print(f"\n[bold]Summary:[/bold]")
        self.console.print(f"  Average KL divergence: {avg_kl:.3f} (±{std_kl:.3f})")
        self.console.print(f"  Min KL: {min_kl:.3f}, Max KL: {max_kl:.3f}")
        
        # Find most/least interpretable tokens
        sorted_results = sorted(results, key=lambda x: x['kl_divergence'])
        self.console.print(f"\n[bold]Most consistent (lowest KL):[/bold]")
        for r in sorted_results[:3]:
            self.console.print(f"  {r['position']:2d}. '{r['token']}' → \"{r['explanation']}\" (KL: {r['kl_divergence']:.3f})")
        
        self.console.print(f"\n[bold]Least consistent (highest KL):[/bold]")
        for r in sorted_results[-3:]:
            self.console.print(f"  {r['position']:2d}. '{r['token']}' → \"{r['explanation']}\" (KL: {r['kl_divergence']:.3f})")
    
    def plot_kl_trajectory(self, results: List[Dict[str, Any]], title: Optional[str] = None):
        """Plot KL divergence across token positions (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            
            positions = [r['position'] for r in results]
            kl_values = [r['kl_divergence'] for r in results]
            tokens = [r['token'] for r in results]
            
            plt.figure(figsize=(12, 6))
            plt.plot(positions, kl_values, 'b-', linewidth=2, marker='o', markersize=6)
            
            # Annotate some points with tokens
            step = max(1, len(positions) // 10)  # Show ~10 labels
            for i in range(0, len(positions), step):
                plt.annotate(repr(tokens[i]), 
                           (positions[i], kl_values[i]),
                           textcoords="offset points",
                           xytext=(0,10),
                           ha='center',
                           fontsize=8)
            
            plt.xlabel('Token Position')
            plt.ylabel('KL Divergence')
            plt.title(title or 'KL Divergence Across Token Positions')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            self.console.print("[yellow]Matplotlib not available for plotting[/yellow]")
    
    def compare_strings(self, texts: List[str]) -> None:
        """Compare analysis results across multiple strings."""
        all_results = []
        
        self.console.print(f"\n[bold cyan]Comparing {len(texts)} strings[/bold cyan]\n")
        
        for i, text in enumerate(texts):
            self.console.print(f"[bold]String {i+1}:[/bold] \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
            results = self.analyze_all_positions(text, show_progress=False)
            all_results.append({
                'text': text,
                'results': results,
                'avg_kl': np.mean([r['kl_divergence'] for r in results])
            })
            self.console.print(f"  Average KL: {all_results[-1]['avg_kl']:.3f}\n")
        
        # Summary comparison
        self.console.print("[bold]Comparison Summary:[/bold]")
        sorted_results = sorted(all_results, key=lambda x: x['avg_kl'])
        
        for i, res in enumerate(sorted_results):
            self.console.print(f"{i+1}. \"{res['text'][:40]}{'...' if len(res['text']) > 40 else ''}\"")
            self.console.print(f"   Average KL: {res['avg_kl']:.3f}")


# Convenience function for interactive use
def quick_analyze(checkpoint_path: str, text: str):
    """Quick analysis function for interactive Python sessions.
    
    Example:
        >>> quick_analyze("checkpoint.pt", "Hello world!")
    """
    analyzer = LensAnalyzer(checkpoint_path)
    return analyzer.analyze_all_positions(text)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python analyze_all_tokens.py <checkpoint_path> <text>")
        print("\nFor interactive use:")
        print("  >>> from analyze_all_tokens import LensAnalyzer")
        print("  >>> analyzer = LensAnalyzer('checkpoint.pt')")
        print("  >>> analyzer.analyze_all_positions('Your text here')")
        sys.exit(1)
    
    checkpoint = sys.argv[1]
    text = " ".join(sys.argv[2:])
    
    analyzer = LensAnalyzer(checkpoint)
    analyzer.analyze_all_positions(text)