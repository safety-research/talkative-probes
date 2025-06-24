#!/usr/bin/env python3
"""Interactive Consistency Lens Analysis Tool

This script provides an interactive interface for analyzing arbitrary text strings
through a trained consistency lens checkpoint. It allows real-time exploration
of how the lens interprets different inputs.

Usage:
    uv run python scripts/interactive_lens_analysis.py checkpoint=path/to/checkpoint.pt
"""

import torch
import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich import print as rprint
import logging

# Add parent d
# Add parent directory to path for imports
toadd = str(Path(__file__).parent)
print("Adding to path:",toadd)
sys.path.append(toadd)

from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.orig import OrigWrapper
from lens.utils.embedding_remap import remap_embeddings

console = Console()

class InteractiveLensAnalyzer:
    """Interactive interface for consistency lens analysis."""
    
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
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
        
    def _load_checkpoint(self):
        """Load models and configuration from checkpoint."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Loading checkpoint...", total=None)
            
            # Load checkpoint data
            ckpt_data = torch.load(self.checkpoint_path, map_location=self.device)
            progress.update(task, description="Checkpoint loaded")
            
            # Extract configuration
            if 'config' in ckpt_data:
                self.config = ckpt_data['config']
                self.layer_l = self.config.get('layer_l', 5)
                model_name = self.config.get('model_name', 'gpt2')
                tokenizer_name = self.config.get('tokenizer_name', model_name)
            else:
                # Fallback for old checkpoints
                self.console.print("[yellow]Warning: No config found in checkpoint, using defaults[/yellow]")
                self.layer_l = 5
                model_name = 'gpt2'
                tokenizer_name = model_name
                
            progress.update(task, description=f"Building models ({model_name})...")
            
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
            
            progress.update(task, description="Loading model weights...")
            
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
            
            progress.update(task, description="Ready!")
    
    def analyze_string(self, text: str, position: Optional[int] = None) -> Dict[str, Any]:
        """Analyze a single string through the lens.
        
        Args:
            text: Input text to analyze
            position: Token position to analyze (None = last token)
            
        Returns:
            Dictionary with analysis results
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(self.device)
        seq_len = input_ids.shape[1]
        
        # Determine position to analyze
        if position is None:
            position = seq_len - 1  # Last token
        elif position < 0:
            position = seq_len + position  # Handle negative indexing
        elif position >= seq_len:
            position = seq_len - 1
            
        with torch.no_grad():
            # Get activations
            outputs = self.orig_model.model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            A = hidden_states[self.layer_l + 1][:, position, :]
            
            # Generate explanation
            tau = self.config.get("gumbel_tau_schedule", {}).get("end_value", 1.0)
            t_text = self.config.get("t_text", 5)
            
            if self.models["dec"].config.use_flash_attention:
                gen = self.models["dec"].generate_soft_kv_flash(A, max_length=t_text, gumbel_tau=tau)
            elif self.models["dec"].config.use_kv_cache:
                gen = self.models["dec"].generate_soft_kv_cached(A, max_length=t_text, gumbel_tau=tau)
            else:
                gen = self.models["dec"].generate_soft(A, max_length=t_text, gumbel_tau=tau)
            
            # Get reconstruction
            A_hat = self.models["enc"](gen.generated_text_embeddings)
            
            # Compute metrics
            mse = torch.nn.functional.mse_loss(A, A_hat).item()
            cosine_sim = torch.nn.functional.cosine_similarity(A, A_hat, dim=-1).mean().item()
            
            # Decode tokens
            generated_tokens = gen.hard_token_ids[0].cpu().tolist()
            explanation = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Get analyzed token
            analyzed_token = self.tokenizer.decode([input_ids[0, position].item()])
            
            # Get token list for display
            tokens = [self.tokenizer.decode([tok_id]) for tok_id in input_ids[0].tolist()]
            
        return {
            'text': text,
            'tokens': tokens,
            'position': position,
            'analyzed_token': analyzed_token,
            'explanation': explanation,
            'mse': mse,
            'cosine_similarity': cosine_sim,
            'generated_tokens': generated_tokens,
            'layer': self.layer_l
        }
    
    def display_results(self, results: Dict[str, Any]):
        """Display analysis results in a formatted way."""
        # Create main results panel
        content = f"""[bold]Input Text:[/bold] {results['text'][:100]}{'...' if len(results['text']) > 100 else ''}
[bold]Analyzed Token:[/bold] '{results['analyzed_token']}' at position {results['position']}
[bold]Layer:[/bold] {results['layer']}

[bold cyan]Explanation:[/bold cyan] {results['explanation']}

[bold]Metrics:[/bold]
  • MSE: {results['mse']:.6f}
  • Cosine Similarity: {results['cosine_similarity']:.4f}"""
        
        self.console.print(Panel(content, title="Analysis Results", border_style="blue"))
        
        # Show token breakdown if requested
        if Confirm.ask("Show token breakdown?", default=False):
            table = Table(title="Token Analysis", show_header=True, header_style="bold magenta")
            table.add_column("Position", style="dim", width=8)
            table.add_column("Token", width=20)
            table.add_column("Analyzed", width=8)
            
            for i, token in enumerate(results['tokens']):
                is_analyzed = "✓" if i == results['position'] else ""
                style = "bold green" if i == results['position'] else ""
                table.add_row(str(i), repr(token), is_analyzed, style=style)
            
            self.console.print(table)
    
    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple texts in batch."""
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task(f"Analyzing {len(texts)} strings...", total=len(texts))
            
            for i, text in enumerate(texts):
                result = self.analyze_string(text)
                results.append(result)
                progress.update(task, advance=1, description=f"Analyzing string {i+1}/{len(texts)}")
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], filename: Optional[str] = None):
        """Save analysis results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lens_analysis_{timestamp}.json"
        
        output_path = Path(filename)
        
        # Prepare data for JSON serialization
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'checkpoint': str(self.checkpoint_path),
            'layer': self.layer_l,
            'device': str(self.device),
            'analyses': results
        }
        
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        self.console.print(f"[green]✓[/green] Results saved to: {output_path}")
    
    def run_interactive_session(self):
        """Run the main interactive session."""
        self.console.print(Panel.fit(
            f"""[bold cyan]Consistency Lens Interactive Analyzer[/bold cyan]
            
Checkpoint: {self.checkpoint_path.name}
Layer: {self.layer_l}
Device: {self.device}
Model: {self.config.get('model_name', 'Unknown')}

Commands:
  • Enter text to analyze
  • 'batch' - Analyze multiple strings
  • 'file' - Load strings from file
  • 'save' - Save results
  • 'config' - Show configuration
  • 'help' - Show help
  • 'quit' - Exit
""", 
            title="Welcome", 
            border_style="bold blue"
        ))
        
        session_results = []
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold green]Enter text or command[/bold green]")
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    if session_results and Confirm.ask("Save session results before exiting?"):
                        self.save_results(session_results)
                    self.console.print("[yellow]Goodbye![/yellow]")
                    break
                
                elif user_input.lower() == 'help':
                    self.show_help()
                
                elif user_input.lower() == 'config':
                    self.show_config()
                
                elif user_input.lower() == 'batch':
                    texts = self.get_batch_input()
                    if texts:
                        results = self.batch_analyze(texts)
                        session_results.extend(results)
                        self.display_batch_summary(results)
                
                elif user_input.lower() == 'file':
                    filepath = Prompt.ask("Enter file path")
                    texts = self.load_from_file(filepath)
                    if texts:
                        results = self.batch_analyze(texts)
                        session_results.extend(results)
                        self.display_batch_summary(results)
                
                elif user_input.lower() == 'save':
                    if session_results:
                        filename = Prompt.ask("Enter filename (or press Enter for auto)", default="")
                        self.save_results(session_results, filename if filename else None)
                    else:
                        self.console.print("[yellow]No results to save yet[/yellow]")
                
                else:
                    # Analyze the input text
                    # Check if user wants to specify position
                    if '@' in user_input:
                        text, pos_str = user_input.rsplit('@', 1)
                        try:
                            position = int(pos_str)
                        except ValueError:
                            position = None
                            text = user_input
                    else:
                        text = user_input
                        position = None
                    
                    result = self.analyze_string(text, position)
                    session_results.append(result)
                    self.display_results(result)
                    
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use 'quit' to exit properly[/yellow]")
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
    
    def show_help(self):
        """Display help information."""
        help_text = """[bold cyan]Help - Interactive Commands[/bold cyan]

[bold]Basic Usage:[/bold]
  • Type any text and press Enter to analyze it
  • The lens will analyze the LAST token by default
  • Use '@position' to analyze a specific token position
    Example: "Hello world@0" analyzes the first token
    Example: "Hello world@-2" analyzes the second-to-last token

[bold]Commands:[/bold]
  • batch - Enter multiple strings for batch analysis
  • file - Load strings from a text file (one per line)
  • save - Save all results from current session
  • config - Display model configuration
  • help - Show this help message
  • quit - Exit the program

[bold]Tips:[/bold]
  • Position indices can be negative (Python-style)
  • Results accumulate during the session
  • Save your results before quitting!"""
        
        self.console.print(Panel(help_text, title="Help", border_style="blue"))
    
    def show_config(self):
        """Display current configuration."""
        config_info = f"""[bold]Model Configuration:[/bold]
  • Model: {self.config.get('model_name', 'Unknown')}
  • Tokenizer: {self.config.get('tokenizer_name', 'Unknown')}
  • Layer: {self.layer_l}
  • Decoder Prompt: '{self.config.get('decoder_prompt', 'Unknown')}'
  • t_text: {self.config.get('t_text', 'Unknown')}
  • Device: {self.device}
  
[bold]Checkpoint Info:[/bold]
  • Path: {self.checkpoint_path}
  • Step: {self.config.get('step', 'Unknown')}
  • Epoch: {self.config.get('epoch', 'Unknown')}"""
        
        self.console.print(Panel(config_info, title="Configuration", border_style="blue"))
    
    def get_batch_input(self) -> List[str]:
        """Get multiple strings from user input."""
        self.console.print("[cyan]Enter strings one per line. Empty line to finish:[/cyan]")
        texts = []
        while True:
            line = Prompt.ask("", default="")
            if not line:
                break
            texts.append(line)
        return texts
    
    def load_from_file(self, filepath: str) -> List[str]:
        """Load strings from a text file."""
        try:
            path = Path(filepath)
            if not path.exists():
                self.console.print(f"[red]File not found: {filepath}[/red]")
                return []
            
            with open(path, 'r') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            self.console.print(f"[green]Loaded {len(texts)} strings from {filepath}[/green]")
            return texts
        except Exception as e:
            self.console.print(f"[red]Error loading file: {e}[/red]")
            return []
    
    def display_batch_summary(self, results: List[Dict[str, Any]]):
        """Display summary of batch analysis results."""
        table = Table(title="Batch Analysis Summary", show_header=True, header_style="bold magenta")
        table.add_column("Text", style="dim", width=40)
        table.add_column("Token", width=15)
        table.add_column("Explanation", width=40)
        table.add_column("MSE", justify="right", width=10)
        table.add_column("Cos Sim", justify="right", width=10)
        
        for result in results:
            text_preview = result['text'][:37] + "..." if len(result['text']) > 40 else result['text']
            explanation_preview = result['explanation'][:37] + "..." if len(result['explanation']) > 40 else result['explanation']
            
            table.add_row(
                text_preview,
                repr(result['analyzed_token']),
                explanation_preview,
                f"{result['mse']:.4f}",
                f"{result['cosine_similarity']:.3f}"
            )
        
        self.console.print(table)
        
        # Summary statistics
        avg_mse = sum(r['mse'] for r in results) / len(results)
        avg_cos = sum(r['cosine_similarity'] for r in results) / len(results)
        
        self.console.print(f"\n[bold]Summary:[/bold] {len(results)} strings analyzed")
        self.console.print(f"Average MSE: {avg_mse:.6f}, Average Cosine Similarity: {avg_cos:.4f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Interactive Consistency Lens Analysis")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--device", help="Device to use (cuda/cpu)", default=None)
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        console.print(f"[red]Error: Checkpoint not found: {args.checkpoint}[/red]")
        sys.exit(1)
    
    # Suppress transformers logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    try:
        analyzer = InteractiveLensAnalyzer(args.checkpoint, args.device)
        analyzer.run_interactive_session()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()