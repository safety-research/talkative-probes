#!/usr/bin/env python3
"""Simple interactive functions for consistency lens analysis.

Quick start in Python:
    >>> from lens_interactive import *
    >>> lens = load_lens("path/to/checkpoint.pt")
    >>> analyze("Hello world!")
"""

import torch
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.orig import OrigWrapper
from lens.utils.embedding_remap import remap_embeddings

# Global analyzer instance for convenience
_analyzer = None


def load_lens(checkpoint_path: str, device: Optional[str] = None) -> 'SimpleAnalyzer':
    """Load a consistency lens checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to use ('cuda' or 'cpu'), auto-detected if None
    
    Returns:
        SimpleAnalyzer instance
        
    Example:
        >>> lens = load_lens("checkpoint.pt")
    """
    global _analyzer
    _analyzer = SimpleAnalyzer(checkpoint_path, device)
    print(f"✓ Lens loaded! Model: {_analyzer.model_name}, Layer: {_analyzer.layer}")
    return _analyzer


def analyze(text: str, positions: Optional[List[int]] = None):
    """Analyze text using the loaded lens.
    
    Args:
        text: Text to analyze
        positions: Specific positions to analyze (None = all positions)
        
    Example:
        >>> analyze("The cat sat on the mat")
        >>> analyze("Hello world", positions=[0, -1])  # First and last tokens
    """
    if _analyzer is None:
        print("No lens loaded! Use load_lens() first.")
        return
    
    if positions is None:
        _analyzer.show_all_tokens(text)
    else:
        results = []
        for pos in positions:
            res = _analyzer.analyze_token(text, pos)
            results.append(res)
        _analyzer.display_results(results)


def analyze_last(text: str):
    """Analyze only the last token of the text.
    
    Example:
        >>> analyze_last("The cat sat on the mat")
    """
    if _analyzer is None:
        print("No lens loaded! Use load_lens() first.")
        return
    
    result = _analyzer.analyze_token(text, -1)
    print(f"\nText: \"{text}\"")
    print(f"Last token: '{result['token']}' at position {result['position']}")
    print(f"Explanation: \"{result['explanation']}\"")
    print(f"KL divergence: {result['kl_divergence']:.3f}")


def compare(*texts: str):
    """Compare multiple texts by analyzing all their tokens.
    
    Example:
        >>> compare("Hello world", "Goodbye world", "Hello there")
    """
    if _analyzer is None:
        print("No lens loaded! Use load_lens() first.")
        return
    
    _analyzer.compare_texts(list(texts))


class SimpleAnalyzer:
    """Simple analyzer for interactive use."""
    
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = Path(checkpoint_path)
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract config
        if 'config' in ckpt:
            config = ckpt['config']
            self.layer = config.get('layer_l', 5)
            self.model_name = config.get('model_name', 'gpt2')
            tokenizer_name = config.get('tokenizer_name', self.model_name)
            self.t_text = config.get('t_text', 5)
            self.tau = config.get('gumbel_tau_schedule', {}).get('end_value', 1.0)
            decoder_prompt = config.get('decoder_prompt', 'Explain: ')
        else:
            self.layer = 5
            self.model_name = 'gpt2'
            tokenizer_name = self.model_name
            self.t_text = 5
            self.tau = 1.0
            decoder_prompt = 'Explain: '
        
        # Build models
        self.decoder = Decoder(DecoderConfig(model_name=self.model_name)).to(self.device)
        self.encoder = Encoder(EncoderConfig(model_name=self.model_name)).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.orig_model = OrigWrapper(self.model_name, load_in_8bit=False)
        self.orig_model.model.to(self.device)
        
        # Remap if needed
        if tokenizer_name != self.model_name:
            base_tok = AutoTokenizer.from_pretrained(self.model_name)
            remap_embeddings(self.decoder.base, base_tok, self.tokenizer)
            if self.encoder.base is not None:
                remap_embeddings(self.encoder.base, base_tok, self.tokenizer)
            remap_embeddings(self.orig_model.model, base_tok, self.tokenizer)
        
        # Load weights
        if "models" in ckpt:
            for name, model in [("dec", self.decoder), ("enc", self.encoder)]:
                if name in ckpt["models"]:
                    state_dict = ckpt["models"][name]
                    if name == "dec" and "prompt_ids" in state_dict:
                        state_dict.pop("prompt_ids")
                    model.load_state_dict(state_dict, strict=False)
        
        # Set prompt and eval mode
        self.decoder.set_prompt(decoder_prompt, self.tokenizer)
        self.decoder.eval()
        self.encoder.eval()
    
    def analyze_token(self, text: str, position: int) -> Dict[str, Any]:
        """Analyze a single token position."""
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(self.device)
        seq_len = input_ids.shape[1]
        
        # Handle position
        if position < 0:
            position = seq_len + position
        position = max(0, min(position, seq_len - 1))
        
        with torch.no_grad():
            # Get hidden states
            outputs = self.orig_model.model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            A = hidden_states[self.layer + 1][:, position, :]
            
            # Generate explanation
            if self.decoder.config.use_kv_cache:
                gen = self.decoder.generate_soft_kv_cached(A, max_length=self.t_text, gumbel_tau=self.tau)
            else:
                gen = self.decoder.generate_soft(A, max_length=self.t_text, gumbel_tau=self.tau)
            
            # Reconstruct
            A_hat = self.encoder(gen.generated_text_embeddings)
            
            # Compute proper KL divergence using forward_with_replacement
            # Get predictions with original activation
            logits_orig = self.orig_model.forward_with_replacement(
                input_ids=input_ids,
                new_activation=A,
                layer_idx=self.layer,
                token_pos=position,
                no_grad=True
            ).logits
            
            # Get predictions with reconstructed activation
            logits_recon = self.orig_model.forward_with_replacement(
                input_ids=input_ids,
                new_activation=A_hat,
                layer_idx=self.layer,
                token_pos=position,
                no_grad=True
            ).logits
            
            # Extract logits at position
            logits_orig_at_pos = logits_orig[:, position, :]
            logits_recon_at_pos = logits_recon[:, position, :]
            
            # Compute KL divergence with numerical stability
            with torch.amp.autocast('cuda', enabled=False):
                logits_orig_f32 = logits_orig_at_pos.float()
                logits_recon_f32 = logits_recon_at_pos.float()
                
                # Subtract max for stability
                logits_orig_f32 = logits_orig_f32 - logits_orig_f32.max(dim=-1, keepdim=True)[0]
                logits_recon_f32 = logits_recon_f32 - logits_recon_f32.max(dim=-1, keepdim=True)[0]
                
                # KL(P_orig || P_recon)
                log_probs_orig = torch.log_softmax(logits_orig_f32, dim=-1)
                log_probs_recon = torch.log_softmax(logits_recon_f32, dim=-1)
                probs_orig = torch.exp(log_probs_orig)
                kl_div = (probs_orig * (log_probs_orig - log_probs_recon)).sum(dim=-1).mean().item()
            
            # Decode
            explanation = self.tokenizer.decode(gen.hard_token_ids[0], skip_special_tokens=True)
            token = self.tokenizer.decode([input_ids[0, position].item()])
        
        return {
            'position': position,
            'token': token,
            'explanation': explanation,
            'kl_divergence': kl_div
        }
    
    def show_all_tokens(self, text: str):
        """Analyze and display all tokens."""
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(self.device)
        seq_len = input_ids.shape[1]
        
        print(f"\nAnalyzing: \"{text}\"")
        print(f"Tokens: {seq_len}")
        print("-" * 80)
        
        # Get all hidden states at once
        with torch.no_grad():
            outputs = self.orig_model.model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states
        
        results = []
        for pos in range(seq_len):
            with torch.no_grad():
                A = hidden_states[self.layer + 1][:, pos, :]
                
                # Generate
                if self.decoder.config.use_kv_cache:
                    gen = self.decoder.generate_soft_kv_cached(A, max_length=self.t_text, gumbel_tau=self.tau)
                else:
                    gen = self.decoder.generate_soft(A, max_length=self.t_text, gumbel_tau=self.tau)
                
                # Reconstruct
                A_hat = self.encoder(gen.generated_text_embeddings)
                
                # Compute proper KL divergence
                logits_orig = self.orig_model.forward_with_replacement(
                    input_ids=input_ids,
                    new_activation=A,
                    layer_idx=self.layer,
                    token_pos=pos,
                    no_grad=True
                ).logits
                
                logits_recon = self.orig_model.forward_with_replacement(
                    input_ids=input_ids,
                    new_activation=A_hat,
                    layer_idx=self.layer,
                    token_pos=pos,
                    no_grad=True
                ).logits
                
                # Extract logits at position
                logits_orig_at_pos = logits_orig[:, pos, :]
                logits_recon_at_pos = logits_recon[:, pos, :]
                
                # Compute KL with numerical stability
                with torch.amp.autocast('cuda', enabled=False):
                    logits_orig_f32 = logits_orig_at_pos.float()
                    logits_recon_f32 = logits_recon_at_pos.float()
                    logits_orig_f32 = logits_orig_f32 - logits_orig_f32.max(dim=-1, keepdim=True)[0]
                    logits_recon_f32 = logits_recon_f32 - logits_recon_f32.max(dim=-1, keepdim=True)[0]
                    log_probs_orig = torch.log_softmax(logits_orig_f32, dim=-1)
                    log_probs_recon = torch.log_softmax(logits_recon_f32, dim=-1)
                    probs_orig = torch.exp(log_probs_orig)
                    kl_div = (probs_orig * (log_probs_orig - log_probs_recon)).sum(dim=-1).mean().item()
                
                # Decode
                explanation = self.tokenizer.decode(gen.hard_token_ids[0], skip_special_tokens=True)
                token = self.tokenizer.decode([input_ids[0, pos].item()])
                
                results.append({
                    'position': pos,
                    'token': token,
                    'explanation': explanation,
                    'kl_divergence': kl_div
                })
        
        self.display_results(results)
    
    def display_results(self, results: List[Dict[str, Any]]):
        """Display results in a nice format."""
        print(f"{'Pos':>4} {'Token':15} {'Explanation':45} {'KL':>8}")
        print("-" * 80)
        
        kl_values = [r['kl_divergence'] for r in results]
        min_kl = min(kl_values) if kl_values else 0
        max_kl = max(kl_values) if kl_values else 1
        
        for r in results:
            token = repr(r['token'])[:15]
            exp = r['explanation'][:45]
            if len(r['explanation']) > 45:
                exp = exp[:42] + "..."
            
            # Color code KL
            kl = r['kl_divergence']
            if max_kl > min_kl:
                normalized = (kl - min_kl) / (max_kl - min_kl)
                if normalized < 0.33:
                    kl_str = f"\033[92m{kl:8.3f}\033[0m"  # Green
                elif normalized < 0.67:
                    kl_str = f"\033[93m{kl:8.3f}\033[0m"  # Yellow
                else:
                    kl_str = f"\033[91m{kl:8.3f}\033[0m"  # Red
            else:
                kl_str = f"{kl:8.3f}"
            
            print(f"{r['position']:4d} {token:15} {exp:45} {kl_str}")
        
        # Summary
        avg_kl = np.mean(kl_values)
        print("-" * 80)
        print(f"Average KL: {avg_kl:.3f} | Min: {min_kl:.3f} | Max: {max_kl:.3f}")
    
    def compare_texts(self, texts: List[str]):
        """Compare multiple texts."""
        print("\nComparing texts:")
        print("=" * 80)
        
        summaries = []
        for i, text in enumerate(texts):
            print(f"\n{i+1}. \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
            
            # Quick analysis
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
            input_ids = inputs.input_ids.to(self.device)
            seq_len = input_ids.shape[1]
            
            with torch.no_grad():
                outputs = self.orig_model.model(input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states
            
            kl_values = []
            for pos in range(seq_len):
                A = hidden_states[self.layer + 1][:, pos, :]
                if self.decoder.config.use_kv_cache:
                    gen = self.decoder.generate_soft_kv_cached(A, max_length=self.t_text, gumbel_tau=self.tau)
                else:
                    gen = self.decoder.generate_soft(A, max_length=self.t_text, gumbel_tau=self.tau)
                A_hat = self.encoder(gen.generated_text_embeddings)
                
                # Compute proper KL
                logits_orig = self.orig_model.forward_with_replacement(
                    input_ids=input_ids, new_activation=A, 
                    layer_idx=self.layer, token_pos=pos, no_grad=True
                ).logits
                logits_recon = self.orig_model.forward_with_replacement(
                    input_ids=input_ids, new_activation=A_hat,
                    layer_idx=self.layer, token_pos=pos, no_grad=True  
                ).logits
                
                with torch.amp.autocast('cuda', enabled=False):
                    logits_o = logits_orig[:, pos, :].float()
                    logits_r = logits_recon[:, pos, :].float()
                    logits_o = logits_o - logits_o.max(dim=-1, keepdim=True)[0]
                    logits_r = logits_r - logits_r.max(dim=-1, keepdim=True)[0]
                    log_p_o = torch.log_softmax(logits_o, dim=-1)
                    log_p_r = torch.log_softmax(logits_r, dim=-1)
                    p_o = torch.exp(log_p_o)
                    kl = (p_o * (log_p_o - log_p_r)).sum(dim=-1).mean().item()
                kl_values.append(kl)
            
            avg_kl = np.mean(kl_values)
            print(f"   Tokens: {seq_len} | Avg KL: {avg_kl:.3f}")
            summaries.append((text, avg_kl, seq_len))
        
        # Ranking
        print("\nRanking by consistency (lowest to highest KL):")
        print("-" * 80)
        for i, (text, kl, tokens) in enumerate(sorted(summaries, key=lambda x: x[1])):
            print(f"{i+1}. KL={kl:.3f} | {tokens} tokens | \"{text[:50]}...\"")


# Convenience functions that work with the global analyzer
def tokens(text: str) -> List[str]:
    """Show how text is tokenized."""
    if _analyzer is None:
        print("No lens loaded! Use load_lens() first.")
        return []
    
    inputs = _analyzer.tokenizer(text, return_tensors="pt")
    token_ids = inputs.input_ids[0].tolist()
    tokens = [_analyzer.tokenizer.decode([tid]) for tid in token_ids]
    
    print(f"\nText: \"{text}\"")
    print(f"Tokens ({len(tokens)}):")
    for i, (token, tid) in enumerate(zip(tokens, token_ids)):
        print(f"  {i:3d}: {repr(token):15} (id: {tid})")
    
    return tokens


def explain(text: str, position: int = -1) -> str:
    """Get explanation for a specific token position."""
    if _analyzer is None:
        print("No lens loaded! Use load_lens() first.")
        return ""
    
    result = _analyzer.analyze_token(text, position)
    return result['explanation']


# Auto-import common functions when used interactively
__all__ = ['load_lens', 'analyze', 'analyze_last', 'compare', 'tokens', 'explain']

print("""
Consistency Lens Interactive Tools
==================================

Quick start:
  >>> lens = load_lens("path/to/checkpoint.pt")
  >>> analyze("Your text here")
  
Functions:
  • load_lens(checkpoint)     - Load a checkpoint
  • analyze(text)            - Analyze all tokens
  • analyze_last(text)       - Analyze only last token  
  • compare(text1, text2,...)- Compare multiple texts
  • tokens(text)             - Show tokenization
  • explain(text, pos)       - Get explanation for position

Examples:
  >>> analyze("The cat sat on the mat")
  >>> explain("Hello world", -1)  # Explain last token
  >>> compare("Hello", "Goodbye", "Greetings")
""")