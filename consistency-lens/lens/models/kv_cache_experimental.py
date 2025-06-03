"""Experimental KV cache implementation for memory testing."""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


@dataclass
class KVCacheExperimental:
    """Enhanced KV cache with better memory tracking."""
    keys: List[torch.Tensor]    # List of tensors per layer
    values: List[torch.Tensor]  # List of tensors per layer
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Calculate memory usage of the cache."""
        total_elements = 0
        for k, v in zip(self.keys, self.values):
            if k is not None:
                total_elements += k.numel() + v.numel()
        
        # Assume float32 (4 bytes per element)
        memory_bytes = total_elements * 4
        return {
            'elements': total_elements,
            'mb': memory_bytes / (1024 * 1024),
            'gb': memory_bytes / (1024 * 1024 * 1024),
        }
    
    def clear(self):
        """Clear the cache."""
        self.keys.clear()
        self.values.clear()


def compute_attention_with_kv_cache_experimental(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    kv_cache: Optional[KVCacheExperimental],
    layer_idx: int,
    attention_mask: Optional[torch.Tensor] = None,
    use_cache: bool = True,
) -> Tuple[torch.Tensor, Optional[KVCacheExperimental]]:
    """
    Enhanced attention computation with KV caching and memory tracking.
    
    This version includes better memory management and debugging capabilities.
    """
    
    if isinstance(layer, GPT2Attention):
        # Split QKV projection
        query, key, value = layer.c_attn(hidden_states).split(layer.split_size, dim=2)
        
        # Reshape for multi-head attention
        batch_size, seq_len = hidden_states.shape[:2]
        query = layer._split_heads(query, layer.num_heads, layer.head_dim)
        key = layer._split_heads(key, layer.num_heads, layer.head_dim)
        value = layer._split_heads(value, layer.num_heads, layer.head_dim)
        
        # Handle KV cache
        if kv_cache is not None and use_cache:
            # Extend cache lists if needed
            while len(kv_cache.keys) <= layer_idx:
                kv_cache.keys.append(None)
                kv_cache.values.append(None)
            
            # Get cached keys/values
            if kv_cache.keys[layer_idx] is not None:
                # Concatenate with cached values
                key = torch.cat([kv_cache.keys[layer_idx], key], dim=-2)
                value = torch.cat([kv_cache.values[layer_idx], value], dim=-2)
            
            # Update cache - store the full concatenated keys/values
            kv_cache.keys[layer_idx] = key.detach()
            kv_cache.values[layer_idx] = value.detach()
        
        # Compute attention scores
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Apply softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value.dtype)
        attn_weights = layer.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape back
        attn_output = layer._merge_heads(attn_output, layer.num_heads, layer.head_dim)
        
        # Final projection
        attn_output = layer.c_proj(attn_output)
        attn_output = layer.resid_dropout(attn_output)
        
        return attn_output, kv_cache
    else:
        # Fallback for non-GPT2 models
        outputs = layer(hidden_states, attention_mask=attention_mask, use_cache=use_cache)
        return outputs[0], None


def generate_with_kv_cache_experimental(
    model: nn.Module,
    input_embeds: torch.Tensor,
    max_length: int,
    temperature: float = 1.0,
    track_memory: bool = True,
) -> Dict:
    """
    Experimental generation with KV cache and detailed memory tracking.
    
    Returns dict with:
        - output_ids: Generated token IDs
        - memory_stats: Memory usage at each step
        - kv_cache: Final KV cache state
    """
    
    device = input_embeds.device
    batch_size = input_embeds.shape[0]
    
    # Initialize KV cache
    kv_cache = KVCacheExperimental(keys=[], values=[])
    
    # Track memory stats
    memory_stats = []
    
    # Start with input embeddings
    current_embeds = input_embeds
    output_ids = []
    
    # Get model components
    if hasattr(model, 'transformer'):
        transformer = model.transformer
        lm_head = model.lm_head
        layers = transformer.h
    else:
        raise ValueError("Model structure not supported")
    
    for step in range(max_length):
        # Track memory before step
        if track_memory and torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated() / (1024**3)  # GB
        
        # Forward through embedding layers
        hidden_states = current_embeds
        
        # Add positional embeddings
        position_ids = torch.arange(
            step, step + hidden_states.shape[1], 
            dtype=torch.long, device=device
        ).unsqueeze(0)
        position_embeds = transformer.wpe(position_ids)
        hidden_states = hidden_states + position_embeds
        
        # Forward through transformer layers with KV cache
        for i, layer in enumerate(layers):
            # Pre-norm
            residual = hidden_states
            hidden_states = layer.ln_1(hidden_states)
            
            # Attention with KV cache
            attn_output, kv_cache = compute_attention_with_kv_cache_experimental(
                layer.attn,
                hidden_states,
                kv_cache,
                layer_idx=i,
                use_cache=True
            )
            
            # Add residual
            hidden_states = residual + attn_output
            
            # MLP
            residual = hidden_states
            hidden_states = layer.ln_2(hidden_states)
            feed_forward_hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + feed_forward_hidden_states
        
        # Final layer norm
        hidden_states = transformer.ln_f(hidden_states)
        
        # Get logits for the last position
        last_hidden = hidden_states[:, -1:, :]
        logits = lm_head(last_hidden)
        
        # Sample next token
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs.squeeze(1), num_samples=1)
        output_ids.append(next_token)
        
        # Get embeddings for next token
        current_embeds = transformer.wte(next_token)
        
        # Track memory after step
        if track_memory and torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_after = torch.cuda.memory_allocated() / (1024**3)  # GB
            cache_stats = kv_cache.get_memory_usage()
            
            memory_stats.append({
                'step': step,
                'memory_before_gb': memory_before,
                'memory_after_gb': memory_after,
                'memory_delta_gb': memory_after - memory_before,
                'cache_size_gb': cache_stats['gb'],
                'cache_elements': cache_stats['elements'],
            })
    
    # Stack output IDs
    output_ids = torch.cat(output_ids, dim=1)
    
    return {
        'output_ids': output_ids,
        'memory_stats': memory_stats,
        'kv_cache': kv_cache,
        'final_cache_memory_gb': kv_cache.get_memory_usage()['gb'],
    }