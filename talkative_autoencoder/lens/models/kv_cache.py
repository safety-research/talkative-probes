"""KV cache implementation for efficient autoregressive generation."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch
import torch.nn as nn


@dataclass
class KVCache:
    """Stores key-value pairs for each transformer layer with gradients."""
    keys: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    
    def clear(self):
        """Clear the cache."""
        self.keys.clear()
        self.values.clear()
    
    def __len__(self):
        """Return number of cached layers."""
        return len(self.keys)
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get the sequence length of cached keys/values."""
        if layer_idx < len(self.keys):
            return self.keys[layer_idx].size(2)  # (batch, heads, seq, head_dim)
        return 0
    
    def is_initialized(self, layer_idx: int) -> bool:
        """Check if cache is initialized for a given layer."""
        return layer_idx < len(self.keys)


class GPT2AttentionWithCache(nn.Module):
    """Wrapper for GPT2 attention that supports KV caching."""
    
    def __init__(self, gpt2_layer):
        super().__init__()
        self.layer = gpt2_layer
        self.attn = gpt2_layer.attn
        self.ln_1 = gpt2_layer.ln_1
        self.ln_2 = gpt2_layer.ln_2
        self.mlp = gpt2_layer.mlp
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        layer_idx: int = 0,
        use_cache: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """Forward pass with optional KV caching.
        
        Args:
            hidden_states: Input hidden states (batch, seq, hidden)
            kv_cache: Optional KV cache to use/update
            layer_idx: Which layer this is (for indexing into cache)
            use_cache: Whether to use/update the cache
            
        Returns:
            hidden_states: Output hidden states
            kv_cache: Updated cache (if use_cache=True)
        """
        batch_size, seq_length, hidden_size = hidden_states.size()
        
        # Pre-norm
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        # Self-attention with potential caching
        if use_cache and kv_cache is not None and kv_cache.is_initialized(layer_idx) and seq_length == 1:
            # Incremental generation - use cached K,V
            hidden_states = self._incremental_attention(
                hidden_states, kv_cache, layer_idx, hidden_size, **kwargs
            )
        else:
            # Full attention computation
            hidden_states = self._full_attention(
                hidden_states, kv_cache, layer_idx, use_cache, **kwargs
            )
        
        # Add residual
        hidden_states = hidden_states + residual
        
        # MLP
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        
        return hidden_states, kv_cache
    
    def _incremental_attention(
        self,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        layer_idx: int,
        hidden_size: int,
        **kwargs
    ) -> torch.Tensor:
        """Compute attention for a single new token using cached K,V."""
        batch_size = hidden_states.size(0)
        
        # Compute Q,K,V for the new token
        qkv = self.attn.c_attn(hidden_states)  # (B, 1, 3*hidden)
        query, key, value = qkv.split(hidden_size, dim=2)
        
        # Reshape for multi-head attention
        num_heads = self.attn.num_heads
        head_dim = hidden_size // num_heads
        
        def split_heads(tensor):
            """(batch, seq, hidden) -> (batch, heads, seq, head_dim)"""
            return tensor.view(batch_size, -1, num_heads, head_dim).permute(0, 2, 1, 3)
        
        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        
        # Append new K,V to cache
        kv_cache.keys[layer_idx] = torch.cat([kv_cache.keys[layer_idx], key], dim=2)
        kv_cache.values[layer_idx] = torch.cat([kv_cache.values[layer_idx], value], dim=2)
        
        # Attention using all cached K,V
        attn_weights = torch.matmul(query, kv_cache.keys[layer_idx].transpose(-1, -2))
        attn_weights = attn_weights / torch.sqrt(torch.tensor(head_dim, dtype=attn_weights.dtype))
        
        # No causal mask needed for single token attending to all previous
        attn_probs = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn.attn_dropout(attn_probs)
        
        attn_output = torch.matmul(attn_probs, kv_cache.values[layer_idx])
        
        # Merge heads
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, 1, hidden_size)
        
        # Output projection
        attn_output = self.attn.c_proj(attn_output)
        attn_output = self.attn.resid_dropout(attn_output)
        
        return attn_output
    
    def _full_attention(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_idx: int,
        use_cache: bool,
        **kwargs
    ) -> torch.Tensor:
        """Compute full attention and optionally cache K,V."""
        batch_size, seq_length, hidden_size = hidden_states.size()
        
        # Standard attention computation with position info
        attn_output = self.attn(hidden_states, **kwargs)[0]
        
        if use_cache and kv_cache is not None:
            # Extract and cache K,V
            qkv = self.attn.c_attn(hidden_states)
            _, key, value = qkv.split(hidden_size, dim=2)
            
            # Reshape for multi-head attention
            num_heads = self.attn.num_heads
            head_dim = hidden_size // num_heads
            
            def split_heads(tensor):
                return tensor.view(batch_size, -1, num_heads, head_dim).permute(0, 2, 1, 3)
            
            key = split_heads(key)
            value = split_heads(value)
            
            # Store in cache
            if layer_idx >= len(kv_cache.keys):
                kv_cache.keys.append(key)
                kv_cache.values.append(value)
            else:
                kv_cache.keys[layer_idx] = key
                kv_cache.values[layer_idx] = value
        
        return attn_output


def compute_with_kv_cache(
    transformer,
    input_embeds: torch.Tensor,
    kv_cache: Optional[KVCache] = None,
    position_offset: int = 0,
    use_cache: bool = True
) -> Tuple[torch.Tensor, Optional[KVCache]]:
    """Forward pass through transformer with KV caching.
    
    Args:
        transformer: The transformer model (e.g., GPT2Model)
        input_embeds: Input embeddings (batch, seq, hidden)
        kv_cache: Optional KV cache to use/update
        position_offset: Starting position for position embeddings
        use_cache: Whether to use/update cache
        
    Returns:
        hidden_states: Output hidden states
        kv_cache: Updated cache
    """
    device = input_embeds.device
    batch_size, seq_length, _ = input_embeds.shape
    
    # Position embeddings
    position_ids = torch.arange(
        position_offset, position_offset + seq_length, 
        dtype=torch.long, device=device
    )
    position_embeds = transformer.wpe(position_ids)
    hidden_states = transformer.drop(input_embeds + position_embeds)
    
    # Initialize cache if needed
    if use_cache and kv_cache is None:
        kv_cache = KVCache()
    
    # Process through transformer layers
    for layer_idx, layer in enumerate(transformer.h):
        # Check if this is an incremental step
        is_incremental = (
            use_cache and 
            kv_cache is not None and 
            kv_cache.is_initialized(layer_idx) and 
            seq_length == 1
        )
        
        if is_incremental:
            # Use our wrapper for incremental attention
            wrapper = GPT2AttentionWithCache(layer)
            hidden_states, kv_cache = wrapper(
                hidden_states, kv_cache, layer_idx, use_cache
            )
        else:
            # Standard processing
            wrapper = GPT2AttentionWithCache(layer)
            hidden_states, kv_cache = wrapper(
                hidden_states, kv_cache, layer_idx, use_cache
            )
    
    # Final layer norm
    hidden_states = transformer.ln_f(hidden_states)
    
    return hidden_states, kv_cache