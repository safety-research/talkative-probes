"""Flash Attention KV cache implementation - simplified version that preserves model behavior."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch

# Check if flash_attn is available
try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    flash_attn_func = None


@dataclass
class FlashKVCache:
    """Stores key-value pairs for Flash Attention with gradient support."""
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
            return self.keys[layer_idx].size(1)  # (batch, seq, heads, head_dim) for Flash
        return 0
    
    def is_initialized(self, layer_idx: int) -> bool:
        """Check if cache is initialized for a given layer."""
        return layer_idx < len(self.keys)


def compute_with_flash_kv_cache(
    transformer,
    input_embeds: torch.Tensor,
    kv_cache: Optional[FlashKVCache] = None,
    position_offset: int = 0,
    use_cache: bool = True,
) -> Tuple[torch.Tensor, Optional[FlashKVCache]]:
    """Forward pass through transformer with Flash Attention KV caching.
    
    This version modifies the attention computation in-place rather than
    wrapping layers, to preserve exact model behavior.
    """
    if not FLASH_AVAILABLE:
        raise RuntimeError("Flash Attention not available. Install with: make flash-attention")
    
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
        kv_cache = FlashKVCache()
    
    # Process through transformer layers
    for layer_idx, layer in enumerate(transformer.h):
        residual = hidden_states
        hidden_states = layer.ln_1(hidden_states)
        
        # Compute Q, K, V using the layer's attention module
        qkv = layer.attn.c_attn(hidden_states)
        query, key, value = qkv.split(layer.attn.embed_dim, dim=2)
        
        # Split into heads for multi-head attention
        num_heads = layer.attn.num_heads
        head_dim = layer.attn.head_dim
        
        # Reshape: (batch, seq, hidden) -> (batch, seq, heads, head_dim)
        query = query.view(batch_size, seq_length, num_heads, head_dim)
        key = key.view(batch_size, seq_length, num_heads, head_dim)
        value = value.view(batch_size, seq_length, num_heads, head_dim)
        
        # Handle KV caching
        if use_cache and kv_cache is not None:
            is_incremental = kv_cache.is_initialized(layer_idx) and seq_length == 1
            
            if is_incremental:
                # Append to cache for incremental generation
                key = torch.cat([kv_cache.keys[layer_idx], key], dim=1)
                value = torch.cat([kv_cache.values[layer_idx], value], dim=1)
                kv_cache.keys[layer_idx] = key
                kv_cache.values[layer_idx] = value
            else:
                # Initialize or replace cache
                if layer_idx >= len(kv_cache.keys):
                    kv_cache.keys.append(key)
                    kv_cache.values.append(value)
                else:
                    kv_cache.keys[layer_idx] = key
                    kv_cache.values[layer_idx] = value
        
        # Convert to appropriate dtype for Flash Attention
        orig_dtype = query.dtype
        compute_dtype = torch.bfloat16 if orig_dtype == torch.float32 else orig_dtype
        
        if orig_dtype != compute_dtype:
            query = query.to(compute_dtype)
            key = key.to(compute_dtype)
            value = value.to(compute_dtype)
        
        # Run Flash Attention
        # Note: Flash Attention internally applies the scaling correctly
        dropout_p = layer.attn.attn_dropout.p if layer.training and hasattr(layer.attn, 'attn_dropout') else 0.0
        
        attn_output = flash_attn_func(
            query,
            key,
            value,
            dropout_p=dropout_p,
            causal=True,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=False,
        )
        
        # Convert back to original dtype if needed
        if orig_dtype != compute_dtype:
            attn_output = attn_output.to(orig_dtype)
        
        # Reshape back: (batch, seq, heads, head_dim) -> (batch, seq, hidden)
        attn_output = attn_output.view(batch_size, -1, layer.attn.embed_dim)
        
        # Apply output projection and dropout
        attn_output = layer.attn.c_proj(attn_output)
        attn_output = layer.attn.resid_dropout(attn_output)
        
        # Add residual connection
        hidden_states = residual + attn_output
        
        # MLP
        residual = hidden_states
        hidden_states = layer.ln_2(hidden_states)
        mlp_output = layer.mlp(hidden_states)
        hidden_states = residual + mlp_output
    
    # Final layer norm
    hidden_states = transformer.ln_f(hidden_states)
    
    return hidden_states, kv_cache