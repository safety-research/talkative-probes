"""Flash Attention KV cache implementation for efficient autoregressive generation."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch
import torch.nn as nn

# Check if flash_attn is available
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    flash_attn_func = None
    flash_attn_varlen_func = None


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


class GPT2AttentionFlash(nn.Module):
    """GPT2 attention using Flash Attention with KV caching support.
    
    This maintains the same interface as GPT2Attention but uses Flash Attention
    for the actual computation.
    """
    
    def __init__(self, gpt2_attn):
        super().__init__()
        self.attn = gpt2_attn
        self.num_heads = gpt2_attn.num_heads
        self.head_dim = gpt2_attn.head_dim
        self.embed_dim = self.num_heads * self.head_dim
        
        # Copy relevant attributes
        self.c_attn = gpt2_attn.c_attn
        self.c_proj = gpt2_attn.c_proj
        self.attn_dropout = gpt2_attn.attn_dropout
        self.resid_dropout = gpt2_attn.resid_dropout
        
        # Flash attention parameters
        self.dropout_p = gpt2_attn.attn_dropout.p if hasattr(gpt2_attn.attn_dropout, 'p') else 0.0
        self.causal = True  # GPT-2 uses causal attention
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[FlashKVCache] = None,
        layer_idx: int = 0,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[FlashKVCache]]:
        """Forward pass using Flash Attention.
        
        Args:
            hidden_states: Input hidden states (batch, seq, hidden)
            kv_cache: Optional KV cache to use/update
            layer_idx: Which layer this is (for indexing into cache)
            use_cache: Whether to use/update the cache
            
        Returns:
            attn_output: Attention output
            kv_cache: Updated cache (if use_cache=True)
        """
        if not FLASH_AVAILABLE:
            raise RuntimeError("Flash Attention not available. Please install flash-attn package.")
        
        batch_size, seq_length, _ = hidden_states.size()
        
        # Compute Q, K, V
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.embed_dim, dim=2)
        
        # Reshape for Flash Attention: (batch, seq, heads, head_dim)
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        if use_cache and kv_cache is not None:
            if kv_cache.is_initialized(layer_idx):
                # Append new K,V to cache
                key = torch.cat([kv_cache.keys[layer_idx], key], dim=1)
                value = torch.cat([kv_cache.values[layer_idx], value], dim=1)
                
                # Update cache
                kv_cache.keys[layer_idx] = key
                kv_cache.values[layer_idx] = value
            else:
                # Initialize cache for this layer
                if layer_idx >= len(kv_cache.keys):
                    kv_cache.keys.append(key)
                    kv_cache.values.append(value)
                else:
                    kv_cache.keys[layer_idx] = key
                    kv_cache.values[layer_idx] = value
        
        # Run Flash Attention
        # Note: Flash attention expects (batch, seq, heads, head_dim) format
        # Flash Attention only supports fp16 and bf16
        orig_dtype = query.dtype
        if orig_dtype not in [torch.float16, torch.bfloat16]:
            # Convert to bfloat16 for computation
            query = query.to(torch.bfloat16)
            key = key.to(torch.bfloat16)
            value = value.to(torch.bfloat16)
        
        attn_output = flash_attn_func(
            query,
            key,
            value,
            dropout_p=self.dropout_p if self.training else 0.0,
            causal=self.causal,
            window_size=(-1, -1),  # No sliding window
            alibi_slopes=None,
            deterministic=False,
        )
        
        # Convert back to original dtype if needed
        if orig_dtype not in [torch.float16, torch.bfloat16]:
            attn_output = attn_output.to(orig_dtype)
        
        # Reshape back to (batch, seq, hidden)
        attn_output = attn_output.view(batch_size, seq_length, self.embed_dim)
        
        # Output projection
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output, kv_cache


class GPT2BlockFlash(nn.Module):
    """GPT2 block using Flash Attention."""
    
    def __init__(self, gpt2_block):
        super().__init__()
        self.ln_1 = gpt2_block.ln_1
        self.ln_2 = gpt2_block.ln_2
        self.mlp = gpt2_block.mlp
        
        # Replace attention with Flash version
        self.attn = GPT2AttentionFlash(gpt2_block.attn)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[FlashKVCache] = None,
        layer_idx: int = 0,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[FlashKVCache]]:
        """Forward pass through the block."""
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        # Self-attention with Flash
        attn_output, kv_cache = self.attn(
            hidden_states,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output
        
        # MLP
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_output = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_output
        
        return hidden_states, kv_cache


def compute_with_flash_kv_cache(
    transformer,
    input_embeds: torch.Tensor,
    kv_cache: Optional[FlashKVCache] = None,
    position_offset: int = 0,
    use_cache: bool = True,
) -> Tuple[torch.Tensor, Optional[FlashKVCache]]:
    """Forward pass through transformer with Flash Attention KV caching.
    
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
    if not FLASH_AVAILABLE:
        raise RuntimeError("Flash Attention not available. Please install flash-attn package.")
    
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
    
    # Process through transformer layers with Flash Attention
    for layer_idx, layer in enumerate(transformer.h):
        # Create Flash-enabled block wrapper
        flash_block = GPT2BlockFlash(layer)
        hidden_states, kv_cache = flash_block(
            hidden_states,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            use_cache=use_cache,
        )
    
    # Final layer norm
    hidden_states = transformer.ln_f(hidden_states)
    
    return hidden_states, kv_cache