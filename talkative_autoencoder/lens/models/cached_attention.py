"""Attention modules with KV-cache support for differentiable generation."""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math

from .kv_cache import KVCache


class CachedMultiHeadAttention(nn.Module):
    """Multi-head attention with KV-cache support.
    
    This is a simplified attention implementation that supports caching
    for autoregressive generation while maintaining gradient flow.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_idx: int = 0,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with optional KV-caching.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            layer_idx: Which layer this is (for cache indexing)
            kv_cache: Optional KV cache to use/update
            use_cache: Whether to return key-value pairs
            
        Returns:
            Tuple of (output, (keys, values)) where keys/values are None if not caching
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, D]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is [B, H, S, D]
        
        # Use cache if provided
        if kv_cache is not None:
            # For cached generation, we typically only have new tokens in hidden_states
            # So seq_len might be 1 (for single token generation)
            if seq_len == 1:
                # Single token generation - use cached KV
                cached_kv = kv_cache.get(layer_idx)
                if cached_kv is not None:
                    cached_k, cached_v = cached_kv
                    # Append new K, V to cache
                    k_full, v_full = kv_cache.update(layer_idx, k, v)
                    k, v = k_full, v_full
                else:
                    # First token - just update cache
                    k, v = kv_cache.update(layer_idx, k, v)
            else:
                # Multi-token input (e.g., initial prompt) - replace cache
                k, v = kv_cache.update(layer_idx, k, v)
        
        # Scaled dot-product attention
        # q: [B, H, S_q, D], k: [B, H, S_k, D]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, S_q, S_k]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for heads dimension
            if attention_mask.dim() == 2:  # [B, S]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
            elif attention_mask.dim() == 3:  # [B, S_q, S_k]
                attention_mask = attention_mask.unsqueeze(1)  # [B, 1, S_q, S_k]
            scores = scores + attention_mask
        
        # Softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # [B, H, S_q, S_k] x [B, H, S_k, D] -> [B, H, S_q, D]
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, S_q, H, D]
        attn_output = attn_output.reshape(batch_size, -1, self.hidden_size)  # [B, S_q, hidden]
        attn_output = self.out_proj(attn_output)
        
        # Return keys/values if caching
        outputs = (attn_output,)
        if use_cache:
            outputs = outputs + ((k, v),)
            
        return outputs


def monkey_patch_model_for_caching(model: nn.Module, model_type: str = "gpt2"):
    """Monkey-patch a HuggingFace model to support KV-caching with gradients.
    
    This function modifies the model's attention modules to use our cached
    attention implementation. This is a simplified example - real implementation
    would need to handle different model architectures.
    
    Args:
        model: HuggingFace model to patch
        model_type: Type of model (gpt2, llama, etc.)
    """
    raise NotImplementedError(
        "Model patching is complex and model-specific. "
        "For production use, consider implementing a custom forward pass "
        "or using HuggingFace's native cache support with gradient modifications."
    )


def create_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a causal attention mask.
    
    Args:
        seq_len: Sequence length
        device: Device to create mask on
        dtype: Data type for mask
        
    Returns:
        Causal mask of shape [seq_len, seq_len]
    """
    mask = torch.ones(seq_len, seq_len, device=device, dtype=dtype)
    mask = torch.tril(mask)
    # Convert to attention mask format (0 = attend, -inf = don't attend)
    mask = (1.0 - mask) * torch.finfo(dtype).min
    return mask