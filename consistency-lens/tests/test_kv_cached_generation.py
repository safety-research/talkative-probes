"""Test implementation of differentiable KV-cached generation for GPT-2."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPT2Model
import numpy as np
from lens.models.decoder import Decoder, DecoderConfig, Generated
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class KVCache:
    """Stores key-value pairs for each layer with gradients."""
    keys: List[torch.Tensor] = None  # One tensor per layer
    values: List[torch.Tensor] = None  # One tensor per layer
    
    def __post_init__(self):
        if self.keys is None:
            self.keys = []
        if self.values is None:
            self.values = []


def gpt2_attention_with_cache(
    layer_module,
    hidden_states: torch.Tensor,
    kv_cache: Optional[KVCache] = None,
    layer_idx: int = 0,
    use_cache: bool = True
) -> Tuple[torch.Tensor, Optional[KVCache]]:
    """
    Compute GPT-2 attention with KV caching support.
    
    This reimplements GPT-2's attention to support incremental KV computation.
    """
    batch_size, seq_length, hidden_size = hidden_states.size()
    
    # Get attention module
    attn = layer_module.attn
    
    # For GPT-2, c_attn projects to 3 * hidden_size (Q, K, V)
    qkv = attn.c_attn(hidden_states)
    query, key, value = qkv.split(hidden_size, dim=2)
    
    # Reshape for multi-head attention
    num_heads = attn.num_heads
    head_dim = hidden_size // num_heads
    
    def split_heads(tensor):
        """Split heads: (batch, seq, hidden) -> (batch, heads, seq, head_dim)"""
        return tensor.view(batch_size, seq_length, num_heads, head_dim).permute(0, 2, 1, 3)
    
    query = split_heads(query)  # (B, heads, seq, head_dim)
    
    if kv_cache is not None and layer_idx < len(kv_cache.keys):
        # We have cached K,V - only compute for the new position
        if seq_length == 1:
            # Incremental step: just use the new K,V
            key = split_heads(key)
            value = split_heads(value)
        else:
            # First step with full sequence - cache everything
            key = split_heads(key)
            value = split_heads(value)
            
        # Update cache
        if layer_idx < len(kv_cache.keys):
            # Append to existing cache
            kv_cache.keys[layer_idx] = torch.cat([kv_cache.keys[layer_idx], key], dim=2)
            kv_cache.values[layer_idx] = torch.cat([kv_cache.values[layer_idx], value], dim=2)
        else:
            # Initialize cache for this layer
            kv_cache.keys.append(key)
            kv_cache.values.append(value)
            
        # Use full cached K,V for attention
        key_for_attn = kv_cache.keys[layer_idx]
        value_for_attn = kv_cache.values[layer_idx]
    else:
        # No cache - compute normally
        key = split_heads(key)
        value = split_heads(value)
        key_for_attn = key
        value_for_attn = value
        
        if use_cache and kv_cache is not None:
            # Initialize cache
            if layer_idx >= len(kv_cache.keys):
                kv_cache.keys.append(key)
                kv_cache.values.append(value)
    
    # Compute attention scores
    attn_weights = torch.matmul(query, key_for_attn.transpose(-1, -2))
    attn_weights = attn_weights / torch.sqrt(torch.tensor(head_dim, dtype=attn_weights.dtype))
    
    # Apply causal mask
    if seq_length > 1:
        # Create causal mask
        causal_mask = torch.tril(torch.ones((seq_length, key_for_attn.size(2)), device=hidden_states.device))
        causal_mask = causal_mask.view(1, 1, seq_length, key_for_attn.size(2))
        attn_weights = attn_weights.masked_fill(causal_mask == 0, float('-inf'))
    
    # Softmax
    attn_probs = nn.functional.softmax(attn_weights, dim=-1)
    attn_probs = attn.attn_dropout(attn_probs)
    
    # Apply attention to values
    attn_output = torch.matmul(attn_probs, value_for_attn)
    
    # Merge heads
    attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
    attn_output = attn_output.view(batch_size, seq_length, hidden_size)
    
    # Final projection
    attn_output = attn.c_proj(attn_output)
    attn_output = attn.resid_dropout(attn_output)
    
    return attn_output, kv_cache


def gpt2_layer_with_cache(
    layer_module,
    hidden_states: torch.Tensor,
    kv_cache: Optional[KVCache] = None,
    layer_idx: int = 0,
    use_cache: bool = True
) -> Tuple[torch.Tensor, Optional[KVCache]]:
    """Process one GPT-2 layer with KV caching."""
    residual = hidden_states
    hidden_states = layer_module.ln_1(hidden_states)
    
    # Attention with cache
    attn_output, kv_cache = gpt2_attention_with_cache(
        layer_module, hidden_states, kv_cache, layer_idx, use_cache
    )
    
    hidden_states = attn_output + residual
    
    # MLP
    residual = hidden_states
    hidden_states = layer_module.ln_2(hidden_states)
    feed_forward_hidden_states = layer_module.mlp(hidden_states)
    hidden_states = residual + feed_forward_hidden_states
    
    return hidden_states, kv_cache


def generate_soft_with_kv_cache(
    decoder: Decoder,
    activation_input: torch.Tensor,
    max_length: int,
    gumbel_tau: float,
    use_projection: bool = True,
) -> Generated:
    """
    Generate using KV caching to avoid redundant computation.
    
    This should produce identical results to generate_soft but with
    O(n) instead of O(n²) attention computation per step.
    """
    # Setup (same as generate_soft)
    main_base = decoder.base
    main_out = decoder.out
    
    activation_input = activation_input.to(decoder.proj.weight.dtype)
    B, d_model = activation_input.shape
    device = activation_input.device
    
    # Get embedding tables
    input_emb_table = main_base.get_input_embeddings().weight
    output_emb_table = main_base.get_output_embeddings().weight
    embeddings_tied = (input_emb_table.data_ptr() == output_emb_table.data_ptr())
    
    # Prepare initial sequence
    parts = []
    if decoder.prompt_left_emb is not None:
        parts.append(decoder.prompt_left_emb.expand(B, -1, -1))
    if decoder.prompt_right_emb is not None:
        parts.append(decoder.prompt_right_emb.expand(B, -1, -1))
    if use_projection:
        a_proj = decoder.proj(activation_input).unsqueeze(1)
    else:
        a_proj = activation_input.unsqueeze(1)
    parts.append(a_proj)
    initial_seq_embs = torch.cat(parts, dim=1)
    
    # Initialize KV cache
    kv_cache = KVCache()
    
    # Process initial sequence through transformer with caching
    hidden_states = initial_seq_embs
    hidden_states = main_base.transformer.wte.dropout(hidden_states)
    
    # Get position embeddings for initial sequence
    position_ids = torch.arange(0, hidden_states.size(1), dtype=torch.long, device=device)
    position_embeds = main_base.transformer.wpe(position_ids)
    hidden_states = hidden_states + position_embeds
    
    # Process through layers with caching
    for i, layer in enumerate(main_base.transformer.h):
        hidden_states, kv_cache = gpt2_layer_with_cache(
            layer, hidden_states, kv_cache, i, use_cache=True
        )
    
    # Final layer norm
    hidden_states = main_base.transformer.ln_f(hidden_states)
    
    # Get logits for last position
    logits_t = main_out(hidden_states[:, -1])
    
    # Initialize generation storage
    logits_list = []
    hard_ids_list = []
    output_embs_list = []
    
    # Current position for position embeddings
    current_pos = initial_seq_embs.size(1)
    
    # Generate tokens one by one
    for step in range(max_length):
        if step > 0:
            # Process only the new token through transformer
            # hidden_states is now just the new token embedding
            hidden_states = emb_t_input.unsqueeze(1)  # (B, 1, d_model)
            
            # Add position embedding for the new position
            position_ids = torch.tensor([current_pos], dtype=torch.long, device=device)
            position_embeds = main_base.transformer.wpe(position_ids)
            hidden_states = hidden_states + position_embeds
            
            # Process through layers using cached K,V
            for i, layer in enumerate(main_base.transformer.h):
                hidden_states, kv_cache = gpt2_layer_with_cache(
                    layer, hidden_states, kv_cache, i, use_cache=True
                )
            
            # Final layer norm
            hidden_states = main_base.transformer.ln_f(hidden_states)
            
            # Get logits
            logits_t = main_out(hidden_states[:, -1])
            
            current_pos += 1
        
        # Gumbel-Softmax sampling (same as original)
        logits_t_scaled = logits_t / 1.0  # T_sampling = 1.0
        ste_token_dist = torch.nn.functional.gumbel_softmax(
            logits_t_scaled,
            tau=gumbel_tau,
            hard=True
        )
        
        # Get embeddings
        emb_t_input = ste_token_dist @ input_emb_table
        if embeddings_tied:
            emb_t_output = emb_t_input
        else:
            emb_t_output = ste_token_dist @ output_emb_table
        
        # Store outputs
        logits_list.append(logits_t)
        output_embs_list.append(emb_t_output)
        hard_ids_list.append(ste_token_dist.argmax(dim=-1))
    
    # Stack outputs
    logits_seq = torch.stack(logits_list, dim=1)
    hard_ids = torch.stack(hard_ids_list, dim=1)
    text_embs = torch.stack(output_embs_list, dim=1)
    
    return Generated(text_embs, logits_seq, hard_ids)


def test_kv_cache_equivalence():
    """Test that KV-cached generation produces identical results."""
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Setup
    model_name = "gpt2"  # Use full GPT-2 for realistic test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create decoder
    config = DecoderConfig(
        model_name=model_name,
        n_prompt_tokens=0,
        base_model=False,
        projection_layer=True,
        output_head=True,
        embedding_head=False,
        eye_init=True,
        trainable_prompts=True,
        use_checkpointing=False
    )
    
    decoder = Decoder(config).to(device)
    
    # Setup tokenizer and prompt
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    decoder.set_prompt("The meaning of <embed> is:", tokenizer)
    
    # Create test input
    batch_size = 2
    d_model = decoder.base.config.hidden_size
    test_activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    
    # Test parameters
    max_length = 8
    gumbel_tau = 1.0
    
    print("Testing KV-cached generation...")
    
    # Generate with original method
    torch.manual_seed(123)
    gen_original = decoder.generate_soft(
        test_activation.clone(),
        max_length=max_length,
        gumbel_tau=gumbel_tau
    )
    
    # Generate with KV caching
    torch.manual_seed(123)
    gen_cached = generate_soft_with_kv_cache(
        decoder,
        test_activation.clone(),
        max_length=max_length,
        gumbel_tau=gumbel_tau
    )
    
    # Check equivalence
    print(f"\nGenerated embeddings match: {torch.allclose(gen_original.generated_text_embeddings, gen_cached.generated_text_embeddings, rtol=1e-4, atol=1e-5)}")
    print(f"Raw logits match: {torch.allclose(gen_original.raw_lm_logits, gen_cached.raw_lm_logits, rtol=1e-4, atol=1e-5)}")
    print(f"Hard token IDs match: {torch.equal(gen_original.hard_token_ids, gen_cached.hard_token_ids)}")
    
    # Test gradients
    print("\nTesting gradient flow...")
    
    # Create loss
    target = torch.randn_like(gen_original.generated_text_embeddings)
    loss_original = nn.functional.mse_loss(gen_original.generated_text_embeddings, target)
    loss_cached = nn.functional.mse_loss(gen_cached.generated_text_embeddings, target)
    
    # Compute gradients
    grad_original = torch.autograd.grad(loss_original, test_activation, retain_graph=True)[0]
    grad_cached = torch.autograd.grad(loss_cached, test_activation, retain_graph=True)[0]
    
    print(f"Losses match: {torch.allclose(loss_original, loss_cached, rtol=1e-4, atol=1e-5)}")
    print(f"Gradients match: {torch.allclose(grad_original, grad_cached, rtol=1e-4, atol=1e-5)}")
    
    # Print any differences
    if not torch.allclose(gen_original.generated_text_embeddings, gen_cached.generated_text_embeddings, rtol=1e-4, atol=1e-5):
        diff = torch.abs(gen_original.generated_text_embeddings - gen_cached.generated_text_embeddings)
        print(f"\nMax embedding difference: {diff.max().item()}")
        print(f"Mean embedding difference: {diff.mean().item()}")
    
    # Measure memory usage
    if torch.cuda.is_available():
        print("\nMemory usage comparison:")
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Original method
        start_mem = torch.cuda.memory_allocated()
        gen1 = decoder.generate_soft(test_activation, max_length=16, gumbel_tau=1.0)
        torch.cuda.synchronize()
        original_mem = torch.cuda.memory_allocated() - start_mem
        
        # Clear for fair comparison
        del gen1
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Cached method
        start_mem = torch.cuda.memory_allocated()
        gen2 = generate_soft_with_kv_cache(decoder, test_activation, max_length=16, gumbel_tau=1.0)
        torch.cuda.synchronize()
        cached_mem = torch.cuda.memory_allocated() - start_mem
        
        print(f"Original method: {original_mem / 1024 / 1024:.1f} MB")
        print(f"Cached method: {cached_mem / 1024 / 1024:.1f} MB")
        print(f"Memory saved: {(original_mem - cached_mem) / 1024 / 1024:.1f} MB ({(original_mem - cached_mem) / original_mem * 100:.1f}%)")
    
    print("\n✓ KV cache test completed!")


if __name__ == "__main__":
    test_kv_cache_equivalence()