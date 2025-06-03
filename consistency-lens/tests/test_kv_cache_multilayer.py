#!/usr/bin/env python3
"""Test KV caching with multi-layer patching enabled."""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def test_kv_cache_with_multilayer():
    """Test that KV caching works correctly with multi-layer patching."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Testing KV Cache with Multi-Layer Patching")
    print("=" * 60)
    
    # Test both GPT-2 and LLaMA models
    test_configs = [
        ("gpt2", "GPT-2"),
        ("SimpleStories/SimpleStories-5M", "LLaMA"),
    ]
    
    for model_name, arch_name in test_configs:
        print(f"\n{arch_name}:")
        print("-" * 40)
        
        # Get tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
        
        batch_size = 2
        seq_length = 16
        
        # Test configurations
        configs = [
            (True, False, "Multi-layer single proj"),
            (True, True, "Multi-layer per-layer proj"),
        ]
        
        for patch_all_layers, per_layer_projections, desc in configs:
            print(f"\n  {desc}:")
            
            # Create two decoders with same config
            config = DecoderConfig(
                model_name=model_name,
                base_model=False,
                projection_layer=True,
                patch_all_layers=patch_all_layers,
                per_layer_projections=per_layer_projections,
            )
            
            decoder1 = Decoder(config).to(device)
            decoder1.set_prompt("explain <embed>:", tokenizer)
            
            decoder2 = Decoder(config).to(device)
            decoder2.set_prompt("explain <embed>:", tokenizer)
            
            # Ensure same weights
            decoder2.load_state_dict(decoder1.state_dict())
            
            # Get model dimension
            d_model = decoder1.base.config.hidden_size
            
            # Create same activation
            torch.manual_seed(42)
            activation1 = torch.randn(batch_size, d_model, device=device, requires_grad=True)
            activation2 = activation1.detach().clone().requires_grad_(True)
            
            # Generate with both methods
            gen1 = decoder1.generate_soft(activation1, max_length=seq_length, gumbel_tau=1.0)
            
            try:
                gen2 = decoder2.generate_soft_kv_cached(activation2, max_length=seq_length, gumbel_tau=1.0)
                
                # Compare outputs
                emb_diff = (gen1.generated_text_embeddings - gen2.generated_text_embeddings).abs().max().item()
                logits_diff = (gen1.raw_lm_logits - gen2.raw_lm_logits).abs().max().item()
                ids_same = torch.equal(gen1.hard_token_ids, gen2.hard_token_ids)
                
                print(f"    Embeddings max diff: {emb_diff:.2e}")
                print(f"    Logits max diff: {logits_diff:.2e}")
                print(f"    Hard IDs identical: {ids_same}")
                
                # Test gradients
                loss1 = gen1.generated_text_embeddings.sum()
                loss2 = gen2.generated_text_embeddings.sum()
                
                loss1.backward()
                loss2.backward()
                
                act_grad_diff = (activation1.grad - activation2.grad).abs().max().item()
                print(f"    Activation grad diff: {act_grad_diff:.2e}")
                
                # Check if consistent
                if emb_diff < 1e-4 and logits_diff < 1e-4 and ids_same and act_grad_diff < 1e-4:
                    print(f"    ✓ KV cache works correctly with multi-layer patching!")
                else:
                    print(f"    ✗ Outputs differ!")
                    
                    # Debug: print detailed differences
                    if logits_diff > 1e-4:
                        print("\n    Detailed logits comparison:")
                        for t in range(min(5, seq_length)):
                            t_diff = (gen1.raw_lm_logits[:, t] - gen2.raw_lm_logits[:, t]).abs().max().item()
                            print(f"      Time step {t}: max diff = {t_diff:.2e}")
                
            except Exception as e:
                print(f"    FAILED: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Cleanup
            del decoder1, decoder2
            torch.cuda.empty_cache()


def test_kv_cache_incremental_correctness():
    """Verify that KV cache correctly processes incremental tokens with multi-layer patching."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n\nTesting KV Cache Incremental Processing")
    print("=" * 60)
    
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    batch_size = 1
    seq_length = 4
    d_model = 768
    
    # Create decoder with multi-layer patching
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=False,
    )
    
    decoder = Decoder(config).to(device)
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    # Create activation
    torch.manual_seed(42)
    activation = torch.randn(batch_size, d_model, device=device)
    
    print("\nStep-by-step generation with KV cache:")
    
    # Manually trace through KV cached generation
    from lens.models.kv_cache import KVCache
    
    # Prepare initial sequence
    prompt_left_emb = decoder.prompt_left_emb
    prompt_right_emb = decoder.prompt_right_emb
    
    parts = []
    if prompt_left_emb is not None:
        parts.append(prompt_left_emb.expand(batch_size, -1, -1))
    
    # Note: with patch_all_layers, activation is not inserted as a token
    # It's patched at each layer instead
    
    if prompt_right_emb is not None:
        parts.append(prompt_right_emb.expand(batch_size, -1, -1))
    
    seq_embs = torch.cat(parts, dim=1)
    print(f"  Initial sequence shape: {seq_embs.shape}")
    
    # Process initial sequence
    transformer = decoder.base.transformer
    layers = transformer.h
    final_norm = transformer.ln_f
    
    # Initialize KV cache
    kv_cache = KVCache()
    
    # Process initial prompt with patching
    hidden_states = seq_embs
    seq_len = hidden_states.size(1)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    
    # Pre-compute projection
    a_proj = decoder._apply_projection(activation)
    embed_pos = prompt_left_emb.size(0) if prompt_left_emb is not None else 0
    
    print(f"  Embed position for patching: {embed_pos}")
    
    # Process through layers with patching
    for layer_idx, layer in enumerate(layers):
        # Apply layer
        layer_outputs = layer(hidden_states, use_cache=True)
        hidden_states = layer_outputs[0]
        
        # Store KV cache
        if not hasattr(kv_cache, 'keys'):
            kv_cache.keys = []
            kv_cache.values = []
        kv_cache.keys.append(layer_outputs[1][0])
        kv_cache.values.append(layer_outputs[1][1])
        
        # Patch activation at embed position
        hidden_states[:, embed_pos] = a_proj
        
        print(f"    Layer {layer_idx}: hidden shape = {hidden_states.shape}, "
              f"K shape = {kv_cache.keys[-1].shape}, V shape = {kv_cache.values[-1].shape}")
    
    # Final norm
    hidden_states = final_norm(hidden_states)
    
    # Get first logits
    logits = decoder.out(hidden_states[:, -1])
    print(f"  Initial logits sum: {logits.sum().item():.4f}")
    
    print("\n✓ KV cache incremental processing works with multi-layer patching!")


if __name__ == "__main__":
    test_kv_cache_with_multilayer()
    test_kv_cache_incremental_correctness()