#!/usr/bin/env python3
"""Test multi-layer patching with LLaMA-style models."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def test_simplestories_model():
    """Test with SimpleStories model (LLaMA architecture)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "SimpleStories/SimpleStories-5M"
    
    print(f"Testing with {model_name}")
    print("=" * 60)
    
    # Try to load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Could not load tokenizer for {model_name}: {e}")
        print("Using GPT-2 tokenizer as fallback")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    
    batch_size = 1
    seq_length = 4
    
    # Test configurations
    configs = [
        ("No patching", False, False),
        ("Multi-layer, single proj", True, False), 
        ("Multi-layer, per-layer proj", True, True),
    ]
    
    for config_name, patch_all_layers, per_layer_projections in configs:
        print(f"\n{config_name}:")
        print("-" * 40)
        
        try:
            # Create decoder
            config = DecoderConfig(
                model_name=model_name,
                base_model=False,
                projection_layer=True,
                patch_all_layers=patch_all_layers,
                per_layer_projections=per_layer_projections,
            )
            
            decoder = Decoder(config).to(device)
            d_model = decoder.base.config.hidden_size
            
            # Set prompt
            decoder.set_prompt("explain <embed>:", tokenizer)
            
            # Create activation
            activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
            
            # Test generation
            gen = decoder.generate_soft(activation, max_length=seq_length, gumbel_tau=1.0)
            
            # Compute loss and gradients
            loss = gen.generated_text_embeddings.sum()
            loss.backward()
            
            print(f"  Generated shape: {gen.generated_text_embeddings.shape}")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Activation grad norm: {activation.grad.norm().item():.4f}")
            
            if per_layer_projections:
                n_layers = decoder.base.config.num_hidden_layers
                print(f"  Per-layer projection gradient norms:")
                for i in range(min(5, n_layers)):  # Show first 5 layers
                    grad_norm = decoder.proj_weight.grad[i].norm().item()
                    print(f"    Layer {i}: {grad_norm:.4f}")
                if n_layers > 5:
                    print(f"    ... ({n_layers - 5} more layers)")
            elif decoder.proj is not None:
                print(f"  Projection grad norm: {decoder.proj.weight.grad.norm().item():.4f}")
            
            print("  ✓ Success!")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Clean up
        torch.cuda.empty_cache()


def test_model_architecture_detection():
    """Test architecture detection for different models."""
    print("\n\nTesting Model Architecture Detection")
    print("=" * 60)
    
    models_to_test = [
        ("gpt2", "GPT-2"),
        ("SimpleStories/SimpleStories-5M", "SimpleStories (LLaMA)"),
    ]
    
    for model_name, description in models_to_test:
        print(f"\n{description} ({model_name}):")
        try:
            config = DecoderConfig(
                model_name=model_name,
                base_model=False,
                patch_all_layers=True,  # Enable to test architecture detection
            )
            decoder = Decoder(config)
            
            # Check architecture
            if hasattr(decoder.base, 'transformer'):
                print("  ✓ Detected GPT-2 style architecture")
                print(f"    - Transformer module: {type(decoder.base.transformer)}")
                print(f"    - Layers: {type(decoder.base.transformer.h)}")
                print(f"    - Final norm: {type(decoder.base.transformer.ln_f)}")
            elif hasattr(decoder.base, 'model'):
                print("  ✓ Detected LLaMA style architecture")
                print(f"    - Model module: {type(decoder.base.model)}")
                print(f"    - Layers: {type(decoder.base.model.layers)}")
                print(f"    - Final norm: {type(decoder.base.model.norm)}")
            else:
                print("  ✗ Unknown architecture")
                
        except Exception as e:
            print(f"  ✗ Error loading model: {e}")


if __name__ == "__main__":
    test_model_architecture_detection()
    test_simplestories_model()
    
    print("\n\n" + "=" * 60)
    print("✓ LLaMA compatibility tests completed!")
    print("=" * 60)