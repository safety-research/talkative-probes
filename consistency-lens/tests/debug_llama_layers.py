#!/usr/bin/env python3
"""Debug how to call LLaMA layers properly."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def debug_llama_layers():
    """Check how LLaMA model processes inputs."""
    model_name = "SimpleStories/SimpleStories-5M"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # Check model structure
    print(f"Model type: {type(model)}")
    print(f"Has transformer: {hasattr(model, 'transformer')}")
    print(f"Has model: {hasattr(model, 'model')}")
    
    if hasattr(model, 'model'):
        base_model = model.model
        print(f"\nBase model type: {type(base_model)}")
        print(f"Has embed_tokens: {hasattr(base_model, 'embed_tokens')}")
        print(f"Has layers: {hasattr(base_model, 'layers')}")
        print(f"Number of layers: {len(base_model.layers)}")
        
        # Create dummy input
        batch_size = 1
        seq_length = 10
        hidden_size = base_model.config.hidden_size
        
        # Create embeddings
        dummy_ids = torch.randint(0, 1000, (batch_size, seq_length), device=device)
        embeddings = base_model.embed_tokens(dummy_ids)
        print(f"\nEmbeddings shape: {embeddings.shape}")
        
        # Create position ids
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)
        
        # Try calling a layer directly
        layer = base_model.layers[0]
        print(f"\nLayer type: {type(layer)}")
        
        # Check what the layer's forward method expects
        import inspect
        sig = inspect.signature(layer.forward)
        print(f"\nLayer forward signature: {sig}")
        
        # Try different ways to call the layer
        print("\n\nTesting layer calls:")
        
        # Test 1: Just hidden states
        try:
            print("Test 1: hidden_states only")
            outputs = layer(embeddings)
            print("  Success! Output shape:", outputs[0].shape)
        except Exception as e:
            print(f"  Failed: {type(e).__name__}: {e}")
        
        # Test 2: With position_ids
        try:
            print("\nTest 2: hidden_states + position_ids")
            outputs = layer(embeddings, position_ids=position_ids)
            print("  Success! Output shape:", outputs[0].shape)
        except Exception as e:
            print(f"  Failed: {type(e).__name__}: {e}")
        
        # Test 3: Check how the full model processes
        print("\n\nChecking full model forward pass:")
        with torch.no_grad():
            # Use the model's forward method
            outputs = model(inputs_embeds=embeddings, output_hidden_states=True)
            print(f"Model outputs keys: {outputs.keys()}")
            
        # Check if we can extract rotary embeddings
        print("\n\nChecking for rotary embeddings:")
        if hasattr(base_model, 'rotary_emb'):
            print(f"Has rotary_emb: {base_model.rotary_emb}")
        
        # Check first layer for rotary
        if hasattr(layer.self_attn, 'rotary_emb'):
            print(f"Layer has rotary_emb: {layer.self_attn.rotary_emb}")
            
            # Try to get rotary embeddings
            rotary_emb = layer.self_attn.rotary_emb
            cos, sin = rotary_emb(embeddings, position_ids)
            print(f"Rotary cos shape: {cos.shape}")
            print(f"Rotary sin shape: {sin.shape}")


if __name__ == "__main__":
    debug_llama_layers()