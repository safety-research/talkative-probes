#!/usr/bin/env python3
"""Test script specifically for SimpleStories-5M model layer arguments."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import inspect
import os

def test_model_layers(model_path):
    """Test a specific model's layer structure and arguments."""
    print(f"\nTesting model: {model_path}")
    
    try:
        # Try to load the model
        if os.path.exists(model_path):
            model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path)
        
        print(f"Model class: {model.__class__.__name__}")
        print(f"Config: {model.config}")
        
        # Navigate to the layers
        if hasattr(model, 'model'):  # LLaMA/GPT2 structure
            base_model = model.model
        elif hasattr(model, 'transformer'):  # GPT2 structure
            base_model = model.transformer
        else:
            print("Unknown model structure!")
            return
        
        print(f"\nBase model class: {base_model.__class__.__name__}")
        
        # Get layers
        if hasattr(base_model, 'layers'):  # LLaMA
            layers = base_model.layers
            layer_attr = 'layers'
        elif hasattr(base_model, 'h'):  # GPT2
            layers = base_model.h
            layer_attr = 'h'
        else:
            print("Cannot find layers!")
            return
        
        print(f"Number of layers: {len(layers)}")
        first_layer = layers[0]
        print(f"Layer class: {first_layer.__class__.__name__}")
        
        # Inspect the forward method
        print(f"\nLayer forward signature:")
        sig = inspect.signature(first_layer.forward)
        for param_name, param in sig.parameters.items():
            print(f"  {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}")
            if param.default != inspect.Parameter.empty:
                print(f"    default: {param.default}")
        
        # Test calling the layer
        batch_size = 2
        seq_len = 10
        hidden_size = model.config.hidden_size
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        print(f"\n--- Testing layer call with hidden_states shape: {hidden_states.shape} ---")
        
        # Minimal call
        try:
            output = first_layer(hidden_states)
            print(f"✓ Minimal call successful")
            if isinstance(output, tuple):
                print(f"  Returns tuple of length {len(output)}")
                print(f"  First element shape: {output[0].shape}")
        except Exception as e:
            print(f"✗ Minimal call failed: {e}")
        
        # With attention mask
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        try:
            output = first_layer(hidden_states, attention_mask=attention_mask)
            print(f"✓ Call with attention_mask successful")
        except Exception as e:
            print(f"✗ Call with attention_mask failed: {e}")
        
        # Check for position embedding handling
        print(f"\n--- Position embedding analysis ---")
        
        # Check if model uses RoPE (Rotary Position Embeddings)
        if hasattr(first_layer, 'self_attn'):
            attn = first_layer.self_attn
            if hasattr(attn, 'rotary_emb'):
                print(f"✓ Uses RoPE (Rotary Position Embeddings)")
                print(f"  RoPE class: {attn.rotary_emb.__class__.__name__}")
            else:
                print("✗ No rotary_emb found in attention")
        
        # Check base model for position embeddings
        if hasattr(base_model, 'embed_positions'):
            print(f"✓ Has embed_positions: {base_model.embed_positions.__class__.__name__}")
        elif hasattr(base_model, 'wpe'):  # GPT2 style
            print(f"✓ Has wpe (position embeddings): {base_model.wpe.__class__.__name__}")
        else:
            print("✗ No explicit position embeddings found")
        
        # Try with position_ids
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        try:
            output = first_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            print(f"✓ Call with position_ids successful")
        except TypeError as e:
            if "position_ids" in str(e):
                print(f"✗ Layer doesn't accept position_ids")
            else:
                print(f"✗ Call with position_ids failed: {e}")
        
        return model, base_model, layers
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None, None, None

def main():
    print("=" * 60)
    print("Testing layer arguments for various models")
    print("=" * 60)
    
    # Test different model possibilities
    test_models = [
        "conceptofmind/cot_simplestories5M",  # Possible SimpleStories-5M
        "roneneldan/TinyStories-33M",         # Similar small model
        "gpt2",                               # Standard GPT2
    ]
    
    # Also check local paths
    local_paths = [
        "./data/models/SimpleStories-5M",
        "../models/SimpleStories-5M",
        "/workspace/models/SimpleStories-5M",
    ]
    
    # Test models
    for model_path in test_models + local_paths:
        if model_path in local_paths and not os.path.exists(model_path):
            continue
        
        model, base_model, layers = test_model_layers(model_path)
        if model is not None:
            print("\n" + "=" * 60)
            break
    
    # If we successfully loaded a model, do more detailed tests
    if model is not None and layers is not None:
        print("\n--- Detailed position encoding test ---")
        
        # Create a simple input
        input_text = "Once upon a time"
        
        # Try to find or create tokenizer
        try:
            if 'tokenizer' in locals():
                pass
            elif hasattr(model, 'config') and hasattr(model.config, 'name_or_path'):
                tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained("gpt2")  # fallback
            
            inputs = tokenizer(input_text, return_tensors="pt")
            input_ids = inputs.input_ids
            
            print(f"Input: '{input_text}'")
            print(f"Input IDs shape: {input_ids.shape}")
            
            # Get embeddings
            if hasattr(base_model, 'embed_tokens'):  # LLaMA
                embeddings = base_model.embed_tokens(input_ids)
            elif hasattr(base_model, 'wte'):  # GPT2
                embeddings = base_model.wte(input_ids)
            else:
                print("Cannot find token embeddings!")
                return
            
            print(f"Token embeddings shape: {embeddings.shape}")
            
            # Add position embeddings if needed
            seq_length = input_ids.shape[1]
            position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
            
            if hasattr(base_model, 'embed_positions'):  # Some LLaMA variants
                pos_embeds = base_model.embed_positions(input_ids)
                hidden_states = embeddings + pos_embeds
                print("Added embed_positions")
            elif hasattr(base_model, 'wpe'):  # GPT2
                pos_embeds = base_model.wpe(position_ids)
                hidden_states = embeddings + pos_embeds
                print("Added wpe position embeddings")
            else:
                hidden_states = embeddings
                print("No explicit position embeddings added")
            
            # Process through first layer
            print(f"\nProcessing through first layer...")
            print(f"Hidden states shape: {hidden_states.shape}")
            
            layer_output = layers[0](hidden_states)
            
            if isinstance(layer_output, tuple):
                print(f"✓ Layer output: tuple of length {len(layer_output)}")
                print(f"  Output shape: {layer_output[0].shape}")
            else:
                print(f"✓ Layer output shape: {layer_output.shape}")
            
        except Exception as e:
            print(f"Detailed test failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()