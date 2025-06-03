#!/usr/bin/env python3
"""Test script to understand LLaMA layer arguments and position embeddings."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import inspect

def main():
    # Load the SimpleStories-5M model
    model_name = "roneneldan/TinyStories-33M"  # Using a publicly available similar model
    print(f"Loading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Print model architecture
    print(f"\nModel type: {type(model)}")
    print(f"Model config: {model.config}")
    
    # Get the transformer/decoder part
    if hasattr(model, 'model'):  # LLaMA structure
        transformer = model.model
        print(f"\nTransformer type: {type(transformer)}")
        
        # Check layers
        if hasattr(transformer, 'layers'):
            layers = transformer.layers
            print(f"Number of layers: {len(layers)}")
            print(f"Layer type: {type(layers[0])}")
            
            # Inspect first layer's forward method
            first_layer = layers[0]
            print(f"\nFirst layer forward signature:")
            print(inspect.signature(first_layer.forward))
            
            # Create dummy inputs
            batch_size = 2
            seq_len = 10
            hidden_size = model.config.hidden_size
            
            # Create input tensors
            hidden_states = torch.randn(batch_size, seq_len, hidden_size)
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
            
            # Test what happens with minimal arguments
            print("\n--- Testing layer calls ---")
            
            # Test 1: Just hidden states
            try:
                print("\nTest 1: Calling with just hidden_states")
                output = first_layer(hidden_states)
                print(f"Success! Output type: {type(output)}")
                if isinstance(output, tuple):
                    print(f"Output length: {len(output)}")
                    print(f"First element shape: {output[0].shape}")
            except Exception as e:
                print(f"Failed: {type(e).__name__}: {e}")
            
            # Test 2: With attention mask
            try:
                print("\nTest 2: Calling with hidden_states and attention_mask")
                output = first_layer(hidden_states, attention_mask=attention_mask)
                print(f"Success! Output type: {type(output)}")
            except Exception as e:
                print(f"Failed: {type(e).__name__}: {e}")
            
            # Test 3: Check position embeddings
            print("\n--- Checking position embeddings ---")
            
            # Check if model has rotary embeddings
            if hasattr(transformer, 'embed_positions'):
                print(f"Model has embed_positions: {type(transformer.embed_positions)}")
            
            # Check if layers use rotary embeddings internally
            if hasattr(first_layer, 'self_attn'):
                attn = first_layer.self_attn
                print(f"\nAttention module type: {type(attn)}")
                print(f"Attention forward signature:")
                print(inspect.signature(attn.forward))
                
                # Check for rotary embeddings
                if hasattr(attn, 'rotary_emb'):
                    print(f"Has rotary_emb: {type(attn.rotary_emb)}")
                
            # Test 4: Try with position_ids
            try:
                print("\nTest 4: Calling with position_ids")
                position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
                output = first_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids
                )
                print(f"Success! Output type: {type(output)}")
            except Exception as e:
                print(f"Failed: {type(e).__name__}: {e}")
            
            # Test 5: Full forward pass through model
            print("\n--- Testing full model forward ---")
            input_ids = tokenizer("Hello world", return_tensors="pt").input_ids
            print(f"Input shape: {input_ids.shape}")
            
            with torch.no_grad():
                outputs = model(input_ids)
                print(f"Model output keys: {outputs.keys() if hasattr(outputs, 'keys') else type(outputs)}")
            
            # Test 6: Extract embeddings and manually process through layers
            print("\n--- Manual layer processing ---")
            
            # Get input embeddings
            if hasattr(transformer, 'embed_tokens'):
                embeddings = transformer.embed_tokens(input_ids)
                print(f"Embeddings shape: {embeddings.shape}")
                
                # Try to process through first layer manually
                hidden_states = embeddings
                
                # Check if we need to add position embeddings before layers
                if hasattr(transformer, 'embed_positions'):
                    print("Adding position embeddings...")
                    positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
                    pos_embeds = transformer.embed_positions(positions)
                    hidden_states = hidden_states + pos_embeds
                
                # Process through first layer
                try:
                    print("\nProcessing through first layer manually...")
                    layer_output = first_layer(hidden_states)
                    print(f"Manual processing successful!")
                    print(f"Output shape: {layer_output[0].shape if isinstance(layer_output, tuple) else layer_output.shape}")
                except Exception as e:
                    print(f"Manual processing failed: {type(e).__name__}: {e}")
    
    print("\n--- Summary ---")
    print("Key findings about layer arguments will be printed above.")

if __name__ == "__main__":
    main()