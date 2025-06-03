#!/usr/bin/env python3
"""Debug how the forward pass works in GPT-2."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def debug_gpt2_forward():
    """Debug GPT-2 forward pass."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    print("GPT-2 Forward Pass Debug")
    print("=" * 60)
    
    # Create simple input
    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    print(f"Input text: '{text}'")
    print(f"Input IDs: {input_ids}")
    
    # Method 1: Regular forward with input_ids
    print("\nMethod 1: Forward with input_ids")
    with torch.no_grad():
        outputs1 = model(input_ids=input_ids, output_hidden_states=True)
        logits1 = outputs1.logits
        hidden1 = outputs1.hidden_states[-1]
    
    # Method 2: Forward with inputs_embeds
    print("\nMethod 2: Forward with inputs_embeds")
    embed_table = model.get_input_embeddings()
    input_embeds = embed_table(input_ids)
    
    with torch.no_grad():
        outputs2 = model(inputs_embeds=input_embeds, output_hidden_states=True)
        logits2 = outputs2.logits
        hidden2 = outputs2.hidden_states[-1]
    
    # Compare
    print("\nComparison:")
    print(f"  Logits diff: {(logits1 - logits2).abs().max().item():.2e}")
    print(f"  Hidden diff: {(hidden1 - hidden2).abs().max().item():.2e}")
    
    # Method 3: Manual forward through transformer
    print("\nMethod 3: Manual forward through transformer")
    transformer = model.transformer
    
    # Get embeddings
    input_embeds = embed_table(input_ids)
    position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=device).unsqueeze(0)
    
    # Add position embeddings
    position_embeds = transformer.wpe(position_ids)
    hidden_states = input_embeds + position_embeds
    hidden_states = transformer.drop(hidden_states)
    
    # Go through layers
    with torch.no_grad():
        for layer in transformer.h:
            layer_outputs = layer(hidden_states)
            hidden_states = layer_outputs[0]
        
        # Final norm
        hidden_states = transformer.ln_f(hidden_states)
        
        # LM head
        logits3 = model.lm_head(hidden_states)
    
    # Compare with method 1
    print(f"  Logits diff (1 vs 3): {(logits1 - logits3).abs().max().item():.2e}")
    print(f"  Hidden diff (1 vs 3): {(hidden1 - hidden_states).abs().max().item():.2e}")
    
    # Check if position IDs matter
    print("\nChecking position IDs...")
    print(f"  Default position IDs: {position_ids}")
    
    # Try with explicit position_ids in forward
    with torch.no_grad():
        outputs4 = model(inputs_embeds=input_embeds, position_ids=position_ids, output_hidden_states=True)
        logits4 = outputs4.logits
    
    print(f"  Logits diff (2 vs 4): {(logits2 - logits4).abs().max().item():.2e}")


if __name__ == "__main__":
    debug_gpt2_forward()