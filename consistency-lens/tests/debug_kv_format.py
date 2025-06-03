#!/usr/bin/env python3
"""Debug the KV cache format and usage."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lens.models.decoder import Decoder, DecoderConfig


def debug_kv_format():
    """Debug how past_key_values should be formatted and used."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Debugging KV Cache Format")
    print("=" * 60)
    
    # First, let's see how the native model does it
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    
    # Test sequence
    text = "The quick brown"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    print(f"Test input: '{text}'")
    
    # Get initial forward pass with caching
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            use_cache=True
        )
        past_kv = outputs.past_key_values
    
    print(f"\nNative model past_key_values:")
    print(f"  Type: {type(past_kv)}")
    print(f"  Length: {len(past_kv)} (number of layers)")
    print(f"  Each element type: {type(past_kv[0])}")
    print(f"  Each element length: {len(past_kv[0])} (key, value)")
    print(f"  Key shape: {past_kv[0][0].shape}")
    print(f"  Value shape: {past_kv[0][1].shape}")
    
    # Generate next token
    next_token_logits = outputs.logits[0, -1]
    next_token = next_token_logits.argmax().unsqueeze(0).unsqueeze(0)
    
    print(f"\nGenerating next token: {next_token.item()} ('{tokenizer.decode([next_token.item()])}')")
    
    # Process next token with cache
    with torch.no_grad():
        next_outputs = model(
            input_ids=next_token,
            past_key_values=past_kv,
            use_cache=True
        )
    
    print(f"Next token logits range: [{next_outputs.logits.min().item():.2f}, {next_outputs.logits.max().item():.2f}]")
    
    # Now let's check our implementation
    print("\n\nOur implementation:")
    
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=False,
    )
    
    decoder = Decoder(config).to(device).eval()
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    # Get components
    d_model = decoder.base.config.hidden_size
    activation = torch.randn(1, d_model, device=device)
    
    # Build initial sequence
    parts = []
    if decoder.prompt_left_emb is not None:
        parts.append(decoder.prompt_left_emb.expand(1, -1, -1))
    a_proj = decoder._apply_projection(activation).unsqueeze(1)
    parts.append(a_proj)
    if decoder.prompt_right_emb is not None:
        parts.append(decoder.prompt_right_emb.expand(1, -1, -1))
    seq_embs = torch.cat(parts, dim=1)
    
    # Try using the model's native forward for initial pass
    print("\nUsing model's native forward for initial pass:")
    with torch.no_grad():
        # We can't use native forward with multi-layer patching
        # So let's check if the past_kv format matches
        
        # Manual forward
        transformer = decoder.base.transformer
        layers = transformer.h
        
        # Collect past_kv from first layer
        hidden = seq_embs
        position_ids = torch.arange(seq_embs.size(1), device=device).unsqueeze(0)
        
        layer_out = layers[0](
            hidden,
            position_ids=position_ids,
            use_cache=True
        )
        
        if len(layer_out) > 1 and layer_out[1] is not None:
            our_kv = layer_out[1]
            print(f"  Our KV type: {type(our_kv)}")
            print(f"  Our KV length: {len(our_kv)}")
            print(f"  Our key shape: {our_kv[0].shape}")
            print(f"  Our value shape: {our_kv[1].shape}")
            
            # Check if format matches
            print(f"\n  Format matches native: {type(our_kv) == type(past_kv[0])}")
            print(f"  Key shape matches: {our_kv[0].shape[1:] == past_kv[0][0].shape[1:]}")  # Ignore batch
    
    # The issue might be elsewhere - let's check layer behavior
    print("\n\nChecking layer behavior with past_key_values:")
    
    # Create a simple test
    test_hidden = torch.randn(1, 1, 768, device=device)
    test_pos = torch.tensor([[3]], device=device)
    
    # Forward without cache
    out_no_cache = layers[0](test_hidden, position_ids=test_pos, use_cache=False)[0]
    
    # Forward with empty cache
    out_with_cache = layers[0](test_hidden, position_ids=test_pos, use_cache=True)
    
    print(f"Output without cache shape: {out_no_cache.shape}")
    print(f"Output with cache shape: {out_with_cache[0].shape}")
    print(f"Output difference: {(out_no_cache - out_with_cache[0]).abs().max().item():.2e}")


if __name__ == "__main__":
    debug_kv_format()