#!/usr/bin/env python3
"""Debug why KV cache produces different logit magnitudes."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def debug_kv_logits():
    """Debug KV cache logit issue."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create decoder
    decoder = Decoder(DecoderConfig(
        model_name=model_name,
        use_kv_cache=True,
        use_flash_attention=False,
        base_model=False,
        projection_layer=True,
    )).to(device)
    
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    print("Debugging KV Cache Logits Issue")
    print("=" * 60)
    
    # Test activation
    torch.manual_seed(42)
    activation = torch.randn(1, 768, device=device)
    
    # Check what decoder.out actually is
    print(f"decoder.out: {decoder.out}")
    print(f"decoder.base.lm_head: {decoder.base.lm_head}")
    print(f"Are they the same object? {decoder.out is decoder.base.lm_head}")
    
    # Check weights
    print(f"\nWeight comparison:")
    print(f"decoder.out weight shape: {decoder.out.weight.shape}")
    print(f"decoder.base.lm_head weight shape: {decoder.base.lm_head.weight.shape}")
    print(f"Weights equal? {torch.allclose(decoder.out.weight, decoder.base.lm_head.weight)}")
    
    # Test with a simple hidden state
    test_hidden = torch.randn(1, 768, device=device)
    logits1 = decoder.out(test_hidden)
    logits2 = decoder.base.lm_head(test_hidden)
    
    print(f"\nTest logits comparison:")
    print(f"decoder.out output: {logits1[0, :5]}")
    print(f"base.lm_head output: {logits2[0, :5]}")
    print(f"Outputs equal? {torch.allclose(logits1, logits2)}")
    
    # Now check if there's any override happening
    print(f"\n\nChecking for overrides in generate_soft_kv_cached:")
    
    # Look at the actual generation
    # Set a breakpoint-like inspection
    original_out = decoder.out
    
    # Track calls
    call_count = 0
    last_input_shape = None
    
    def tracking_forward(self, x):
        nonlocal call_count, last_input_shape
        call_count += 1
        last_input_shape = x.shape
        print(f"  Output layer called #{call_count} with shape {x.shape}")
        result = original_out._forward(x)
        print(f"  Result shape: {result.shape}, norm: {result.norm():.3f}")
        return result
    
    # Monkey patch to track calls
    import types
    decoder.out._forward = decoder.out.forward
    decoder.out.forward = types.MethodType(tracking_forward, decoder.out)
    
    print("\nRunning generation with tracking:")
    with torch.no_grad():
        gen = decoder.generate_soft_kv_cached(
            activation.clone(), 
            max_length=1,
            gumbel_tau=0.0
        )
    
    print(f"\nTotal output layer calls: {call_count}")
    print(f"Generated logits: {gen.raw_lm_logits[0, 0, :5]}")
    
    # Check if override_model_base_and_out is being used somehow
    print(f"\n\nChecking override logic:")
    
    # Simulate what happens in generate_soft_kv_cached
    B, d_model = activation.shape
    
    # Project activation
    emb_a = decoder.proj(activation)
    
    # Build prompt
    left_prompt_embs = decoder.prompt_left_emb.unsqueeze(0).expand(B, -1, -1)
    right_prompt_embs = decoder.prompt_right_emb.unsqueeze(0).expand(B, -1, -1)
    seq_embs = torch.cat([left_prompt_embs, emb_a.unsqueeze(1), right_prompt_embs], dim=1)
    
    # Check the override_model_base_and_out logic
    override_model_base_and_out = None  # Default in the method
    
    if override_model_base_and_out is not None:
        print("Override is set!")
    else:
        # This is what actually runs
        main_model = decoder
        main_base = decoder.base
        main_out = decoder.out
        
        print(f"Using decoder's own components:")
        print(f"  main_out is decoder.out: {main_out is decoder.out}")
        print(f"  main_out is decoder.base.lm_head: {main_out is decoder.base.lm_head}")


if __name__ == "__main__":
    debug_kv_logits()