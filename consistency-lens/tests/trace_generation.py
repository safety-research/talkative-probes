#!/usr/bin/env python3
"""Trace through generation to find where -12 values come from."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def trace_generation():
    """Trace generation to find issue."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create decoders
    decoder_kv = Decoder(DecoderConfig(
        model_name=model_name,
        use_kv_cache=True,
        use_flash_attention=False,
        base_model=False,
        projection_layer=True,
    )).to(device)
    
    decoder_flash = Decoder(DecoderConfig(
        model_name=model_name,
        use_kv_cache=False,
        use_flash_attention=True,
        base_model=False,
        projection_layer=True,
    )).to(device)
    
    # Copy weights
    decoder_flash.load_state_dict(decoder_kv.state_dict())
    
    # Set prompts
    decoder_kv.set_prompt("explain <embed>:", tokenizer)
    decoder_flash.set_prompt("explain <embed>:", tokenizer)
    
    # Test activation
    torch.manual_seed(42)
    activation = torch.randn(1, 768, device=device)
    
    print("Tracing generation issue")
    print("=" * 60)
    
    # Generate with both and print intermediate values
    print("\nGenerating with KV cache...")
    with torch.no_grad():
        gen_kv = decoder_kv.generate_soft_kv_cached(
            activation.clone(), 
            max_length=2,  # Just 2 steps
            gumbel_tau=0.0,
            print_prompt=True
        )
    
    print(f"\nKV generation output:")
    print(f"  Shape: {gen_kv.raw_lm_logits.shape}")
    print(f"  Step 0 top 5: {[f'{v:.3f}' for v in gen_kv.raw_lm_logits[0, 0].topk(5).values.tolist()]}")
    print(f"  Step 1 top 5: {[f'{v:.3f}' for v in gen_kv.raw_lm_logits[0, 1].topk(5).values.tolist()]}")
    
    print("\n\nGenerating with Flash...")
    with torch.no_grad():
        gen_flash = decoder_flash.generate_soft_kv_flash(
            activation.clone(), 
            max_length=2,
            gumbel_tau=0.0,
            print_prompt=True
        )
    
    print(f"\nFlash generation output:")
    print(f"  Shape: {gen_flash.raw_lm_logits.shape}")
    print(f"  Step 0 top 5: {[f'{v:.3f}' for v in gen_flash.raw_lm_logits[0, 0].topk(5).values.tolist()]}")
    print(f"  Step 1 top 5: {[f'{v:.3f}' for v in gen_flash.raw_lm_logits[0, 1].topk(5).values.tolist()]}")
    
    # Check the actual decoder outputs
    print("\n\nChecking decoder components:")
    print(f"KV decoder output layer: {decoder_kv.out}")
    print(f"Flash decoder output layer: {decoder_flash.out}")
    
    # Check if they're using the same underlying model
    print(f"\nKV base model: {type(decoder_kv.base)}")
    print(f"Flash base model: {type(decoder_flash.base)}")
    
    # Try manual computation
    print("\n\nManual computation check:")
    
    # Get prompt embeddings
    B = 1
    emb_a = decoder_kv.proj(activation)
    left_prompt_embs = decoder_kv.prompt_left_emb.unsqueeze(0).expand(B, -1, -1)
    right_prompt_embs = decoder_kv.prompt_right_emb.unsqueeze(0).expand(B, -1, -1)
    prompt_embs = torch.cat([left_prompt_embs, emb_a.unsqueeze(1), right_prompt_embs], dim=1)
    
    # Get transformer
    transformer = decoder_kv.base.transformer if hasattr(decoder_kv.base, 'transformer') else decoder_kv.base
    
    # Forward through transformer
    outputs = transformer(inputs_embeds=prompt_embs)
    hidden = outputs.last_hidden_state
    
    # Apply output layer
    logits_manual = decoder_kv.out(hidden[:, -1])
    print(f"Manual logits top 5: {[f'{v:.3f}' for v in logits_manual[0].topk(5).values.tolist()]}")
    
    # Check decoder.out vs decoder.base.lm_head
    if hasattr(decoder_kv.base, 'lm_head'):
        logits_lm_head = decoder_kv.base.lm_head(hidden[:, -1])
        print(f"Using base.lm_head top 5: {[f'{v:.3f}' for v in logits_lm_head[0].topk(5).values.tolist()]}")


if __name__ == "__main__":
    trace_generation()