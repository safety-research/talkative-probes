#!/usr/bin/env python3
"""Debug output layer differences."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def debug_output_layers():
    """Debug differences between output layers."""
    
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
    
    print("Debugging output layers")
    print("=" * 60)
    
    # Check what output layers exist
    print("\n1. Decoder structure:")
    print(f"   decoder.out: {decoder.out}")
    print(f"   decoder.base type: {type(decoder.base)}")
    print(f"   Has lm_head: {hasattr(decoder.base, 'lm_head')}")
    if hasattr(decoder.base, 'lm_head'):
        print(f"   decoder.base.lm_head: {decoder.base.lm_head}")
    
    # Check if weights are the same
    print("\n2. Weight comparison:")
    if hasattr(decoder.base, 'lm_head'):
        same_weights = torch.allclose(decoder.out.weight, decoder.base.lm_head.weight)
        print(f"   Weights are same: {same_weights}")
        if not same_weights:
            weight_diff = (decoder.out.weight - decoder.base.lm_head.weight).abs()
            print(f"   Max weight diff: {weight_diff.max().item():.6f}")
            print(f"   Mean weight diff: {weight_diff.mean().item():.6f}")
    
    # Test with a sample hidden state
    torch.manual_seed(42)
    test_hidden = torch.randn(1, 768, device=device)
    test_hidden = test_hidden * 200.939 / test_hidden.norm()  # Scale to match observed norm
    
    print("\n3. Output comparison with test hidden state:")
    print(f"   Test hidden norm: {test_hidden.norm().item():.3f}")
    
    # Using decoder.out
    logits_decoder_out = decoder.out(test_hidden)
    print(f"   decoder.out logits top 5: {[f'{v:.3f}' for v in logits_decoder_out[0].topk(5).values.tolist()]}")
    
    # Using decoder.base.lm_head if available
    if hasattr(decoder.base, 'lm_head'):
        logits_lm_head = decoder.base.lm_head(test_hidden)
        print(f"   base.lm_head logits top 5: {[f'{v:.3f}' for v in logits_lm_head[0].topk(5).values.tolist()]}")
        
    # Check get_output_embeddings
    output_embeddings = decoder.base.get_output_embeddings()
    print(f"\n4. get_output_embeddings: {output_embeddings}")
    if output_embeddings is not None:
        logits_output_emb = output_embeddings(test_hidden)
        print(f"   get_output_embeddings logits top 5: {[f'{v:.3f}' for v in logits_output_emb[0].topk(5).values.tolist()]}")
    
    # Check if there's any difference in how they're applied
    print("\n5. Checking actual generation code path:")
    print("   In generate_soft_kv_cached:")
    print("   - When override_model_base_and_out is None: uses self.out")
    print("   - When override_model_base_and_out is set: uses model.lm_head or get_output_embeddings()")


if __name__ == "__main__":
    debug_output_layers()