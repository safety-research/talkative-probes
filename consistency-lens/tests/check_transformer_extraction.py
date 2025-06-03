#!/usr/bin/env python3
"""Check transformer extraction logic."""

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from lens.models.decoder import Decoder, DecoderConfig


def check_transformer():
    """Check transformer extraction."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    
    # Create decoder
    decoder = Decoder(DecoderConfig(
        model_name=model_name,
        use_kv_cache=True,
        use_flash_attention=False,
        base_model=False,
        projection_layer=True,
    )).to(device)
    
    print("Checking transformer extraction")
    print("=" * 60)
    
    # Check the structure
    print(f"decoder.base type: {type(decoder.base)}")
    print(f"decoder.base attributes: {[attr for attr in dir(decoder.base) if not attr.startswith('_')][:10]}")
    
    # Check if it has transformer
    print(f"\nhasattr(decoder.base, 'transformer'): {hasattr(decoder.base, 'transformer')}")
    print(f"hasattr(decoder.base, 'model'): {hasattr(decoder.base, 'model')}")
    
    # What the KV method tries
    main_base = decoder.base
    if hasattr(main_base, 'transformer'):
        transformer_kv = main_base.transformer
        print(f"\nKV method gets: transformer (type: {type(transformer_kv)})")
    elif hasattr(main_base, 'model'):
        transformer_kv = main_base.model
        print(f"\nKV method gets: model (type: {type(transformer_kv)})")
    else:
        print(f"\nKV method fails!")
        transformer_kv = None
    
    # What the Flash method tries
    transformer_flash = main_base.transformer if hasattr(main_base, 'transformer') else main_base
    print(f"Flash method gets: {type(transformer_flash)}")
    
    # Compare a direct GPT2LMHeadModel
    direct_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    print(f"\nDirect GPT2LMHeadModel has transformer: {hasattr(direct_model, 'transformer')}")
    print(f"Direct transformer type: {type(direct_model.transformer)}")
    
    # Test if compute_with_kv_cache works with the wrong input
    if transformer_kv is not None:
        print(f"\nTesting compute_with_kv_cache with extracted transformer...")
        test_input = torch.randn(1, 5, 768, device=device)
        
        from lens.models.kv_cache import compute_with_kv_cache
        try:
            hidden, _ = compute_with_kv_cache(transformer_kv, test_input, use_cache=False)
            print(f"Success! Output shape: {hidden.shape}")
        except Exception as e:
            print(f"Error: {e}")
            
    # What if we pass the full model?
    print(f"\nTesting with full decoder.base...")
    try:
        hidden, _ = compute_with_kv_cache(decoder.base, test_input, use_cache=False)
        print(f"Success! Output shape: {hidden.shape}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    check_transformer()