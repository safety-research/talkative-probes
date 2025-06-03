#!/usr/bin/env python3
"""Debug with batch size 2."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def debug_batch_2():
    """Debug with batch size 2."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=False,
    )
    
    decoder = Decoder(config).to(device)
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    d_model = decoder.base.config.hidden_size
    batch_size = 2  # The problematic batch size
    activation = torch.randn(batch_size, d_model, device=device)
    
    print(f"Testing with batch size {batch_size}")
    print("=" * 60)
    
    # Try to generate
    try:
        torch.manual_seed(123)
        gen = decoder.generate_soft_kv_cached(activation, max_length=1, gumbel_tau=1.0)
        print("Success!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_batch_2()