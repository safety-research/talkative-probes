#!/usr/bin/env python3
"""Check if generate_soft is using the right variable."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def main():
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
    
    decoder = Decoder(config).to(device).eval()
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    d_model = decoder.base.config.hidden_size
    activation = torch.randn(1, d_model, device=device)
    
    # Check generate_soft behavior
    print("Testing generate_soft with patch_all_layers=True")
    print("Looking at lines 443-449 in decoder.py:")
    print("  It seems to pass 'hidden_states' instead of 'input_to_this_layer'")
    print("  This would mean the patching has no effect!")
    
    # Generate a few tokens to see the pattern
    torch.manual_seed(42)
    gen = decoder.generate_soft(activation, max_length=5, gumbel_tau=1.0)
    tokens = gen.hard_token_ids[0].tolist()
    text = tokenizer.decode(tokens)
    print(f"\nGenerated tokens: {tokens}")
    print(f"Generated text: '{text}'")
    
    # The issue is that the generate_soft is not using the patched input properly


if __name__ == "__main__":
    main()