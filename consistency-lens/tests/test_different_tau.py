#!/usr/bin/env python3
"""Test with different gumbel_tau values."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def test_tau(tau):
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
    
    # Generate with both methods
    torch.manual_seed(42)
    gen1 = decoder.generate_soft(activation.clone(), max_length=10, gumbel_tau=tau)
    
    torch.manual_seed(42)
    gen2 = decoder.generate_soft_kv_cached(activation.clone(), max_length=10, gumbel_tau=tau)
    
    # Compare results
    tokens1 = gen1.hard_token_ids[0].tolist()
    tokens2 = gen2.hard_token_ids[0].tolist()
    
    text1 = tokenizer.decode(tokens1)
    text2 = tokenizer.decode(tokens2)
    
    match = tokens1 == tokens2
    print(f"\nTau={tau}:")
    print(f"  generate_soft:      '{text1}'")
    print(f"  generate_kv_cached: '{text2}'")
    print(f"  Match: {match}")
    
    return match


def main():
    print("Testing different gumbel_tau values")
    print("=" * 60)
    
    tau_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    for tau in tau_values:
        test_tau(tau)


if __name__ == "__main__":
    main()