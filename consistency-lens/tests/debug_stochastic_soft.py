#!/usr/bin/env python3
"""Debug stochastic differences and soft token handling."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def debug_stochastic_and_soft():
    """Check for stochastic differences and soft token handling."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Debugging Stochastic Effects and Soft Tokens")
    print("=" * 60)
    
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=False,
    )
    
    decoder = Decoder(config).to(device)
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    # Put model in eval mode to disable dropout
    decoder.eval()
    
    d_model = decoder.base.config.hidden_size
    activation = torch.randn(1, d_model, device=device)
    
    # Test 1: Check if results are deterministic in eval mode
    print("\nTest 1: Determinism in eval mode")
    
    torch.manual_seed(123)
    gen1a = decoder.generate_soft(activation.clone(), max_length=1, gumbel_tau=1.0)
    
    torch.manual_seed(123)
    gen1b = decoder.generate_soft(activation.clone(), max_length=1, gumbel_tau=1.0)
    
    print(f"  generate_soft run 1 vs run 2:")
    print(f"    Token: {gen1a.hard_token_ids[0, 0].item()} vs {gen1b.hard_token_ids[0, 0].item()}")
    print(f"    Logits diff: {(gen1a.raw_lm_logits - gen1b.raw_lm_logits).abs().max().item():.2e}")
    
    torch.manual_seed(123)
    gen2a = decoder.generate_soft_kv_cached(activation.clone(), max_length=1, gumbel_tau=1.0)
    
    torch.manual_seed(123)
    gen2b = decoder.generate_soft_kv_cached(activation.clone(), max_length=1, gumbel_tau=1.0)
    
    print(f"  generate_soft_kv_cached run 1 vs run 2:")
    print(f"    Token: {gen2a.hard_token_ids[0, 0].item()} vs {gen2b.hard_token_ids[0, 0].item()}")
    print(f"    Logits diff: {(gen2a.raw_lm_logits - gen2b.raw_lm_logits).abs().max().item():.2e}")
    
    # Test 2: Check soft token generation for multiple tokens
    print("\n\nTest 2: Soft token generation (2 tokens)")
    
    torch.manual_seed(456)
    gen_soft = decoder.generate_soft(activation.clone(), max_length=2, gumbel_tau=1.0)
    
    torch.manual_seed(456)
    gen_kv = decoder.generate_soft_kv_cached(activation.clone(), max_length=2, gumbel_tau=1.0)
    
    print(f"  Tokens from generate_soft: {gen_soft.hard_token_ids[0].tolist()}")
    print(f"  Tokens from KV cached: {gen_kv.hard_token_ids[0].tolist()}")
    print(f"  Embeddings diff: {(gen_soft.generated_text_embeddings - gen_kv.generated_text_embeddings).abs().max().item():.2e}")
    
    # Test 3: Check if the issue is in the first token or subsequent tokens
    print("\n\nTest 3: Token-by-token comparison")
    
    torch.manual_seed(789)
    gen_soft_3 = decoder.generate_soft(activation.clone(), max_length=4, gumbel_tau=1.0)
    
    torch.manual_seed(789)
    gen_kv_3 = decoder.generate_soft_kv_cached(activation.clone(), max_length=4, gumbel_tau=1.0)
    
    for i in range(4):
        t_soft = gen_soft_3.hard_token_ids[0, i].item()
        t_kv = gen_kv_3.hard_token_ids[0, i].item()
        logit_diff = (gen_soft_3.raw_lm_logits[0, i] - gen_kv_3.raw_lm_logits[0, i]).abs().max().item()
        
        print(f"  Token {i}: {t_soft} vs {t_kv} ('{tokenizer.decode([t_soft])}' vs '{tokenizer.decode([t_kv])}'), logit diff: {logit_diff:.2e}")
    
    # Test 4: Check if both methods are using Gumbel-Softmax correctly
    print("\n\nTest 4: Gumbel-Softmax temperature effect")
    
    # High temperature (more random)
    torch.manual_seed(999)
    gen_high_temp = decoder.generate_soft(activation.clone(), max_length=1, gumbel_tau=10.0)
    
    # Low temperature (more deterministic)
    torch.manual_seed(999)
    gen_low_temp = decoder.generate_soft(activation.clone(), max_length=1, gumbel_tau=0.1)
    
    print(f"  High temp (tau=10.0) token: {gen_high_temp.hard_token_ids[0, 0].item()}")
    print(f"  Low temp (tau=0.1) token: {gen_low_temp.hard_token_ids[0, 0].item()}")
    
    # Check logits distribution
    high_logits = gen_high_temp.raw_lm_logits[0, 0]
    low_logits = gen_low_temp.raw_lm_logits[0, 0]
    
    print(f"  High temp logits std: {high_logits.std().item():.2f}")
    print(f"  Low temp logits std: {low_logits.std().item():.2f}")


if __name__ == "__main__":
    debug_stochastic_and_soft()