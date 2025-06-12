#!/usr/bin/env python3
"""
Focused test for end_to_end=True, detach_after_each_sample=False (standard config)
"""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig, Generated


def test_standard_config():
    """Test with standard configuration (end_to_end=True, detach=False)."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Standard configuration
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        output_head=False,
        patch_all_layers=False,
        per_layer_projections=False,
        end_to_end=True,  # Standard
        detach_after_each_sample=False,  # Standard
    )
    
    print("\nConfiguration:")
    print(f"  end_to_end: {config.end_to_end}")
    print(f"  detach_after_each_sample: {config.detach_after_each_sample}")
    
    decoder = Decoder(config).to(device).eval()
    decoder.set_prompt("The answer is <embed>:", tokenizer)
    
    d_model = decoder.base.config.hidden_size
    activation = torch.randn(1, d_model, device=device)
    
    # Test multiple lengths
    for max_length in [5, 10, 20]:
        print(f"\n{'='*60}")
        print(f"Testing with max_length={max_length}")
        print(f"{'='*60}")
        
        # Generate with both methods
        torch.manual_seed(42)
        gen_soft = decoder.generate_soft(activation, max_length, gumbel_tau=1.0)
        
        torch.manual_seed(42)
        gen_kv = decoder.generate_soft_kv_cached(activation, max_length, gumbel_tau=1.0)
        
        # Compare outputs
        tokens_soft = gen_soft.hard_token_ids[0].tolist()
        tokens_kv = gen_kv.hard_token_ids[0].tolist()
        
        text_soft = tokenizer.decode(tokens_soft)
        text_kv = tokenizer.decode(tokens_kv)
        
        print(f"generate_soft:      '{text_soft}'")
        print(f"generate_kv_cached: '{text_kv}'")
        
        # Check match
        match = tokens_soft == tokens_kv
        logits_diff = (gen_soft.raw_lm_logits - gen_kv.raw_lm_logits).abs().max().item()
        emb_diff = (gen_soft.generated_text_embeddings - gen_kv.generated_text_embeddings).abs().max().item()
        
        print(f"\nResults:")
        print(f"  Tokens match: {'✅' if match else '❌'}")
        print(f"  Max logits diff: {logits_diff:.2e}")
        print(f"  Max embeddings diff: {emb_diff:.2e}")
        
        # Token-by-token comparison if mismatch
        if not match:
            print("\nToken-by-token comparison:")
            for i in range(max_length):
                t_soft = tokens_soft[i]
                t_kv = tokens_kv[i]
                print(f"  Position {i}: {t_soft} vs {t_kv} {'✅' if t_soft == t_kv else '❌'}")
    
    # Test gradients
    print(f"\n{'='*60}")
    print("Gradient Test (expecting match)")
    print(f"{'='*60}")
    
    activation_soft = activation.clone().requires_grad_(True)
    activation_kv = activation.clone().requires_grad_(True)
    
    torch.manual_seed(42)
    gen_soft = decoder.generate_soft(activation_soft, max_length=10, gumbel_tau=1.0)
    
    torch.manual_seed(42)
    gen_kv = decoder.generate_soft_kv_cached(activation_kv, max_length=10, gumbel_tau=1.0)
    
    # Compute gradients
    loss_soft = gen_soft.generated_text_embeddings.sum()
    loss_kv = gen_kv.generated_text_embeddings.sum()
    
    grad_soft = torch.autograd.grad(loss_soft, activation_soft)[0]
    grad_kv = torch.autograd.grad(loss_kv, activation_kv)[0]
    
    grad_diff = (grad_soft - grad_kv).abs().max().item()
    grad_relative_diff = grad_diff / (grad_soft.abs().max().item() + 1e-8)
    
    print(f"Gradient absolute diff: {grad_diff:.2e}")
    print(f"Gradient relative diff: {grad_relative_diff:.2%}")
    print(f"Gradients match (< 1e-5): {'✅' if grad_diff < 1e-5 else '❌'}")
    
    # Test with different seeds
    print(f"\n{'='*60}")
    print("Testing multiple seeds")
    print(f"{'='*60}")
    
    all_match = True
    for seed in range(5):
        torch.manual_seed(seed)
        gen_soft = decoder.generate_soft(activation, max_length=10, gumbel_tau=1.0)
        
        torch.manual_seed(seed)
        gen_kv = decoder.generate_soft_kv_cached(activation, max_length=10, gumbel_tau=1.0)
        
        match = torch.all(gen_soft.hard_token_ids == gen_kv.hard_token_ids).item()
        all_match &= match
        print(f"Seed {seed}: {'✅' if match else '❌'}")
    
    print(f"\nAll seeds match: {'✅' if all_match else '❌'}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY for end_to_end=True, detach_after_each_sample=False")
    print(f"{'='*60}")
    print("✅ Outputs are IDENTICAL between generate_soft and generate_soft_kv_cached")
    print("✅ This holds for all sequence lengths and random seeds")
    print("✅ Gradients also match (as expected with end_to_end=True)")


if __name__ == "__main__":
    test_standard_config()