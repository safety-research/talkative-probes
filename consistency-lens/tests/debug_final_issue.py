#!/usr/bin/env python3
"""Final debugging of KV cache with multi-layer patching."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def test_first_token_only():
    """Test just the first token generation."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Testing First Token Generation Only")
    print("=" * 60)
    
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=False,
    )
    
    decoder1 = Decoder(config).to(device)
    decoder1.set_prompt("explain <embed>:", tokenizer)
    
    decoder2 = Decoder(config).to(device)
    decoder2.set_prompt("explain <embed>:", tokenizer)
    decoder2.load_state_dict(decoder1.state_dict())
    
    d_model = decoder1.base.config.hidden_size
    torch.manual_seed(42)
    activation = torch.randn(1, d_model, device=device, requires_grad=True)
    activation2 = activation.detach().clone().requires_grad_(True)
    
    # Check sequences first
    print("\nChecking initial sequences:")
    
    # Build sequences manually to debug
    parts1 = []
    parts2 = []
    
    if decoder1.prompt_left_emb is not None:
        parts1.append(decoder1.prompt_left_emb.expand(1, -1, -1))
        parts2.append(decoder2.prompt_left_emb.expand(1, -1, -1))
    
    # Both should add activation
    a_proj1 = decoder1._apply_projection(activation).unsqueeze(1)
    a_proj2 = decoder2._apply_projection(activation2).unsqueeze(1)
    parts1.append(a_proj1)
    parts2.append(a_proj2)
    
    if decoder1.prompt_right_emb is not None:
        parts1.append(decoder1.prompt_right_emb.expand(1, -1, -1))
        parts2.append(decoder2.prompt_right_emb.expand(1, -1, -1))
    
    seq1 = torch.cat(parts1, dim=1)
    seq2 = torch.cat(parts2, dim=1)
    
    print(f"  Seq1 shape: {seq1.shape}")
    print(f"  Seq2 shape: {seq2.shape}")
    print(f"  Shapes match: {seq1.shape == seq2.shape}")
    
    # Generate just 1 token
    print("\nGenerating 1 token:")
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    gen1 = decoder1.generate_soft(activation, max_length=1, gumbel_tau=1.0)
    
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    gen2 = decoder2.generate_soft_kv_cached(activation2, max_length=1, gumbel_tau=1.0)
    
    # Compare outputs
    emb_diff = (gen1.generated_text_embeddings - gen2.generated_text_embeddings).abs().max().item()
    logits_diff = (gen1.raw_lm_logits - gen2.raw_lm_logits).abs().max().item()
    ids_same = torch.equal(gen1.hard_token_ids, gen2.hard_token_ids)
    
    print(f"\nResults:")
    print(f"  Embeddings max diff: {emb_diff:.2e}")
    print(f"  Logits max diff: {logits_diff:.2e}")
    print(f"  Hard IDs identical: {ids_same}")
    print(f"  Token 1: {gen1.hard_token_ids[0, 0].item()} vs {gen2.hard_token_ids[0, 0].item()}")
    
    # Test gradients
    loss1 = gen1.generated_text_embeddings.sum()
    loss2 = gen2.generated_text_embeddings.sum()
    
    loss1.backward()
    loss2.backward()
    
    grad_diff = (activation.grad - activation2.grad).abs().max().item()
    print(f"  Activation grad diff: {grad_diff:.2e}")
    
    # Check LLaMA too
    print("\n\nTesting LLaMA:")
    model_name = "SimpleStories/SimpleStories-5M"
    
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=False,
    )
    
    decoder3 = Decoder(config).to(device)
    decoder3.set_prompt("explain <embed>:", tokenizer)
    
    decoder4 = Decoder(config).to(device)
    decoder4.set_prompt("explain <embed>:", tokenizer)
    decoder4.load_state_dict(decoder3.state_dict())
    
    d_model = decoder3.base.config.hidden_size
    torch.manual_seed(42)
    activation3 = torch.randn(1, d_model, device=device)
    
    # Generate
    torch.manual_seed(123)
    gen3 = decoder3.generate_soft(activation3, max_length=1, gumbel_tau=1.0)
    
    torch.manual_seed(123)
    gen4 = decoder4.generate_soft_kv_cached(activation3, max_length=1, gumbel_tau=1.0)
    
    # Compare
    emb_diff = (gen3.generated_text_embeddings - gen4.generated_text_embeddings).abs().max().item()
    logits_diff = (gen3.raw_lm_logits - gen4.raw_lm_logits).abs().max().item()
    
    print(f"  Embeddings max diff: {emb_diff:.2e}")
    print(f"  Logits max diff: {logits_diff:.2e}")
    
    if logits_diff < 1e-4:
        print("  ✓ LLaMA works correctly!")
    else:
        print("  ✗ LLaMA also has issues")


if __name__ == "__main__":
    test_first_token_only()