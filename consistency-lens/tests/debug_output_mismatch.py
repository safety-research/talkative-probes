#!/usr/bin/env python3
"""Debug why outputs don't match between generate_soft and generate_soft_chkpt."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def debug_output_mismatch():
    """Find why outputs differ between methods."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Debugging Output Mismatch")
    print("=" * 60)
    
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=False,
    )
    
    # Create two identical decoders
    torch.manual_seed(42)
    decoder1 = Decoder(config).to(device)
    decoder1.set_prompt("explain <embed>:", tokenizer)
    
    torch.manual_seed(42)
    decoder2 = Decoder(config).to(device)
    decoder2.set_prompt("explain <embed>:", tokenizer)
    
    # Verify they have identical weights
    for (n1, p1), (n2, p2) in zip(decoder1.named_parameters(), decoder2.named_parameters()):
        if not torch.equal(p1, p2):
            print(f"Parameters {n1} differ!")
            break
    else:
        print("✓ All parameters identical")
    
    d_model = decoder1.base.config.hidden_size
    batch_size = 1
    seq_length = 2  # Very short for debugging
    
    # Create identical activations
    torch.manual_seed(123)
    activation = torch.randn(batch_size, d_model, device=device)
    
    # Set same random seed for generation
    print("\nGenerating with same random seed...")
    torch.manual_seed(456)
    torch.cuda.manual_seed_all(456)
    gen1 = decoder1.generate_soft(activation.clone(), max_length=seq_length, gumbel_tau=1.0)
    
    torch.manual_seed(456)
    torch.cuda.manual_seed_all(456)
    gen2 = decoder2.generate_soft_chkpt(activation.clone(), max_length=seq_length, gumbel_tau=1.0)
    
    # Compare step by step
    print(f"\nComparison (seq_length={seq_length}):")
    
    # Embeddings
    for t in range(seq_length):
        emb1 = gen1.generated_text_embeddings[0, t]
        emb2 = gen2.generated_text_embeddings[0, t]
        diff = (emb1 - emb2).abs().max().item()
        print(f"\n  Time step {t}:")
        print(f"    Embedding diff: {diff:.2e}")
        print(f"    Embedding norm 1: {emb1.norm().item():.4f}")
        print(f"    Embedding norm 2: {emb2.norm().item():.4f}")
        
        # Logits
        logits1 = gen1.raw_lm_logits[0, t]
        logits2 = gen2.raw_lm_logits[0, t]
        logit_diff = (logits1 - logits2).abs().max().item()
        print(f"    Logit diff: {logit_diff:.2e}")
        
        # Hard IDs
        id1 = gen1.hard_token_ids[0, t].item()
        id2 = gen2.hard_token_ids[0, t].item()
        print(f"    Token 1: {id1} ('{tokenizer.decode([id1])}')")
        print(f"    Token 2: {id2} ('{tokenizer.decode([id2])}')")
        
        if diff > 1e-5:
            print(f"    → Divergence at time step {t}!")
            break
    
    # Test with tau=0 (no randomness)
    print("\n\nTesting with tau=0 (no Gumbel noise):")
    gen1_det = decoder1.generate_soft(activation.clone(), max_length=seq_length, gumbel_tau=0.0)
    gen2_det = decoder2.generate_soft_chkpt(activation.clone(), max_length=seq_length, gumbel_tau=0.0)
    
    emb_diff_det = (gen1_det.generated_text_embeddings - gen2_det.generated_text_embeddings).abs().max().item()
    ids_same_det = torch.equal(gen1_det.hard_token_ids, gen2_det.hard_token_ids)
    
    print(f"  Embedding diff: {emb_diff_det:.2e}")
    print(f"  IDs identical: {ids_same_det}")
    
    if emb_diff_det > 1e-5:
        print("  ✗ Still differs without randomness - issue is deterministic!")
    else:
        print("  ✓ Matches without randomness - issue is with random seed handling")


def test_checkpoint_every_n():
    """Test different checkpoint frequencies."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\n\nTesting Different Checkpoint Frequencies")
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
    
    d_model = decoder.base.config.hidden_size
    activation = torch.randn(1, d_model, device=device)
    seq_length = 8
    
    # Reference
    torch.manual_seed(42)
    gen_ref = decoder.generate_soft(activation.clone(), max_length=seq_length, gumbel_tau=1.0)
    
    # Test different checkpoint frequencies
    for freq in [1, 2, 4, 8]:
        print(f"\n  Checkpoint every {freq} tokens:")
        torch.manual_seed(42)
        gen_chk = decoder.generate_soft_chkpt(
            activation.clone(), 
            max_length=seq_length, 
            gumbel_tau=1.0,
            checkpoint_every_n_tokens=freq
        )
        
        emb_diff = (gen_ref.generated_text_embeddings - gen_chk.generated_text_embeddings).abs().max().item()
        ids_same = torch.equal(gen_ref.hard_token_ids, gen_chk.hard_token_ids)
        
        print(f"    Embedding diff: {emb_diff:.2e}")
        print(f"    IDs identical: {ids_same}")


if __name__ == "__main__":
    debug_output_mismatch()
    test_checkpoint_every_n()