#!/usr/bin/env python3
"""Debug why layer 11 has zero gradient."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def debug_layer_11():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    batch_size = 1
    d_model = 768
    seq_length = 4
    
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=True,
    )
    
    decoder = Decoder(config).to(device)
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    # Check prompt structure
    print(f"Prompt text: {decoder.prompt_text}")
    if decoder.prompt_left_emb is not None:
        print(f"Left prompt length: {decoder.prompt_left_emb.size(0)}")
    else:
        print("No left prompt")
    if decoder.prompt_right_emb is not None:
        print(f"Right prompt length: {decoder.prompt_right_emb.size(0)}")
    else:
        print("No right prompt")
    
    # The embed position would be after the left prompt
    embed_pos = decoder.prompt_left_emb.size(0) if decoder.prompt_left_emb is not None else 0
    print(f"Embed position: {embed_pos}")
    
    # Create activation and generate
    activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    gen = decoder.generate_soft(activation, max_length=seq_length, gumbel_tau=1.0)
    
    print(f"\nGenerated embeddings shape: {gen.generated_text_embeddings.shape}")
    print(f"Sequence parts: left_prompt({decoder.prompt_left_emb.size(0) if decoder.prompt_left_emb is not None else 0}) + embed(1) + right_prompt({decoder.prompt_right_emb.size(0) if decoder.prompt_right_emb is not None else 0}) + generated({seq_length})")
    
    # Total sequence length during generation
    prompt_len = (decoder.prompt_left_emb.size(0) if decoder.prompt_left_emb is not None else 0) + \
                 1 + \
                 (decoder.prompt_right_emb.size(0) if decoder.prompt_right_emb is not None else 0)
    total_seq_len = prompt_len + seq_length
    print(f"Total sequence length during generation: {total_seq_len}")
    print(f"Position used for next token prediction: -1 (position {total_seq_len-1})")
    
    # Compute different losses to see which positions matter
    print("\nTesting different loss functions:")
    
    # Loss 1: Sum all positions (what we use in tests)
    decoder.zero_grad()
    activation.grad = None
    loss1 = gen.generated_text_embeddings.sum()
    loss1.backward()
    grad1_norm = decoder.proj_weight.grad[11].norm().item()
    print(f"Loss = sum all positions: Layer 11 grad norm = {grad1_norm:.4f}")
    
    # Loss 2: Only last position
    decoder.zero_grad()
    activation.grad = None
    gen2 = decoder.generate_soft(activation, max_length=seq_length, gumbel_tau=1.0)
    loss2 = gen2.generated_text_embeddings[:, -1].sum()
    loss2.backward()
    grad2_norm = decoder.proj_weight.grad[11].norm().item()
    print(f"Loss = last position only: Layer 11 grad norm = {grad2_norm:.4f}")
    
    # Loss 3: Only first position
    decoder.zero_grad()
    activation.grad = None
    gen3 = decoder.generate_soft(activation, max_length=seq_length, gumbel_tau=1.0)
    loss3 = gen3.generated_text_embeddings[:, 0].sum()
    loss3.backward()
    grad3_norm = decoder.proj_weight.grad[11].norm().item()
    print(f"Loss = first position only: Layer 11 grad norm = {grad3_norm:.4f}")


if __name__ == "__main__":
    debug_layer_11()