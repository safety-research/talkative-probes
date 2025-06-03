#!/usr/bin/env python3
"""Verify sequence order is the issue."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def verify_sequence_order():
    """Verify that sequence order is causing the issue."""
    
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
    
    # Set prompt
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    # Test activation
    torch.manual_seed(42)
    activation = torch.randn(1, 768, device=device)
    
    print("Verifying sequence order issue")
    print("=" * 60)
    
    # Get components
    B = 1
    emb_a = decoder.proj(activation)
    left_prompt = decoder.prompt_left_emb.unsqueeze(0).expand(B, -1, -1)
    right_prompt = decoder.prompt_right_emb.unsqueeze(0).expand(B, -1, -1)
    
    print(f"Left prompt shape: {left_prompt.shape}")
    print(f"Activation shape: {emb_a.unsqueeze(1).shape}")
    print(f"Right prompt shape: {right_prompt.shape}")
    
    # Correct order: [left, activation, right]
    seq_correct = torch.cat([left_prompt, emb_a.unsqueeze(1), right_prompt], dim=1)
    
    # Wrong order (what generate_soft_kv_cached does): [left, right, activation]
    seq_wrong = torch.cat([left_prompt, right_prompt, emb_a.unsqueeze(1)], dim=1)
    
    # Get transformer
    transformer = decoder.base.transformer if hasattr(decoder.base, 'transformer') else decoder.base
    
    # Process both sequences
    print("\n1. Correct order [left, activation, right]:")
    outputs_correct = transformer(inputs_embeds=seq_correct)
    hidden_correct = outputs_correct.last_hidden_state
    logits_correct = decoder.out(hidden_correct[:, -1])
    print(f"   Logits top 5: {[f'{v:.3f}' for v in logits_correct[0].topk(5).values.tolist()]}")
    
    print("\n2. Wrong order [left, right, activation]:")
    outputs_wrong = transformer(inputs_embeds=seq_wrong)
    hidden_wrong = outputs_wrong.last_hidden_state
    logits_wrong = decoder.out(hidden_wrong[:, -1])
    print(f"   Logits top 5: {[f'{v:.3f}' for v in logits_wrong[0].topk(5).values.tolist()]}")
    
    print("\n3. The prompt text:")
    print(f"   Prompt: '{decoder.prompt_text}'")
    print(f"   Left tokens: {decoder.prompt_ids[:len(decoder.prompt_left_emb)].tolist() if decoder.prompt_left_emb is not None else 'None'}")
    print(f"   Right tokens: {decoder.prompt_ids[-len(decoder.prompt_right_emb):].tolist() if decoder.prompt_right_emb is not None else 'None'}")
    
    # Decode tokens to verify
    if hasattr(decoder, 'prompt_ids') and len(decoder.prompt_ids) > 0:
        left_len = len(decoder.prompt_left_emb) if decoder.prompt_left_emb is not None else 0
        right_len = len(decoder.prompt_right_emb) if decoder.prompt_right_emb is not None else 0
        if left_len > 0:
            left_text = tokenizer.decode(decoder.prompt_ids[:left_len])
            print(f"   Left text: '{left_text}'")
        if right_len > 0:
            right_text = tokenizer.decode(decoder.prompt_ids[-right_len:])
            print(f"   Right text: '{right_text}'")


if __name__ == "__main__":
    verify_sequence_order()