#!/usr/bin/env python3
"""
Debug fwd_tokens issue.
"""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def debug_fwd_tokens():
    """Debug the fwd_tokens shape issue."""
    
    device = torch.device("cuda")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("="*80)
    print("DEBUGGING FWD_TOKENS")
    print("="*80)
    
    # Create decoder with output_head=True
    decoder = Decoder(DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        output_head=True,
        patch_all_layers=False,
        per_layer_projections=False,
    )).to(device).eval()
    
    decoder.set_prompt("The answer is <embed>:", tokenizer)
    
    # Check decoder components
    print(f"\nDecoder components:")
    print(f"  Base model: {decoder.base.__class__.__name__}")
    print(f"  Hidden size: {decoder.base.config.hidden_size}")
    print(f"  Vocab size: {decoder.base.config.vocab_size}")
    print(f"  Output head shape: {decoder.out.weight.shape}")
    
    # Create simple test
    d_model = decoder.base.config.hidden_size
    activation = torch.randn(1, d_model, device=device)
    test_tokens = torch.tensor([1, 2, 3, 4, 5], device=device)  # 5 tokens
    
    print(f"\nTest inputs:")
    print(f"  Activation shape: {activation.shape}")
    print(f"  Test tokens shape: {test_tokens.shape}")
    
    # Manually trace through fwd_tokens to find the issue
    print(f"\nTracing through fwd_tokens:")
    
    # Get embedding tables
    input_emb_table = decoder.base.get_input_embeddings().weight
    output_emb_table = decoder.base.get_output_embeddings().weight
    print(f"  Input embedding table shape: {input_emb_table.shape}")
    print(f"  Output embedding table shape: {output_emb_table.shape}")
    
    # Check if embeddings are tied
    embeddings_tied = (input_emb_table.data_ptr() == output_emb_table.data_ptr())
    print(f"  Embeddings tied: {embeddings_tied}")
    
    # Prepare sequence
    parts = []
    if decoder.prompt_left_emb is not None:
        parts.append(decoder.prompt_left_emb.unsqueeze(0))
        print(f"  Prompt left shape: {decoder.prompt_left_emb.shape}")
    
    # Apply projection to activation
    a_proj = decoder._apply_projection(activation).unsqueeze(1)
    parts.append(a_proj)
    print(f"  Projected activation shape: {a_proj.shape}")
    
    if decoder.prompt_right_emb is not None:
        parts.append(decoder.prompt_right_emb.unsqueeze(0))
        print(f"  Prompt right shape: {decoder.prompt_right_emb.shape}")
    
    # Embed test tokens
    test_tokens_expanded = test_tokens.unsqueeze(0)  # Add batch dimension
    token_embs = input_emb_table[test_tokens_expanded]
    parts.append(token_embs)
    print(f"  Token embeddings shape: {token_embs.shape}")
    
    seq_embs = torch.cat(parts, dim=1)
    print(f"  Full sequence embeddings shape: {seq_embs.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = decoder.base(inputs_embeds=seq_embs)
    
    hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
    print(f"  Hidden states shape after forward: {hidden_states.shape}")
    
    # Check what main_out expects
    print(f"\nChecking main_out (decoder.out):")
    print(f"  Type: {type(decoder.out)}")
    print(f"  Weight shape: {decoder.out.weight.shape}")
    print(f"  Expected input shape: (*, {decoder.out.weight.shape[1]})")
    print(f"  Expected output shape: (*, {decoder.out.weight.shape[0]})")
    
    # Test main_out with correct shape
    try:
        # Try with last hidden state only
        test_logits = decoder.out(hidden_states[:, -1, :])
        print(f"  ✅ main_out(hidden_states[:, -1, :]) works! Output shape: {test_logits.shape}")
    except Exception as e:
        print(f"  ❌ main_out(hidden_states[:, -1, :]) failed: {e}")
    
    try:
        # Try with full hidden states
        test_logits = decoder.out(hidden_states)
        print(f"  ✅ main_out(hidden_states) works! Output shape: {test_logits.shape}")
    except Exception as e:
        print(f"  ❌ main_out(hidden_states) failed: {e}")
    
    # Now try actual fwd_tokens
    print(f"\nTrying actual fwd_tokens:")
    try:
        probs, entropies = decoder.fwd_tokens(
            activation_input=activation,
            use_projection=True,
            input_tokens=test_tokens
        )
        print(f"  ✅ fwd_tokens succeeded!")
        print(f"  Probabilities shape: {probs.shape}")
        print(f"  Entropies shape: {entropies.shape if entropies is not None else 'None'}")
    except Exception as e:
        print(f"  ❌ fwd_tokens failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_fwd_tokens()