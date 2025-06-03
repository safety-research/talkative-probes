#!/usr/bin/env python3
"""Debug incremental generation shape issue."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def trace_kv_generation():
    """Trace through KV generation to find shape issue."""
    
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
    activation = torch.randn(1, d_model, device=device)
    
    print("Manually tracing KV generation...")
    print("=" * 60)
    
    # Try to generate tokens to see where it fails
    for num_tokens in [1, 2, 8]:
        print(f"\nTrying {num_tokens} tokens...")
        try:
            torch.manual_seed(123)
            gen = decoder.generate_soft_kv_cached(activation, max_length=num_tokens, gumbel_tau=1.0)
            print(f"Success with {num_tokens} tokens!")
        except Exception as e:
            print(f"Failed with {num_tokens} tokens: {e}")
            if num_tokens == 2:
                # Debug the 2-token case
                import traceback
                traceback.print_exc()
            break
    
    return
    
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Let's trace manually
        print("\nManual trace:")
        
        # Setup
        from lens.models.kv_cache import KVCache, GPT2AttentionWithCache
        
        parts = []
        if decoder.prompt_left_emb is not None:
            parts.append(decoder.prompt_left_emb.expand(1, -1, -1))
        a_proj = decoder._apply_projection(activation).unsqueeze(1)
        parts.append(a_proj)
        if decoder.prompt_right_emb is not None:
            parts.append(decoder.prompt_right_emb.expand(1, -1, -1))
        seq_embs = torch.cat(parts, dim=1)
        
        print(f"Initial seq_embs shape: {seq_embs.shape}")
        
        kv_cache = KVCache()
        transformer = decoder.base.transformer
        layers = transformer.h
        
        hidden_states = seq_embs
        seq_length = hidden_states.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(1, -1)
        
        # Add position embeddings
        position_embeds = transformer.wpe(position_ids.squeeze(0))
        print(f"Position embeds shape: {position_embeds.shape}")
        
        # Check what happens when we add
        print(f"Hidden states shape: {hidden_states.shape}")
        print(f"Position embeds unsqueeze shape: {position_embeds.unsqueeze(0).shape}")
        
        result = hidden_states + position_embeds.unsqueeze(0)
        print(f"Addition result shape: {result.shape}")
        
        # But what if position_embeds is already the wrong shape?
        print(f"\nChecking wpe call:")
        print(f"position_ids: {position_ids}")
        print(f"position_ids.squeeze(0): {position_ids.squeeze(0)}")
        print(f"position_ids.squeeze(0).shape: {position_ids.squeeze(0).shape}")
        
        # Try calling wpe directly
        pos_emb_direct = transformer.wpe(position_ids.squeeze(0))
        print(f"Direct wpe result shape: {pos_emb_direct.shape}")


if __name__ == "__main__":
    trace_kv_generation()