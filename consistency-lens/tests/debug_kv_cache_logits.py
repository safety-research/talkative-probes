#!/usr/bin/env python3
"""Debug why KV cache produces different logits."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.kv_cache import compute_with_kv_cache, KVCache


def debug_kv_cache():
    """Debug KV cache logits issue."""
    
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
    
    print("Debugging KV cache logits issue")
    print("=" * 60)
    
    # Get components
    B = 1
    emb_a = decoder.proj(activation)
    left_prompt_embs = decoder.prompt_left_emb.unsqueeze(0).expand(B, -1, -1)
    right_prompt_embs = decoder.prompt_right_emb.unsqueeze(0).expand(B, -1, -1)
    seq_embs = torch.cat([left_prompt_embs, emb_a.unsqueeze(1), right_prompt_embs], dim=1)
    
    print(f"Sequence embeddings shape: {seq_embs.shape}")
    print(f"Sequence length: {seq_embs.shape[1]}")
    
    # Method 1: Manual computation (correct)
    print("\n1. Manual computation:")
    transformer = decoder.base.transformer if hasattr(decoder.base, 'transformer') else decoder.base
    outputs = transformer(inputs_embeds=seq_embs)
    hidden_manual = outputs.last_hidden_state
    logits_manual = decoder.out(hidden_manual[:, -1])
    print(f"   Hidden shape: {hidden_manual.shape}")
    print(f"   Hidden last pos norm: {hidden_manual[:, -1].norm().item():.3f}")
    print(f"   Logits top 5: {[f'{v:.3f}' for v in logits_manual[0].topk(5).values.tolist()]}")
    
    # Method 2: KV cache computation 
    print("\n2. KV cache computation:")
    kv_cache = KVCache()
    hidden_kv, kv_cache = compute_with_kv_cache(
        transformer, seq_embs, kv_cache, position_offset=0
    )
    logits_kv = decoder.out(hidden_kv[:, -1])
    print(f"   Hidden shape: {hidden_kv.shape}")
    print(f"   Hidden last pos norm: {hidden_kv[:, -1].norm().item():.3f}")
    print(f"   Logits top 5: {[f'{v:.3f}' for v in logits_kv[0].topk(5).values.tolist()]}")
    
    # Compare hidden states
    print("\n3. Comparing hidden states:")
    diff = (hidden_manual - hidden_kv).abs()
    print(f"   Max diff: {diff.max().item():.6f}")
    print(f"   Mean diff: {diff.mean().item():.6f}")
    print(f"   Diff at last pos: {diff[:, -1].max().item():.6f}")
    
    # Check if the issue is in the output layer
    print("\n4. Output layer check:")
    print(f"   decoder.out weight shape: {decoder.out.weight.shape}")
    print(f"   decoder.out weight norm: {decoder.out.weight.norm().item():.3f}")
    
    # Test with a simple tensor
    test_hidden = torch.randn(1, 768, device=device)
    test_logits = decoder.out(test_hidden)
    print(f"   Test hidden norm: {test_hidden.norm().item():.3f}")
    print(f"   Test logits range: [{test_logits.min().item():.3f}, {test_logits.max().item():.3f}]")
    
    # Check if the hidden states are actually different
    print("\n5. Hidden state details:")
    print(f"   Manual last hidden: {hidden_manual[0, -1, :5].tolist()}")
    print(f"   KV cache last hidden: {hidden_kv[0, -1, :5].tolist()}")
    
    # Check intermediate steps in KV cache
    print("\n6. Checking position embeddings:")
    position_ids_manual = torch.arange(seq_embs.shape[1], dtype=torch.long, device=device)
    position_embeds_manual = transformer.wpe(position_ids_manual)
    print(f"   Position embeds shape: {position_embeds_manual.shape}")
    print(f"   Position embeds norm: {position_embeds_manual.norm().item():.3f}")


if __name__ == "__main__":
    debug_kv_cache()