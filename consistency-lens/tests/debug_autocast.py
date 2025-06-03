#!/usr/bin/env python3
"""Debug autocast effect on logits."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def debug_autocast():
    """Debug autocast effect."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create simple test
    print("Autocast test")
    print("=" * 60)
    
    # Test tensor
    torch.manual_seed(42)
    logits = torch.randn(1, 100, device=device) * 20 - 70  # Similar magnitude to our logits
    
    print(f"Original logits:")
    print(f"  dtype: {logits.dtype}")
    print(f"  Top 5: {logits[0].topk(5).values.tolist()}")
    
    # Without autocast
    logits_stable = logits - logits.max(dim=-1, keepdim=True)[0].detach()
    print(f"\nAfter stability subtraction (no autocast):")
    print(f"  dtype: {logits_stable.dtype}")
    print(f"  Top 5: {logits_stable[0].topk(5).values.tolist()}")
    
    # With autocast
    with torch.amp.autocast('cuda',enabled=False):
        logits_f32 = logits.float()
        logits_f32_stable = logits_f32 - logits_f32.max(dim=-1, keepdim=True)[0].detach()
        print(f"\nWith autocast(enabled=False) and float():")
        print(f"  dtype: {logits_f32_stable.dtype}")
        print(f"  Top 5: {logits_f32_stable[0].topk(5).values.tolist()}")
    
    # Now test with actual decoder
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    decoder = Decoder(DecoderConfig(
        model_name=model_name,
        use_kv_cache=True,
        use_flash_attention=False,
        base_model=False,
        projection_layer=True,
    )).to(device)
    
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    # Override the generate method to debug
    print("\n\nTesting in decoder context:")
    
    activation = torch.randn(1, 768, device=device)
    
    # Call the method but intercept
    import types
    
    original_generate = decoder.generate_soft_kv_cached
    
    def debug_generate(self, *args, **kwargs):
        # Set up as in original
        activation_input = args[0]
        B, d_model = activation_input.shape
        
        # Project activation
        emb_a = self.proj(activation_input)
        
        # Build prompt
        left_prompt_embs = self.prompt_left_emb.unsqueeze(0).expand(B, -1, -1)
        right_prompt_embs = self.prompt_right_emb.unsqueeze(0).expand(B, -1, -1)
        seq_embs = torch.cat([left_prompt_embs, emb_a.unsqueeze(1), right_prompt_embs], dim=1)
        
        # Get components
        transformer = self.base.transformer if hasattr(self.base, 'transformer') else self.base
        main_out = self.out
        
        # Process prompt
        from lens.models.kv_cache import compute_with_kv_cache
        hidden_states, kv_cache = compute_with_kv_cache(
            transformer, seq_embs, use_cache=True
        )
        
        # Get logits
        logits_t = main_out(hidden_states[:, -1])
        
        print(f"Before autocast block:")
        print(f"  Logits dtype: {logits_t.dtype}")
        print(f"  Logits top 5: {[f'{v:.3f}' for v in logits_t[0].topk(5).values.tolist()]}")
        
        # Now do what the original does
        logits_t_scaled = logits_t / 1.0
        with torch.amp.autocast('cuda',enabled=False):
            logits_f32 = logits_t_scaled.float()
            logits_f32 = logits_f32 - logits_f32.max(dim=-1, keepdim=True)[0].detach()
            
            print(f"\nInside autocast block:")
            print(f"  Logits dtype: {logits_f32.dtype}")
            print(f"  Logits top 5 (after stable): {[f'{v:.3f}' for v in logits_f32[0].topk(5).values.tolist()]}")
        
        # Return something minimal
        return type('Generated', (), {
            'raw_lm_logits': logits_t.unsqueeze(1),
            'generated_text_embeddings': seq_embs,
            'hard_token_ids': torch.zeros(1, 1, dtype=torch.long, device=device)
        })()
    
    # Monkey patch
    decoder.generate_soft_kv_cached = types.MethodType(debug_generate, decoder)
    
    result = decoder.generate_soft_kv_cached(activation)
    print(f"\nFinal stored logits: {[f'{v:.3f}' for v in result.raw_lm_logits[0, 0].topk(5).values.tolist()]}")


if __name__ == "__main__":
    debug_autocast()