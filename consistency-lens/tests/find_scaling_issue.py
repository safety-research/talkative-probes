#!/usr/bin/env python3
"""Find where the scaling/transformation happens."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def find_scaling():
    """Find where logits get transformed."""
    
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
    
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    print("Finding logit transformation")
    print("=" * 60)
    
    # Monkey patch to trace the generation more closely
    import types
    
    original_generate = decoder.generate_soft_kv_cached
    
    def traced_generate(self, activation_input, max_length=64, gumbel_tau=1.0, **kwargs):
        print("\nInside generate_soft_kv_cached")
        
        # Setup
        B, d_model = activation_input.shape
        main_model = self
        main_base = self.base
        main_out = self.out
        
        # Get embeddings
        input_emb_table = main_base.get_input_embeddings().weight
        
        # Project activation
        emb_a = self.proj(activation_input)
        
        # Build prompt
        left_prompt_embs = self.prompt_left_emb.unsqueeze(0).expand(B, -1, -1)
        right_prompt_embs = self.prompt_right_emb.unsqueeze(0).expand(B, -1, -1)
        seq_embs = torch.cat([left_prompt_embs, emb_a.unsqueeze(1), right_prompt_embs], dim=1)
        
        # Get transformer
        transformer = main_base.transformer if hasattr(main_base, 'transformer') else main_base
        
        # Process initial sequence
        from lens.models.kv_cache import compute_with_kv_cache
        hidden_states, kv_cache = compute_with_kv_cache(
            transformer, seq_embs, position_offset=0
        )
        
        print(f"Hidden states shape: {hidden_states.shape}")
        print(f"Hidden states norm: {hidden_states.norm():.3f}")
        
        # Get logits for the last position
        logits_t = main_out(hidden_states[:, -1])  # (B, V)
        
        print(f"\nLogits right after main_out:")
        print(f"  Shape: {logits_t.shape}")
        print(f"  Norm: {logits_t.norm():.3f}")
        print(f"  Top 5: {logits_t[0].topk(5).values.tolist()[:5]}")
        print(f"  Bottom 5: {logits_t[0].topk(5, largest=False).values.tolist()[:5]}")
        
        # Check the scaling
        logits_t_scaled = logits_t / 1.0  # T_sampling = 1.0
        print(f"\nAfter scaling by 1.0:")
        print(f"  Top 5: {logits_t_scaled[0].topk(5).values.tolist()[:5]}")
        
        # Continue with one step of generation
        logits_list = []
        
        # The autocast block
        with torch.amp.autocast('cuda',enabled=False):
            logits_f32 = logits_t_scaled.float()
            print(f"\nInside autocast block (float32):")
            print(f"  Top 5: {logits_f32[0].topk(5).values.tolist()[:5]}")
            
            # Subtract max for numerical stability
            logits_f32_stable = logits_f32 - logits_f32.max(dim=-1, keepdim=True)[0].detach()
            print(f"\nAfter stability subtraction:")
            print(f"  Top 5: {logits_f32_stable[0].topk(5).values.tolist()[:5]}")
            print(f"  Range: [{logits_f32_stable.min():.3f}, {logits_f32_stable.max():.3f}]")
        
        # What gets stored
        logits_list.append(logits_t)
        print(f"\nWhat gets stored in logits_list:")
        print(f"  Top 5: {logits_t[0].topk(5).values.tolist()[:5]}")
        
        # Create minimal return
        return type('Generated', (), {
            'raw_lm_logits': torch.stack([logits_t], dim=1),
            'generated_text_embeddings': seq_embs,
            'hard_token_ids': torch.zeros(1, 1, dtype=torch.long, device=device)
        })()
    
    # Apply patch
    decoder.generate_soft_kv_cached = types.MethodType(traced_generate, decoder)
    
    # Test
    torch.manual_seed(42)
    activation = torch.randn(1, 768, device=device)
    
    with torch.no_grad():
        gen = decoder.generate_soft_kv_cached(activation, max_length=1, gumbel_tau=0.0)
    
    print(f"\n\nFinal returned logits:")
    print(f"  Shape: {gen.raw_lm_logits.shape}")
    print(f"  Top 5: {gen.raw_lm_logits[0, 0].topk(5).values.tolist()[:5]}")
    
    # Now test with the original method
    print("\n\n" + "="*60)
    print("Testing with original unpatched method:")
    
    # Create fresh decoder
    decoder2 = Decoder(DecoderConfig(
        model_name=model_name,
        use_kv_cache=True,
        use_flash_attention=False,
        base_model=False,
        projection_layer=True,
    )).to(device)
    
    # decoder2.load_state_dict(decoder.state_dict())  # Skip this, causes issues with prompt
    decoder2.set_prompt("explain <embed>:", tokenizer)
    
    with torch.no_grad():
        gen2 = decoder2.generate_soft_kv_cached(activation.clone(), max_length=1, gumbel_tau=0.0)
    
    print(f"Original method logits top 5: {gen2.raw_lm_logits[0, 0].topk(5).values.tolist()[:5]}")


if __name__ == "__main__":
    find_scaling()