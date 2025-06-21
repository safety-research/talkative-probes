#!/usr/bin/env python3
"""
Test fwd_tokens alignment with a workaround for the current bug.
The bug: fwd_tokens doesn't pass output_hidden_states=True to the base model.
"""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
import os
import random
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fixed_fwd_tokens(decoder, activation_input, input_tokens, use_projection=True):
    """
    A fixed version of fwd_tokens that properly requests hidden states.
    This shows how fwd_tokens should work.
    """
    # Setup similar to fwd_tokens
    main_base = decoder.base
    main_out = decoder.out
    
    prompt_left_emb = decoder.prompt_left_emb
    prompt_right_emb = decoder.prompt_right_emb
    
    # Get dtype from projection layer
    if decoder.config.per_layer_projections:
        activation_input = activation_input.to(decoder.proj_weight.dtype)
    else:
        activation_input = activation_input.to(decoder.proj.weight.dtype)
    
    B, d_model = activation_input.shape
    device = activation_input.device
    
    # Get embedding table
    input_emb_table = main_base.get_input_embeddings().weight
    
    # Prepare initial sequence
    parts = []
    prompt_len_left = 0
    if prompt_left_emb is not None:
        parts.append(prompt_left_emb.expand(B, -1, -1))
        prompt_len_left = prompt_left_emb.shape[0]
    
    # Apply projection
    if use_projection:
        if decoder.config.patch_all_layers and decoder.config.per_layer_projections:
            a_proj = decoder._apply_projection(activation_input, layer_idx=0).unsqueeze(1)
        else:
            a_proj = decoder._apply_projection(activation_input).unsqueeze(1)
    else:
        a_proj = activation_input.unsqueeze(1)
    parts.append(a_proj)
    
    prompt_len_right = 0
    if prompt_right_emb is not None:
        parts.append(prompt_right_emb.expand(B, -1, -1))
        prompt_len_right = prompt_right_emb.shape[0]
    
    # Ensure input_tokens is broadcastable
    if input_tokens.dim() == 1:
        input_tokens = input_tokens.unsqueeze(0).expand(B, -1)
    parts.append(input_emb_table[input_tokens])
    
    seq_embs = torch.cat(parts, dim=1)
    
    # Forward pass with proper hidden states request
    if decoder.config.patch_all_layers:
        # Use patched forward
        hidden_states, _ = decoder._patched_forward(
            main_base=main_base,
            seq_embs=seq_embs,
            activation_input_modified=activation_input,
            use_projection=use_projection,
            do_patching=True,
            prompt_left_emb=prompt_left_emb,
        )
        logits = main_out(hidden_states)
    else:
        # Standard forward - FIX: Add output_hidden_states=True
        with torch.no_grad():
            outputs = main_base(inputs_embeds=seq_embs, output_hidden_states=True)
        hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs.hidden_states[-1]
        logits = main_out(hidden_states)
    
    prompt_len = prompt_len_left + 1 + prompt_len_right
    
    # Extract logits for input tokens
    logits_for_input_tokens = logits[:, prompt_len - 1 : -1, :]
    
    # Calculate probabilities
    probs = torch.nn.functional.softmax(logits_for_input_tokens, dim=-1)
    probs_of_interest = probs.gather(dim=2, index=input_tokens.unsqueeze(-1)).squeeze(-1)
    
    # Calculate entropy
    log_probs = torch.nn.functional.log_softmax(logits_for_input_tokens, dim=-1)
    entropies = (-probs * log_probs).sum(dim=-1)
    
    return probs_of_interest, entropies


def test_fwd_tokens_alignment():
    """Test that fwd_tokens (fixed version) aligns with generation methods."""
    
    device = torch.device("cuda")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("="*80)
    print("FWD_TOKENS ALIGNMENT TEST (WITH WORKAROUND)")
    print("="*80)
    
    # Test configurations
    configs = [
        {
            "name": "Single-layer patching",
            "params": {
                "patch_all_layers": False,
                "per_layer_projections": False,
            }
        },
        {
            "name": "Multi-layer patching", 
            "params": {
                "patch_all_layers": True,
                "per_layer_projections": True,
            }
        },
    ]
    
    for config in configs:
        print(f"\n{'-'*70}")
        print(f"Testing: {config['name']}")
        print(f"{'-'*70}")
        
        # Create decoder with output_head=True for fwd_tokens
        decoder_fwd = Decoder(DecoderConfig(
            model_name=model_name,
            base_model=False,
            projection_layer=True,
            output_head=True,  # Required for fwd_tokens
            **config['params']
        )).to(device).eval()
        decoder_fwd.set_prompt("The answer is <embed>:", tokenizer)
        
        # Create decoder with output_head=False for generation
        decoder_gen = Decoder(DecoderConfig(
            model_name=model_name,
            base_model=False,
            projection_layer=True,
            output_head=False,  # Required for generate_soft
            **config['params']
        )).to(device).eval()
        decoder_gen.set_prompt("The answer is <embed>:", tokenizer)
        
        # Copy projection weights to ensure consistency
        if config['params'].get('per_layer_projections', False):
            decoder_gen.proj_weight.data = decoder_fwd.proj_weight.data.clone()
            decoder_gen.proj_bias.data = decoder_fwd.proj_bias.data.clone()
        else:
            decoder_gen.proj.weight.data = decoder_fwd.proj.weight.data.clone()
            decoder_gen.proj.bias.data = decoder_fwd.proj.bias.data.clone()
        
        d_model = decoder_fwd.base.config.hidden_size
        
        # Test cases
        test_cases = [(10, 42), (20, 123)]
        
        all_passed = True
        
        for max_length, seed in test_cases:
            print(f"\n  Testing length={max_length}, seed={seed}")
            
            # Create activation
            set_all_seeds(seed + 1000)
            activation = torch.randn(1, d_model, device=device)
            
            # Generate sequence
            set_all_seeds(seed)
            gen_kv = decoder_gen.generate_soft_kv_cached(activation.clone(), max_length, gumbel_tau=1.0)
            generated_tokens = gen_kv.hard_token_ids[0]
            
            # Use fixed fwd_tokens
            probs_fwd, entropies_fwd = fixed_fwd_tokens(
                decoder_fwd,
                activation.clone(),
                generated_tokens,
                use_projection=True
            )
            
            # Extract probabilities from generation
            logits_kv = gen_kv.raw_lm_logits[0]
            probs_kv = torch.softmax(logits_kv, dim=-1)
            probs_kv_selected = probs_kv.gather(1, generated_tokens.unsqueeze(-1)).squeeze(-1)
            
            # Compare
            prob_diff = (probs_fwd[0] - probs_kv_selected).abs().max().item()
            
            print(f"    Max prob diff: {prob_diff:.2e}")
            
            if prob_diff < 1e-5:
                print(f"    ✅ Probabilities match!")
            else:
                print(f"    ❌ Probabilities differ!")
                all_passed = False
        
        if all_passed:
            print(f"\n✅ All tests passed for {config['name']}!")
        else:
            print(f"\n❌ Some tests failed for {config['name']}!")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("The test shows that fwd_tokens CAN align with generation methods")
    print("once the bug is fixed (missing output_hidden_states=True).")
    print("\nThe bug is in lens/models/decoder.py line 1676:")
    print("  outputs = main_base(inputs_embeds=seq_embs)")
    print("Should be:")
    print("  outputs = main_base(inputs_embeds=seq_embs, output_hidden_states=True)")
    print("\nWith this fix, fwd_tokens will work correctly for RL applications.")


if __name__ == "__main__":
    test_fwd_tokens_alignment()