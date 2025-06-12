#!/usr/bin/env python3
"""Test if decoder prompt embeddings are loaded correctly from checkpoint."""

import torch
from pathlib import Path
import argparse
from transformers import AutoTokenizer

from lens.models.decoder import Decoder, DecoderConfig
from lens.utils.checkpoint import load as checkpoint_load


def main(checkpoint_path):
    """Test prompt loading."""
    
    # Load raw checkpoint
    print("Loading checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get decoder state
    decoder_state = ckpt['models']['decoder']
    
    # Check prompt embeddings in checkpoint
    has_left = 'prompt_left_emb' in decoder_state
    has_right = 'prompt_right_emb' in decoder_state
    
    print(f"\nCheckpoint has prompt embeddings: left={has_left}, right={has_right}")
    
    if has_left:
        ckpt_left_norm = decoder_state['prompt_left_emb'].norm().item()
        print(f"Checkpoint left prompt norm: {ckpt_left_norm:.6f}")
    
    if has_right:
        ckpt_right_norm = decoder_state['prompt_right_emb'].norm().item()
        print(f"Checkpoint right prompt norm: {ckpt_right_norm:.6f}")
    
    # Create decoder
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    decoder = Decoder(DecoderConfig(
        model_name=model_name,
        per_layer_projections=True,  # Match checkpoint
        base_model=True,
        projection_layer=True,
        output_head=False,
        use_kv_cache=True,
        patch_all_layers=True,
        end_to_end=True,
        detach_after_each_sample=False
    ))
    
    # Set prompt
    prompt = "<|endoftext|>Short explanation of <embed>. Language, topic, sentiment, claims, speaker, style, etc:"
    print(f"\nSetting prompt: {prompt}")
    decoder.set_prompt(prompt, tokenizer)
    
    # Check fresh embeddings
    fresh_left_norm = decoder.prompt_left_emb.norm().item()
    fresh_right_norm = decoder.prompt_right_emb.norm().item()
    print(f"\nFresh left prompt norm: {fresh_left_norm:.6f}")
    print(f"Fresh right prompt norm: {fresh_right_norm:.6f}")
    
    # Load checkpoint
    print("\nLoading checkpoint into model...")
    checkpoint_load(checkpoint_path, models={'decoder': decoder}, map_location='cpu')
    
    # Check loaded embeddings
    loaded_left_norm = decoder.prompt_left_emb.norm().item()
    loaded_right_norm = decoder.prompt_right_emb.norm().item()
    print(f"\nLoaded left prompt norm: {loaded_left_norm:.6f}")
    print(f"Loaded right prompt norm: {loaded_right_norm:.6f}")
    
    # Compare
    print("\n=== Comparison ===")
    if has_left:
        print(f"Left prompt - Checkpoint: {ckpt_left_norm:.6f}, Fresh: {fresh_left_norm:.6f}, Loaded: {loaded_left_norm:.6f}")
        if abs(loaded_left_norm - ckpt_left_norm) < 1e-6:
            print("✓ Left prompt loaded correctly")
        else:
            print("✗ Left prompt NOT loaded correctly!")
    
    if has_right:
        print(f"Right prompt - Checkpoint: {ckpt_right_norm:.6f}, Fresh: {fresh_right_norm:.6f}, Loaded: {loaded_right_norm:.6f}")
        if abs(loaded_right_norm - ckpt_right_norm) < 1e-6:
            print("✓ Right prompt loaded correctly")
        else:
            print("✗ Right prompt NOT loaded correctly!")
    
    # Check if loaded values match fresh (indicating no loading happened)
    if abs(loaded_left_norm - fresh_left_norm) < 1e-6 and abs(loaded_right_norm - fresh_right_norm) < 1e-6:
        print("\n⚠️  WARNING: Loaded values match fresh initialization!")
        print("This means the checkpoint prompt embeddings were NOT loaded.")
        print("This is causing the KL loss jump on resume!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    
    main(args.checkpoint)