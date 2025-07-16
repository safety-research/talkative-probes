#!/usr/bin/env python3
"""Check if checkpoint prompt embeddings match their nearest token embeddings."""

import torch
import numpy as np
from pathlib import Path
import argparse
from transformers import AutoTokenizer

from lens.models.decoder import Decoder, DecoderConfig


def find_nearest_tokens(embedding, token_embeddings, tokenizer, top_k=5):
    """Find the nearest token embeddings to a given embedding."""
    # Compute cosine similarities
    embedding_norm = embedding / embedding.norm()
    token_norms = token_embeddings / token_embeddings.norm(dim=1, keepdim=True)
    similarities = token_norms @ embedding_norm
    
    # Get top k
    top_values, top_indices = similarities.topk(top_k)
    
    results = []
    for val, idx in zip(top_values, top_indices):
        token = tokenizer.decode([idx.item()])
        results.append((token, idx.item(), val.item()))
    
    return results


def main(checkpoint_path):
    """Check if prompt embeddings match hard prompt tokens."""
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get decoder state
    decoder_state = ckpt['models']['decoder']
    
    # Extract prompt embeddings
    if 'prompt_left_emb' not in decoder_state or 'prompt_right_emb' not in decoder_state:
        print("No prompt embeddings found in checkpoint!")
        return
    
    prompt_left = decoder_state['prompt_left_emb']
    prompt_right = decoder_state['prompt_right_emb']
    
    print(f"\nPrompt left shape: {prompt_left.shape}")
    print(f"Prompt right shape: {prompt_right.shape}")
    
    # Create model to get token embeddings
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    decoder = Decoder(DecoderConfig(
        model_name=model_name,
        per_layer_projections=True,
        base_model=True,
        projection_layer=True,
        output_head=False,
        use_kv_cache=True,
        patch_all_layers=True,
        end_to_end=True,
        detach_after_each_sample=False
    ))
    
    # Get token embedding table
    token_embeddings = decoder.base.get_input_embeddings().weight.detach()
    
    # Tokenize the expected prompt
    expected_prompt = "<|endoftext|>Short explanation of <embed>. Language, topic, sentiment, claims, speaker, style, etc:"
    left_str, right_str = expected_prompt.split("<embed>")
    left_ids = tokenizer(left_str, add_special_tokens=False).input_ids
    right_ids = tokenizer(right_str, add_special_tokens=False).input_ids
    
    print(f"\nExpected left tokens: {left_ids}")
    print(f"Expected left text: '{tokenizer.decode(left_ids)}'")
    print(f"\nExpected right tokens: {right_ids}")
    print(f"Expected right text: '{tokenizer.decode(right_ids)}'")
    
    # Check if prompt embeddings match expected tokens
    print("\n" + "="*80)
    print("LEFT PROMPT ANALYSIS")
    print("="*80)
    
    for i in range(prompt_left.shape[0]):
        print(f"\nPosition {i}:")
        
        # Get expected token embedding
        if i < len(left_ids):
            expected_token_id = left_ids[i]
            expected_token = tokenizer.decode([expected_token_id])
            expected_embedding = token_embeddings[expected_token_id]
            
            # Compare with checkpoint embedding
            checkpoint_embedding = prompt_left[i]
            distance = (checkpoint_embedding - expected_embedding).norm().item()
            cosine_sim = torch.nn.functional.cosine_similarity(
                checkpoint_embedding.unsqueeze(0), 
                expected_embedding.unsqueeze(0)
            ).item()
            
            print(f"  Expected token: '{expected_token}' (id={expected_token_id})")
            print(f"  Distance from expected: {distance:.6f}")
            print(f"  Cosine similarity: {cosine_sim:.6f}")
            
            if distance < 0.01:
                print(f"  ✓ MATCHES expected token embedding!")
            else:
                print(f"  ✗ Does NOT match expected token")
        
        # Find nearest tokens
        nearest = find_nearest_tokens(prompt_left[i], token_embeddings, tokenizer, top_k=3)
        print(f"  Nearest tokens:")
        for token, token_id, sim in nearest:
            print(f"    '{token}' (id={token_id}, sim={sim:.4f})")
    
    print("\n" + "="*80)
    print("RIGHT PROMPT ANALYSIS")
    print("="*80)
    
    for i in range(prompt_right.shape[0]):
        print(f"\nPosition {i}:")
        
        # Get expected token embedding
        if i < len(right_ids):
            expected_token_id = right_ids[i]
            expected_token = tokenizer.decode([expected_token_id])
            expected_embedding = token_embeddings[expected_token_id]
            
            # Compare with checkpoint embedding
            checkpoint_embedding = prompt_right[i]
            distance = (checkpoint_embedding - expected_embedding).norm().item()
            cosine_sim = torch.nn.functional.cosine_similarity(
                checkpoint_embedding.unsqueeze(0), 
                expected_embedding.unsqueeze(0)
            ).item()
            
            print(f"  Expected token: '{expected_token}' (id={expected_token_id})")
            print(f"  Distance from expected: {distance:.6f}")
            print(f"  Cosine similarity: {cosine_sim:.6f}")
            
            if distance < 0.01:
                print(f"  ✓ MATCHES expected token embedding!")
            else:
                print(f"  ✗ Does NOT match expected token")
        
        # Find nearest tokens
        nearest = find_nearest_tokens(prompt_right[i], token_embeddings, tokenizer, top_k=3)
        print(f"  Nearest tokens:")
        for token, token_id, sim in nearest:
            print(f"    '{token}' (id={token_id}, sim={sim:.4f})")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Check overall norms
    expected_left_embeddings = token_embeddings[torch.tensor(left_ids)]
    expected_right_embeddings = token_embeddings[torch.tensor(right_ids)]
    
    print(f"\nCheckpoint left norm: {prompt_left.norm().item():.6f}")
    print(f"Expected left norm: {expected_left_embeddings.norm().item():.6f}")
    print(f"Difference: {abs(prompt_left.norm().item() - expected_left_embeddings.norm().item()):.6f}")
    
    print(f"\nCheckpoint right norm: {prompt_right.norm().item():.6f}")
    print(f"Expected right norm: {expected_right_embeddings.norm().item():.6f}")
    print(f"Difference: {abs(prompt_right.norm().item() - expected_right_embeddings.norm().item()):.6f}")
    
    # Check if they're identical
    left_matches = torch.allclose(prompt_left, expected_left_embeddings, atol=1e-5)
    right_matches = torch.allclose(prompt_right, expected_right_embeddings, atol=1e-5)
    
    print(f"\nLeft prompt matches hard tokens: {left_matches}")
    print(f"Right prompt matches hard tokens: {right_matches}")
    
    if left_matches and right_matches:
        print("\n⚠️  WARNING: Checkpoint contains UNTRAINED prompt embeddings!")
        print("The prompts are still at their initialization values.")
        print("This explains the KL loss jump on resume.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    
    main(args.checkpoint)