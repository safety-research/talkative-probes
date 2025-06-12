#!/usr/bin/env python3
"""Debug why decoder prompts aren't loading."""

import torch
from pathlib import Path
import argparse
from transformers import AutoTokenizer

from lens.models.decoder import Decoder, DecoderConfig


def main(checkpoint_path):
    """Debug prompt loading issue."""
    
    # Load raw checkpoint
    print("Loading checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get decoder state
    decoder_state = ckpt['models']['decoder']
    
    # Print all keys
    print("\nCheckpoint decoder keys:")
    for k in sorted(decoder_state.keys()):
        if 'prompt' in k:
            v = decoder_state[k]
            if isinstance(v, torch.Tensor):
                if v.dtype.is_floating_point:
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}, norm={v.norm().item():.6f}")
                else:
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: {type(v)}")
    
    # Create decoder
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
    
    # Set prompt
    prompt = "<|endoftext|>Short explanation of <embed>. Language, topic, sentiment, claims, speaker, style, etc:"
    decoder.set_prompt(prompt, tokenizer)
    
    print("\nDecoder state dict keys after set_prompt:")
    for k in sorted(decoder.state_dict().keys()):
        if 'prompt' in k:
            v = decoder.state_dict()[k]
            if v.dtype.is_floating_point:
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}, norm={v.norm().item():.6f}")
            else:
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    
    # Check parameter vs buffer
    print("\nPrompt embedding registration:")
    print(f"  prompt_left_emb is parameter: {isinstance(decoder.prompt_left_emb, torch.nn.Parameter)}")
    print(f"  prompt_right_emb is parameter: {isinstance(decoder.prompt_right_emb, torch.nn.Parameter)}")
    
    # Try manual loading
    print("\nTrying manual state dict load...")
    try:
        decoder.load_state_dict(decoder_state, strict=True)
        print("Success with strict=True!")
    except Exception as e:
        print(f"Failed with strict=True: {e}")
        print("\nTrying with strict=False...")
        missing, unexpected = decoder.load_state_dict(decoder_state, strict=False)
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")
    
    # Check if values changed
    print("\nAfter loading:")
    if hasattr(decoder, 'prompt_left_emb'):
        print(f"  prompt_left_emb norm: {decoder.prompt_left_emb.norm().item():.6f}")
    if hasattr(decoder, 'prompt_right_emb'):
        print(f"  prompt_right_emb norm: {decoder.prompt_right_emb.norm().item():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    
    main(args.checkpoint)