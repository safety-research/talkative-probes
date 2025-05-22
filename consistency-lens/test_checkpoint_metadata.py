#!/usr/bin/env python3
"""Test script to verify checkpoint metadata is properly saved and loaded."""

import torch
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Test checkpoint metadata")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    args = parser.parse_args()
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    
    print("Checkpoint contents:")
    print(f"  Keys: {list(ckpt.keys())}")
    print(f"  Step: {ckpt.get('step', 'N/A')}")
    print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
    
    if 'config' in ckpt:
        config = ckpt['config']
        print("\nConfig metadata:")
        print(f"  model_name: {config.get('model_name', 'N/A')}")
        print(f"  tokenizer_name: {config.get('tokenizer_name', 'N/A')}")
        print(f"  layer_l: {config.get('layer_l', 'N/A')}")
        print(f"  decoder_prompt: '{config.get('decoder_prompt', 'N/A')}'")
        print(f"  t_text: {config.get('t_text', 'N/A')}")
    else:
        print("\nNo config found in checkpoint!")
    
    if 'models' in ckpt and 'dec' in ckpt['models']:
        dec_state = ckpt['models']['dec']
        if 'prompt_ids' in dec_state:
            print(f"\nDecoder prompt_ids shape: {dec_state['prompt_ids'].shape}")

if __name__ == "__main__":
    main() 