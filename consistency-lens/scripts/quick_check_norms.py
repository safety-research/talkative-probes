#!/usr/bin/env python3
import torch
import sys

ckpt_path = sys.argv[1]
print(f"Checking: {ckpt_path}")

try:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    decoder_state = ckpt['models']['decoder']
    
    left_norm = decoder_state['prompt_left_emb'].norm().item()
    right_norm = decoder_state['prompt_right_emb'].norm().item()
    
    print(f"Left norm: {left_norm:.6f}")
    print(f"Right norm: {right_norm:.6f}")
    
    # Check if these match fresh initialization
    if abs(left_norm - 7.4487) < 0.01 and abs(right_norm - 13.4579) < 0.01:
        print("WARNING: These are FRESH initialization values!")
    else:
        print("These appear to be trained values.")
        
except Exception as e:
    print(f"Error: {e}")