#!/usr/bin/env python3
import torch

# Check the checkpoint
ckpt_path = "outputs/checkpoints/gpt2_frozen_e6_wider1p0multigpu2chgprompt_OC_GPT2_L6_e5_prog-unfreeze_lr1e-4_t32_20ep_resume_0608_2238_tinyLMKVext_dist4/gpt2_frozen_step94000_epoch13_.pt"
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

left_emb = ckpt['models']['decoder']['prompt_left_emb']
right_emb = ckpt['models']['decoder']['prompt_right_emb']

print("Checkpoint embeddings:")
print(f"  Left shape: {left_emb.shape}, norm: {left_emb.norm().item():.6f}")
print(f"  Right shape: {right_emb.shape}, norm: {right_emb.norm().item():.6f}")

# Check if these values appear multiple times in training
print("\nChecking other checkpoints from same run...")
import glob
other_ckpts = sorted(glob.glob("outputs/checkpoints/gpt2_frozen_e6_wider1p0multigpu2chgprompt_OC_GPT2_L6_e5_prog-unfreeze_lr1e-4_t32_20ep_resume_0608_2238_tinyLMKVext_dist4/*.pt"))

for ckpt_file in other_ckpts[-5:]:  # Check last 5 checkpoints
    try:
        other_ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)
        other_left = other_ckpt['models']['decoder']['prompt_left_emb']
        other_right = other_ckpt['models']['decoder']['prompt_right_emb']
        print(f"\n{ckpt_file.split('/')[-1]}:")
        print(f"  Left norm: {other_left.norm().item():.6f}")
        print(f"  Right norm: {other_right.norm().item():.6f}")
    except:
        pass

# The key insight: These exact values (7.449415, 13.464390) appear to be 
# the initialization values for THIS SPECIFIC prompt text in THIS SPECIFIC model.
# They're not the untrained token embeddings, but they ARE the fresh nn.Parameter
# initialization values.