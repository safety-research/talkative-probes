#!/usr/bin/env python3
import torch
from transformers import GPT2Tokenizer, GPT2Model

# Initialize tokenizer and base model
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
base_model = GPT2Model.from_pretrained("openai-community/gpt2")

# Get the prompt tokens
prompt_text = "<|endoftext|>Short explanation of <embed>. Language, topic, sentiment, claims, speaker, style, etc:"
left_text = "<|endoftext|>Short explanation of "
right_text = ". Language, topic, sentiment, claims, speaker, style, etc:"

left_ids = tokenizer.encode(left_text, add_special_tokens=False)
right_ids = tokenizer.encode(right_text, add_special_tokens=False)

print(f"Left tokens: {left_ids}")
print(f"Right tokens: {right_ids}")

# Get embeddings from base model
with torch.no_grad():
    left_emb = base_model.wte(torch.tensor(left_ids))
    right_emb = base_model.wte(torch.tensor(right_ids))

print(f"\nFresh initialization values from GPT-2 embedding table:")
print(f"Left norm: {left_emb.norm().item():.6f}")
print(f"Right norm: {right_emb.norm().item():.6f}")

# Now check what's in the checkpoint
ckpt_path = "outputs/checkpoints/gpt2_frozen_e6_wider1p0multigpu2chgprompt_OC_GPT2_L6_e5_prog-unfreeze_lr1e-4_t32_20ep_resume_0608_2238_tinyLMKVext_dist4/gpt2_frozen_step94000_epoch13_.pt"
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

ckpt_left_norm = ckpt['models']['decoder']['prompt_left_emb'].norm().item()
ckpt_right_norm = ckpt['models']['decoder']['prompt_right_emb'].norm().item()

print(f"\nCheckpoint values:")
print(f"Left norm: {ckpt_left_norm:.6f}")
print(f"Right norm: {ckpt_right_norm:.6f}")

print(f"\nDifference from fresh init:")
print(f"Left diff: {abs(left_emb.norm().item() - ckpt_left_norm):.6f}")
print(f"Right diff: {abs(right_emb.norm().item() - ckpt_right_norm):.6f}")

# The issue might be in how the comparison is done in the training script
print(f"\nTraining script comparison (with 0.01 tolerance):")
print(f"Left match: {abs(left_emb.norm().item() - ckpt_left_norm) < 0.01}")
print(f"Right match: {abs(right_emb.norm().item() - ckpt_right_norm) < 0.01}")