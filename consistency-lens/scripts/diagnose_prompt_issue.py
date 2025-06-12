#!/usr/bin/env python3
import torch
from pathlib import Path
from transformers import GPT2Tokenizer
from lens.models.decoder import Decoder
from hydra import compose, initialize_config_dir
import os

# Load config
config_path = Path("conf/gpt2_frozen_e6_wider1p0multigpu2chgprompt.yaml").resolve()
with initialize_config_dir(config_dir=str(config_path.parent.absolute()), version_base=None):
    config = compose(config_name=config_path.stem)

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
decoder_config = config['decoder']
decoder = Decoder(decoder_config, tokenizer.vocab_size)

# Set prompt to get fresh initialization
decoder.set_prompt(config['decoder_prompt'], tokenizer)
fresh_left_norm = decoder.prompt_left_emb.norm().item()
fresh_right_norm = decoder.prompt_right_emb.norm().item()

print(f"Fresh initialization norms:")
print(f"  Left: {fresh_left_norm:.6f}")
print(f"  Right: {fresh_right_norm:.6f}")

# Load checkpoint
ckpt_path = "outputs/checkpoints/gpt2_frozen_e6_wider1p0multigpu2chgprompt_OC_GPT2_L6_e5_prog-unfreeze_lr1e-4_t32_20ep_resume_0608_2238_tinyLMKVext_dist4/gpt2_frozen_step94000_epoch13_.pt"
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

ckpt_left_norm = ckpt['models']['decoder']['prompt_left_emb'].norm().item()
ckpt_right_norm = ckpt['models']['decoder']['prompt_right_emb'].norm().item()

print(f"\nCheckpoint norms:")
print(f"  Left: {ckpt_left_norm:.6f}")
print(f"  Right: {ckpt_right_norm:.6f}")

# Check if they match
if abs(fresh_left_norm - ckpt_left_norm) < 0.001 and abs(fresh_right_norm - ckpt_right_norm) < 0.001:
    print("\n⚠️  WARNING: Checkpoint contains FRESH initialization values!")
    print("This means the model saved untrained prompt embeddings.")
else:
    print("\n✓ Checkpoint contains different values from fresh initialization")
    print("But the training script is still reporting them as matching...")
    
# Let's check what happens when we load the checkpoint
decoder2 = Decoder(decoder_config, tokenizer.vocab_size)
decoder2.set_prompt(config['decoder_prompt'], tokenizer)

# Before loading
before_left = decoder2.prompt_left_emb.norm().item()
before_right = decoder2.prompt_right_emb.norm().item()

# Load state dict
decoder2.load_state_dict(ckpt['models']['decoder'], strict=True)

# After loading
after_left = decoder2.prompt_left_emb.norm().item()
after_right = decoder2.prompt_right_emb.norm().item()

print(f"\nLoading test:")
print(f"  Before load - Left: {before_left:.6f}, Right: {before_right:.6f}")
print(f"  After load - Left: {after_left:.6f}, Right: {after_right:.6f}")
print(f"  Successfully loaded: {abs(after_left - ckpt_left_norm) < 0.001}")

# Now let's see what the training script is comparing
print(f"\nWhat the training script sees:")
print(f"  Fresh norm (after set_prompt): {fresh_left_norm:.4f}")
print(f"  Checkpoint norm: {ckpt_left_norm:.4f}")
print(f"  Match check: {abs(fresh_left_norm - ckpt_left_norm) < 0.01}")