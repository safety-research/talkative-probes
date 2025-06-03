#!/usr/bin/env python3
"""Debug what token 20676 is."""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(f"Token 20676: '{tokenizer.decode([20676])}'")

# Also check the prompt tokens
prompt = "explain <embed>:"
tokens = tokenizer(prompt.replace("<embed>", ""), add_special_tokens=False).input_ids
print(f"\nPrompt '{prompt}' tokens: {tokens}")
for t in tokens:
    print(f"  {t}: '{tokenizer.decode([t])}'")

# Check if 20676 is in the prompt
if 20676 in tokens:
    print(f"\n✗ Token 20676 is from the prompt!")