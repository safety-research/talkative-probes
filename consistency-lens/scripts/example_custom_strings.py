#!/usr/bin/env python3
"""Example of using custom string evaluation with 02_eval.py

This script demonstrates how to evaluate a trained consistency lens checkpoint
on arbitrary input strings instead of pre-computed activations.
"""

import subprocess
import sys
from pathlib import Path

# Example 1: Using custom_strings directly in command line
print("Example 1: Direct string input")
print("-" * 50)

cmd = [
    "uv", "run", "python", "scripts/02_eval.py",
    # Path to your checkpoint
    "checkpoint=outputs/checkpoints/YOUR_CHECKPOINT.pt",
    
    # Provide strings directly
    'evaluation.custom_strings=["The cat sat on the mat.", "Once upon a time", "Machine learning is"]',
    
    # Optional: save results
    "evaluation.save_results=true",
]

print("Command:", " ".join(cmd))
print("\nThis will analyze the provided strings through the lens.\n")

# Example 2: Using a file of strings
print("\nExample 2: File input")
print("-" * 50)

# Create an example file
example_file = Path("example_strings.txt")
with open(example_file, "w") as f:
    f.write("""The quick brown fox jumps over the lazy dog.
In the beginning was the Word.
To be or not to be, that is the question.
import numpy as np
Machine learning models can learn patterns from data.
The weather today is sunny and warm.
def hello_world():
    print("Hello, World!")
""")

cmd2 = [
    "uv", "run", "python", "scripts/02_eval.py", 
    "checkpoint=outputs/checkpoints/YOUR_CHECKPOINT.pt",
    f"evaluation.custom_strings_file={example_file}",
    "evaluation.save_results=true",
]

print(f"Created example file: {example_file}")
print("Command:", " ".join(cmd2))
print("\nThis will analyze all strings from the file.\n")

# Example 3: Combining both methods
print("\nExample 3: Combined input")
print("-" * 50)

cmd3 = [
    "uv", "run", "python", "scripts/02_eval.py",
    "checkpoint=outputs/checkpoints/YOUR_CHECKPOINT.pt",
    'evaluation.custom_strings=["Additional string 1", "Additional string 2"]',
    f"evaluation.custom_strings_file={example_file}",
    "evaluation.save_results=true",
    "evaluation.output_dir=outputs/my_custom_eval",
]

print("Command:", " ".join(cmd3))
print("\nThis combines strings from both sources.\n")

print("\nNOTE: Replace 'YOUR_CHECKPOINT.pt' with your actual checkpoint path!")
print("\nThe evaluation will:")
print("1. Extract activations from the LAST TOKEN of each string at the specified layer")
print("2. Generate explanations through the decoder for that token's activation")
print("3. Reconstruct activations through the encoder")
print("4. Compute reconstruction metrics (MSE, cosine similarity)")
print("5. Optionally analyze additional token positions within each string")
print("\nBy default, the lens analyzes the last token of each input string.")
print("This is typically the most informative position as it has seen the full context.")
print("\nResults will be saved to JSON and a human-readable summary.")