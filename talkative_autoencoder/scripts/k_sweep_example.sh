#!/bin/bash
# Example: How to run the K-sweep analysis

# The K-sweep analysis requires the following:
# 1. A trained checkpoint file
# 2. The evaluation config specifying K values

# Example commands:

# Quick test with 3 K values (1, 4, 16) on a subset of data:
echo "Quick test command:"
echo "CONFIG=+eval=k_sweep_quick ./scripts/run_k_sweep.sh eval.checkpoint_path=/path/to/checkpoint.pt"

echo ""
echo "Full K-sweep with all values (1, 2, 4, 8, 16, 32, 64):"
echo "./scripts/run_k_sweep.sh eval.checkpoint_path=/path/to/checkpoint.pt"

echo ""
echo "Direct execution with custom K values:"
echo "python scripts/03_best_of_k_sweep.py +eval=k_sweep eval.checkpoint_path=/path/to/checkpoint.pt eval.k_values=[1,5,10,20] eval.output_file=custom_k_sweep.json"

echo ""
echo "Note: You need to run this from the project root with the proper environment activated."
echo "The run_k_sweep.sh script handles environment setup automatically."