#!/bin/bash

# Script to run training with torch.compile enabled and proper cache directory

# Set up torch inductor cache in user's home directory
export TORCHINDUCTOR_CACHE_DIR="${HOME}/.cache/torchinductor"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

# Optional: Enable verbose output for debugging compilation issues
# export TORCHDYNAMO_VERBOSE=1
# export TORCH_LOGS="+dynamo"

# Print cache directory for confirmation
echo "Using torch inductor cache at: ${TORCHINDUCTOR_CACHE_DIR}"

# Run training with all arguments passed through
python scripts/01_train.py "$@"