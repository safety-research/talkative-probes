#!/bin/bash
set -euo pipefail  # Exit on error, undefined vars, pipe failures

# This script runs on RunPod startup to set up the environment
echo "Starting RunPod setup..."

# Define variables
REPO_URL="${REPO_URL:-https://github.com/kitft/talkative-probes.git}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-qwen2_5_WCHAT_14b_frozen_nopostfix.pt}"
CHECKPOINT_PATH="/workspace/checkpoints/$CHECKPOINT_NAME"

# Function to handle errors
error_exit() {
    echo "Error: $1" >&2
    exit 1
}

# Clone repository if needed
if [ ! -d "/workspace/talkative-probes" ]; then
    echo "Cloning repository..."
    cd /workspace
    git clone "$REPO_URL" || error_exit "Failed to clone repository"
    cd talkative-probes
    git submodule update --init --recursive || error_exit "Failed to update submodules"
fi

cd /workspace/talkative-probes/talkative_autoencoder

# Run environment setup
echo "Setting up Python environment..."
if [ -f "Makefile" ]; then
    make || error_exit "Failed to run make"
else
    error_exit "Makefile not found"
fi

# Download checkpoint if needed
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Checkpoint not found. Please mount a volume with the checkpoint at $CHECKPOINT_PATH"
    # For security, we don't download from arbitrary URLs
    # Instead, expect the checkpoint to be mounted via RunPod volumes
    error_exit "Checkpoint not found at $CHECKPOINT_PATH"
fi

# Verify checkpoint
echo "Verifying checkpoint..."
if [ ! -s "$CHECKPOINT_PATH" ]; then
    error_exit "Checkpoint file is empty"
fi

# Install backend dependencies
cd /workspace/talkative-probes/talkative_autoencoder/website/backend
echo "Installing backend dependencies..."
uv sync || error_exit "Failed to install dependencies"

# Create necessary directories
mkdir -p logs

# Create a simple logging configuration
cat > logging.yaml <<EOF
version: 1
disable_existing_loggers: false
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
    stream: ext://sys.stdout
  file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: default
    filename: logs/app.log
    maxBytes: 10485760  # 10MB
    backupCount: 3
root:
  level: INFO
  handlers: [console, file]
EOF

# Start the API server with proper error handling
echo "Starting API server..."
exec uv run uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --loop uvloop \
    --access-log \
    --log-config logging.yaml