#!/bin/bash
# Simplified environment setup for multi-node environments using uv
# This script ensures uv is available and sets up the uv_run helper function

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Ensure uv is installed
if ! command -v uv >/dev/null 2>&1; then
    echo "[ensure_env] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Clear any conflicting VIRTUAL_ENV variable
unset VIRTUAL_ENV

# Export helper function for using uv run
export UV_PROJECT_ROOT="$PROJECT_ROOT"
# Use shared UV cache for package downloads
export UV_CACHE_DIR="/workspace/.uv_cache"
mkdir -p "$UV_CACHE_DIR"

# Set up node-local virtual environment location
if [ -n "$SLURM_JOB_ID" ] && [ -n "$SLURM_TMPDIR" ]; then
    # On compute nodes, use fast local storage
    export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/uv-venv-consistency-lens"
    echo "[ensure_env] Using node-local venv: $UV_PROJECT_ENVIRONMENT"
else
    # On login nodes or local dev, use home directory
    export UV_PROJECT_ENVIRONMENT="$HOME/.cache/uv/envs/consistency-lens"
    echo "[ensure_env] Using cached venv: $UV_PROJECT_ENVIRONMENT"
fi

# Helper function that ensures we're in the project root and uses uv run
uv_run() {
    cd "$UV_PROJECT_ROOT" && PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$UV_CACHE_DIR" UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT" uv run "$@"
}
export -f uv_run

# Let users know the environment is ready
if [ -n "$SLURM_JOB_ID" ]; then
    echo "[ensure_env] Running on SLURM node: $(hostname)"
fi
echo "[ensure_env] Environment ready. Use 'uv run' (not uv_run) for all Python commands."
echo "[ensure_env] Note: The uv_run function is available as a shortcut in this shell."