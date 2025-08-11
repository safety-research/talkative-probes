#!/bin/bash
# Simplified environment setup for multi-node environments using uv
# This script ensures uv is available and sets up the uv_run helper function
# It also creates the uv environment if it doesn't exist

#set -e

# Determine PROJECT_ROOT in a POSIX-compatible way (works when sourced by /bin/sh)
if [ -n "${BASH_VERSION:-}" ] && [ -n "${BASH_SOURCE:-}" ]; then
    SOURCE_PATH="${BASH_SOURCE[0]}"
    while [ -h "$SOURCE_PATH" ]; do
        DIR="$(cd -P "$(dirname "$SOURCE_PATH")" >/dev/null 2>&1 && pwd)"
        LINK="$(readlink "$SOURCE_PATH")"
        case "$LINK" in
            /*) SOURCE_PATH="$LINK" ;;
            *) SOURCE_PATH="$DIR/$LINK" ;;
        esac
    done
    SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE_PATH")" >/dev/null 2>&1 && pwd)"
elif [ -f "./scripts/ensure_env.sh" ]; then
    SCRIPT_DIR="$(cd -P "./scripts" >/dev/null 2>&1 && pwd)"
else
    SCRIPT_DIR="$(cd -P "$(dirname "$0")" >/dev/null 2>&1 && pwd)"
fi
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# Ensure a stable symlink named 'consistency-lens'
SYMLINK_PATH_DEFAULT="$(dirname "$PROJECT_ROOT")/consistency-lens"
SYMLINK_PATH="${SYMLINK_PATH:-$SYMLINK_PATH_DEFAULT}"
if [ -e "$SYMLINK_PATH" ] && [ ! -L "$SYMLINK_PATH" ]; then
    echo "[ensure_env] Error: $SYMLINK_PATH exists and is not a symlink"
    return 1 2>/dev/null || exit 1
fi
ln -sfn "$PROJECT_ROOT" "$SYMLINK_PATH"
echo "[ensure_env] Symlink ready: $SYMLINK_PATH -> $PROJECT_ROOT"

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
if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_TMPDIR:-}" ]; then
    # On compute nodes, use fast local storage
    export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/uv-venv-consistency-lens"
    echo "[ensure_env] Using node-local venv: $UV_PROJECT_ENVIRONMENT"
else
    # On login nodes or local dev, use home directory
    export UV_PROJECT_ENVIRONMENT="$HOME/.cache/uv/envs/consistency-lens"
    echo "[ensure_env] Using cached venv: $UV_PROJECT_ENVIRONMENT"
fi

# Check if environment exists, create it if not
if [ ! -d "$UV_PROJECT_ENVIRONMENT" ] || [ ! -f "$UV_PROJECT_ENVIRONMENT/pyvenv.cfg" ]; then
    echo "[ensure_env] Environment not found, creating it... from $UV_PROJECT_ROOT"
    cd "$UV_PROJECT_ROOT" && PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$UV_CACHE_DIR" UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT" uv sync --frozen
    echo "[ensure_env] Environment created successfully!"
else
    echo "[ensure_env] Environment already exists at $UV_PROJECT_ENVIRONMENT"
fi

# Helper function that ensures we're in the project root and uses uv run
uv_run() {
    cd "$UV_PROJECT_ROOT" && PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$UV_CACHE_DIR" UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT" uv run "$@"
}
# Try to export function if supported, but don't fail if not
if [ -n "$BASH_VERSION" ]; then
    export -f uv_run 2>/dev/null || true
fi

# Let users know the environment is ready
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "[ensure_env] Running on SLURM node: $(hostname)"
fi
echo "[ensure_env] Environment ready. Use 'uv run' (not uv_run) for all Python commands."
echo "[ensure_env] Note: The uv_run function is available as a shortcut in this shell."
