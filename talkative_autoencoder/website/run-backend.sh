#!/bin/bash
# Wrapper script to run the backend with proper environment setup

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Source the environment setup
source "$PROJECT_ROOT/scripts/ensure_env.sh"

# Run the backend
cd "$PROJECT_ROOT"
exec uv run python -m website.backend.app.main