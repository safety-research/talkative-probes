# Consistency Lens Environment Setup - Simplified for Multi-Node
# 
# This Makefile provides a simplified setup using uv's built-in environment management.
# 
# Key features:
# - Installs uv package manager if needed
# - Configures shared UV cache in /workspace/.uv_cache
# - No manual venv management - uv handles environments automatically
# - Works seamlessly across multiple nodes
#
# Usage:
#   make          - Initial setup (install uv, configure cache)
#   uv run python scripts/01_train.py  - Run any Python command

.PHONY: all setup help

# Default target
all: setup

# Help target
help:
	@echo "Consistency Lens Simplified Setup"
	@echo "  make       - Initial setup (install uv, configure cache)"
	@echo ""
	@echo "After setup, use 'uv run' for all Python commands:"
	@echo "  uv run python scripts/01_train.py"
	@echo "  uv run pytest tests/"
	@echo ""
	@echo "No manual venv activation needed!"

# Setup - just ensure uv is installed and cache is configured
setup:
	@echo "Setting up uv-based environment..."
	@# Install uv if not present
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo ""; \
		echo "=== IMPORTANT ==="; \
		echo "Add uv to your PATH by running:"; \
		echo "  source $$HOME/.cargo/env"; \
		echo "Or add it to your shell profile."; \
		echo "================="; \
	else \
		echo "uv is already installed"; \
	fi
	@# Run dotfiles setup if available
	@if [ -f /workspace/kitf/dotfiles/setup_env.sh ]; then \
		echo "Running dotfiles setup..."; \
		/workspace/kitf/dotfiles/setup_env.sh; \
	fi
	@# Configure shared cache
	@mkdir -p /workspace/.uv_cache
	@echo ""
	@echo "✓ Setup complete! You only need to run this once."
	@echo ""
	@echo "Usage: prefix all Python commands with 'uv run'"
	@echo "  uv run python scripts/01_train.py"
	@echo "  uv run pytest tests/"
	@echo ""
	@echo "Note: First 'uv run' on each node will be slower (~1-2 min) as packages install."
