# Talkative proves autoencoder Environment Setup - Simplified for Multi-Node
# 
# This Makefile provides a simplified setup using uv's built-in environment management.
# 
# Key features:
# - Installs uv package manager if needed
# - Configures shared UV cache in /workspace/.uv_cache
# - Creates and syncs the uv environment
# - Works seamlessly across multiple nodes
#
# Usage:
#   make          - Initial setup (install uv, create env, sync dependencies)
#   uv run python scripts/01_train.py  - Run any Python command

.PHONY: all setup help flash-attention sync-env wandb-login

# Default target
all: setup

# Help target
help:
	@echo "Talkative Probes Autoencoder Setup"
	@echo "  make              - Initial setup (install uv, create env, sync dependencies)"
	@echo "  make sync-env     - Sync/update the environment with latest dependencies"
	@echo "  make wandb-login  - Login to Weights & Biases (requires WANDB_API_KEY env var)"
	@echo "  make flash-attention - Install Flash Attention 2 (optional, for performance)"
	@echo ""
	@echo "After setup, use 'uv run' for all Python commands:"
	@echo "  uv run python scripts/01_train.py"
	@echo "  uv run pytest tests/"
	@echo ""
	@echo "No manual venv activation needed!"

# Setup - install uv, configure cache, and create environment
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
	@# Source ensure_env.sh to get environment variables
	@echo "Sourcing environment setup..."
	@export UV_CACHE_DIR="/workspace/.uv_cache" && \
	export UV_PROJECT_ROOT="$$(pwd)" && \
	if [ -n "$$SLURM_JOB_ID" ] && [ -n "$${SLURM_TMPDIR:-}" ]; then \
		export UV_PROJECT_ENVIRONMENT="$$SLURM_TMPDIR/uv-venv-consistency-lens"; \
	else \
		export UV_PROJECT_ENVIRONMENT="$$HOME/.cache/uv/envs/consistency-lens"; \
	fi && \
	echo "Creating/syncing uv environment at $$UV_PROJECT_ENVIRONMENT..." && \
	PATH="$$HOME/.local/bin:$$PATH" uv sync --frozen
	@# Login to wandb if API key is available (source .env first)
	@if [ -f .env ]; then \
		echo "Loading environment variables from .env..."; \
		export $$(grep -v '^#' .env | xargs); \
	fi && \
	if [ -n "$$WANDB_API_KEY" ]; then \
		echo "Logging into Weights & Biases..."; \
		export UV_CACHE_DIR="/workspace/.uv_cache" && \
		export UV_PROJECT_ROOT="$$(pwd)" && \
		if [ -n "$$SLURM_JOB_ID" ] && [ -n "$${SLURM_TMPDIR:-}" ]; then \
			export UV_PROJECT_ENVIRONMENT="$$SLURM_TMPDIR/uv-venv-consistency-lens"; \
		else \
			export UV_PROJECT_ENVIRONMENT="$$HOME/.cache/uv/envs/consistency-lens"; \
		fi && \
		PATH="$$HOME/.local/bin:$$PATH" uv run wandb login $$WANDB_API_KEY && \
		echo "✓ Successfully logged into Weights & Biases!"; \
	else \
		echo "Skipping wandb login (WANDB_API_KEY not found)"; \
	fi
	@echo ""
	@echo "✓ Setup complete! Environment created and dependencies installed."
	@echo ""
	@echo "Usage: prefix all Python commands with 'uv run'"
	@echo "  uv run python scripts/01_train.py"
	@echo "  uv run pytest tests/"
	@echo ""
	@echo "Note: The environment is now ready to use immediately."

# Sync environment with latest dependencies
sync-env:
	@echo "Syncing environment with latest dependencies..."
	@export UV_CACHE_DIR="/workspace/.uv_cache" && \
	export UV_PROJECT_ROOT="$$(pwd)" && \
	if [ -n "$$SLURM_JOB_ID" ] && [ -n "$${SLURM_TMPDIR:-}" ]; then \
		export UV_PROJECT_ENVIRONMENT="$$SLURM_TMPDIR/uv-venv-consistency-lens"; \
	else \
		export UV_PROJECT_ENVIRONMENT="$$HOME/.cache/uv/envs/consistency-lens"; \
	fi && \
	PATH="$$HOME/.local/bin:$$PATH" uv sync
	@echo "✓ Environment synced successfully!"

# Login to Weights & Biases
wandb-login:
	@if [ -z "$$WANDB_API_KEY" ]; then \
		echo "Error: WANDB_API_KEY environment variable is not set"; \
		echo "Please set it with: export WANDB_API_KEY=your_api_key"; \
		exit 1; \
	fi
	@echo "Logging into Weights & Biases..."
	@export UV_CACHE_DIR="/workspace/.uv_cache" && \
	export UV_PROJECT_ROOT="$$(pwd)" && \
	if [ -n "$$SLURM_JOB_ID" ] && [ -n "$${SLURM_TMPDIR:-}" ]; then \
		export UV_PROJECT_ENVIRONMENT="$$SLURM_TMPDIR/uv-venv-consistency-lens"; \
	else \
		export UV_PROJECT_ENVIRONMENT="$$HOME/.cache/uv/envs/consistency-lens"; \
	fi && \
	PATH="$$HOME/.local/bin:$$PATH" uv run wandb login $$WANDB_API_KEY
	@echo "✓ Successfully logged into Weights & Biases!"

# Install Flash Attention 2 (optional)
flash-attention:
	@./scripts/install_flash_attention.sh
