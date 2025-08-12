# Talkative proves autoencoder Environment Setup - Simplified for Multi-Node
# 
# This Makefile provides a simplified setup using uv's built-in environment management.
# 
# Key features:
# - Installs uv package manager if needed
# - Configures shared UV cache in /workspace/.uv_cache
# - Creates and syncs the uv environment
# - Works seamlessly across multiple nodes
# - Handles B200 node-specific dependencies
#
# Usage:
#   make          - Initial setup (install uv, create env, sync dependencies)
#   uv run python scripts/01_train.py  - Run any Python command

.PHONY: all setup help flash-attention sync-env wandb-login check-b200 link-consistency-lens

# Default target
all: setup

# Where to place the symlink (override with: make SYMLINK_PATH=/path/consistency-lens link-consistency-lens)
SYMLINK_PATH ?= $(abspath $(CURDIR)/../consistency-lens)

# Help target
help:
	@echo "Talkative Probes Autoencoder Setup"
	@echo "  make              - Initial setup (install uv, create env, sync dependencies)"
	@echo "  make sync-env     - Sync/update the environment with latest dependencies"
	@echo "  make wandb-login  - Login to Weights & Biases (requires WANDB_API_KEY env var)"
	@echo "  make flash-attention - Install Flash Attention 2 (optional, for performance)"
	@echo "  make link-consistency-lens - Create symlink 'consistency-lens' to this dir"
	@echo ""
	@echo "After setup, use 'uv run' for all Python commands:"
	@echo "  uv run python scripts/01_train.py"
	@echo "  uv run pytest tests/"
	@echo ""
	@echo "No manual venv activation needed!"

# Check if running on B200 node
check-b200:
	@if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q "B200"; then \
		echo "DETECTED_B200=true"; \
	else \
		echo "DETECTED_B200=false"; \
	fi

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
    @# Source ensure_env.sh and sync environment first
    @echo "Sourcing environment setup..."
    @. ./scripts/ensure_env.sh && \
    echo "Creating/syncing uv environment..." && \
    if [ -f uv.lock ]; then uv sync --frozen; else uv sync; fi
	@$(MAKE) link-consistency-lens
	@# Check if on B200 node and remove packages if needed
	@if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q "B200"; then \
		echo "Detected B200 node - removing incompatible packages..."; \
		. ./scripts/ensure_env.sh && \
		uv remove bitsandbytes cairosvg || true; \
		echo "Removed bitsandbytes and cairosvg for B200 compatibility"; \
		echo "Installing PyTorch and markupsafe..."; \
		. ./scripts/ensure_env.sh && \
		uv add "markupsafe==2.1.3" && \
		uv add torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128 ; \
	fi
	@# Login to wandb if API key is available (source .env first)
	@if [ -f .env ]; then \
		echo "Loading environment variables from .env..."; \
		export $$(grep -v '^#' .env | xargs); \
	fi && \
	if [ -n "$$WANDB_API_KEY" ]; then \
		echo "Logging into Weights & Biases..."; \
		. ./scripts/ensure_env.sh && \
		uv run wandb login $$WANDB_API_KEY && \
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
	@. ./scripts/ensure_env.sh && \
	uv sync
	@echo "✓ Environment synced successfully!"

# Login to Weights & Biases
wandb-login:
	@if [ -z "$$WANDB_API_KEY" ]; then \
		echo "Error: WANDB_API_KEY environment variable is not set"; \
		echo "Please set it with: export WANDB_API_KEY=your_api_key"; \
		exit 1; \
	fi
	@echo "Logging into Weights & Biases..."
	@. ./scripts/ensure_env.sh && \
	uv run wandb login $$WANDB_API_KEY
	@echo "✓ Successfully logged into Weights & Biases!"

# Install Flash Attention 2 (optional)
flash-attention:
	@./scripts/install_flash_attention.sh

# Create/update symlink to this project directory as 'consistency-lens'
link-consistency-lens:
	@echo "Symlink: $(SYMLINK_PATH) -> $(abspath $(CURDIR))"
	@mkdir -p "$(dir $(SYMLINK_PATH))"
	@if [ -e "$(SYMLINK_PATH)" ] && [ ! -L "$(SYMLINK_PATH)" ]; then \
		echo "Error: $(SYMLINK_PATH) exists and is not a symlink"; \
		exit 1; \
	fi
	@ln -sfn "$(abspath $(CURDIR))" "$(SYMLINK_PATH)"
	@echo "✓ Symlink ready"
