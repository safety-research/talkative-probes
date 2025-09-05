# Talkative Probes Autoencoder - Environment & Serving Docs
# 
# This Makefile provides a simplified setup using uv's built-in environment management
# and documents the vLLM serving workflow for gpt-oss-20b (messages-with-thinking).
# 
# Key features:
# - Installs uv package manager if needed
# - Configures shared UV cache in /workspace/.uv_cache
# - Creates and syncs the uv environment
# - Works seamlessly across multiple nodes
# - Handles B200 node-specific dependencies
# - Keeps vLLM out of the default dependency set; use the SBATCH script to install on-demand
#
# Usage:
#   make          - Initial setup (install uv, create env, sync dependencies)
#   uv run python scripts/01_train.py  - Run any Python command
#   make serve-docs - Print vLLM serving instructions (8× H100 node is the main target)
#
# Serving gpt-oss-20b via vLLM (summary):
# - Primary: single node with 8× H100: tensor-parallel-size=8, port 8000
# - Submit: sbatch talkative_autoencoder/scripts/run_vllm_gpt-oss.sh
#   (script installs Ninja/CMake for FlashInfer JIT, waits for server readiness)
# - Generate ~4k-token messages with thinking:
#   uv run python -m talkative_autoencoder.scripts.generate_thinking_dataset ...
#   (stores messages-only; prints first example and token stats at end)
# - Pretokenize the saved dataset directory with orig_tokenizer_name=openai/gpt-oss-20b
#   (native 'thinking' preserved for gpt-oss; <think> inlined otherwise)

.PHONY: all setup help serve-docs flash-attention sync-env wandb-login check-b200 link-consistency-lens link-venv unlink-venv

# Default target
all: setup

# Where to place the symlink (override with: make SYMLINK_PATH=/path/consistency-lens link-consistency-lens)
#SYMLINK_PATH ?= $(abspath $(CURDIR)/../consistency-lens)

# Help target
help:
	@echo "Talkative Probes Autoencoder Setup"
	@echo "  make              - Initial setup (install uv, create env, sync dependencies)"
	@echo "  make sync-env     - Sync/update the environment with latest dependencies"
	@echo "  make wandb-login  - Login to Weights & Biases (requires WANDB_API_KEY env var)"
	@echo "  make flash-attention - Install Flash Attention 2 (optional, for performance)"
	@echo "  make link-venv    - Symlink .venv -> machine-local uv environment for editors/tools"
	@echo "  make serve-docs  - Print vLLM serving instructions (gpt-oss-20b, 8× H100)"
	@echo ""
	@echo "After setup, use 'uv run' for all Python commands:"
	@echo "  uv run python scripts/01_train.py"
	@echo "  uv run pytest tests/"
	@echo ""
	@echo "No manual venv activation needed!"

serve-docs:
	@echo "vLLM serving (gpt-oss-20b) - Main path: single 8× H100 node (TP=8)"
	@echo "  sbatch talkative_autoencoder/scripts/run_vllm_gpt-oss.sh"
	@echo "  # Script starts vLLM with --tensor-parallel-size 8 and waits for readiness"
	@echo "  # Then runs: python -m talkative_autoencoder.scripts.generate_thinking_dataset ..."
	@echo ""
	@echo "Generator script saves messages-only with native 'thinking' for gpt-oss and logs token stats."
	@echo "Pretokenize the saved directory with orig_tokenizer_name=openai/gpt-oss-20b."
	@echo "For multi-node, either run multiple independent servers and pass --server-urls, or use Ray TP."

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
		uv lock --index-strategy unsafe-best-match --prerelease allow && \
		if [ -n "$$LIGHTWEIGHT" ]; then \
			uv sync --group dev --index-strategy unsafe-best-match --prerelease allow; \
		else \
			uv sync --group dev --frozen --index-strategy unsafe-best-match --prerelease allow; \
		fi
	@$(MAKE) link-venv
	@# Check if on B200 node and remove packages if needed
	@if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q "B200"; then \
		echo "Detected B200 node - removing incompatible packages..."; \
		. ./scripts/ensure_env.sh && \
		uv remove bitsandbytes cairosvg triton-kernels || true; \
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
	uv lock --index-strategy unsafe-best-match --prerelease allow && \
	uv sync --group dev --frozen --index-strategy unsafe-best-match --prerelease allow
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

# Create/update .venv symlink pointing to the active uv environment
link-venv:
	@echo ".venv -> active uv environment"
	@if [ -e ".venv" ] && [ ! -L ".venv" ]; then \
		echo "Error: .venv exists and is not a symlink"; \
		exit 1; \
	fi
	@. ./scripts/ensure_env.sh >/dev/null 2>&1 && \
		echo "Linking .venv to $$UV_PROJECT_ENVIRONMENT" && \
		ln -sfn "$$UV_PROJECT_ENVIRONMENT" .venv && \
		echo "✓ .venv symlink ready"

# Remove .venv symlink (safe)
unlink-venv:
	@if [ -L ".venv" ]; then \
		rm -f .venv && echo "✓ Removed .venv symlink"; \
	else \
		echo ".venv is not a symlink or does not exist"; \
	fi
