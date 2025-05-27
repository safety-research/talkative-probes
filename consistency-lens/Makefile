.PHONY: all setup install uv help

# Default target - do everything
all: setup

# Help target
help:
	@echo "Consistency Lens Setup"
	@echo "  make         - Do everything (setup environment and install)"
	@echo "  make all     - Same as make"
	@echo "  make setup   - Create virtual environment and install dependencies"
	@echo "  make uv      - Install uv package manager"
	@echo "  make install - Install dependencies (assumes venv exists)"

# Install uv if not present
uv:
	@command -v uv >/dev/null 2>&1 || { \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "uv installed! You may need to restart your shell or run: source $$HOME/.cargo/env"; \
	}

# Setup environment
setup: uv
	@if [ -f /workspace/kitf/dotfiles/setup_env.sh ]; then \
		echo "Running setup_env.sh..."; \
		/workspace/kitf/dotfiles/setup_env.sh; \
	fi
	PATH="$$HOME/.local/bin:$$PATH" uv venv
	PATH="$$HOME/.local/bin:$$PATH" uv pip install -e .
	@if [ -f .env ]; then \
		echo "Configuring wandb..."; \
		. .venv/bin/activate && \
		export $$(grep WANDB_API_KEY .env | xargs) && \
		wandb login $$WANDB_API_KEY; \
	fi
	@echo "Setup complete! Activate with: source .venv/bin/activate"

# Quick install (assumes venv exists)
install:
	PATH="$$HOME/.local/bin:$$PATH" uv pip install -e .
