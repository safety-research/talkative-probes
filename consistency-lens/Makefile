.PHONY: setup install uv help

# Default target
help:
	@echo "Consistency Lens Setup"
	@echo "  make uv      - Install uv package manager"
	@echo "  make setup   - Create virtual environment and install dependencies"
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
	uv venv
	uv pip install -e .
	@echo "Setup complete! Activate with: source .venv/bin/activate"

# Quick install (assumes venv exists)
install:
	uv pip install -e .