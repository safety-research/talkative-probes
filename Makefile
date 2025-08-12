# Default target (running just 'make') is 'setup'
.PHONY: setup clean lightweight lightweight-run

# Define SHELL for compatibility and features
SHELL := /bin/zsh

# Define virtual environment directory
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python

setup: $(VENV_DIR)/bin/activate # Depends on the venv being created
	@echo "--- Installing project dependencies and submodules ---"
	@uv pip install -e ./safety-tooling
	@uv pip install -e ./dictionary_learning
	@uv pip install . # Install main project dependencies from pyproject.toml
	@echo "--- Setup complete! ---"
	@echo "To activate the virtual environment, run: source $(VENV_DIR)/bin/activate"

# Lightweight install: only talkative_autoencoder and website
lightweight:
	@echo "--- Lightweight setup (talkative_autoencoder + website only) ---"
	@LIGHTWEIGHT=1 $(MAKE) -C talkative_autoencoder setup
	@$(MAKE) -C talkative_autoencoder/website setup
	@echo "--- Lightweight setup complete ---"
	@echo "Next: cd talkative_autoencoder/website && make run"

# Lightweight setup and run website backend immediately
lightweight-run: lightweight
	@$(MAKE) -C talkative_autoencoder/website run

$(VENV_DIR)/bin/activate: pyproject.toml # Depend on pyproject.toml instead of requirements.txt
	@uv --version >/dev/null 2>&1 || { echo >&2 "Error: uv not found or not executable. Please install uv: https://github.com/astral-sh/uv"; exit 1; }
	@echo "--- Initializing and updating submodules ---"
	@git submodule update --init --recursive
	@echo "--- Creating virtual environment at $(VENV_DIR) ---"
	@uv venv $(VENV_DIR)
	@touch $(VENV_DIR)/bin/activate # Mark venv as created

clean:
	@echo "--- Cleaning up virtual environment and build artifacts ---"
	@rm -rf $(VENV_DIR)
	@find . -type f -name '*.py[co]' -delete
	@find . -type d -name '__pycache__' -delete

jupyter:
	@echo "---installing jupyter and seaborn"
	@source $(VENV_DIR)/bin/activate
	@uv add jupyter seaborn plotly
# Target to install git hooks using pre-commit
.PHONY: hooks
hooks:
	@echo "--- Installing Git hooks ---"
	@pre-commit install --overwrite --install-hooks --hook-type pre-commit --hook-type post-checkout --hook-type pre-push
	git checkout
