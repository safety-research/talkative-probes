# Migration to uv/pyproject.toml

## What Changed

1. **Dependency Management**: Migrated from `requirements.txt` to `pyproject.toml`
2. **Package Manager**: Now using `uv` instead of `pip`
3. **Environment Setup**: Using `uv sync` instead of `pip install -r requirements.txt`

## Key Files

- `pyproject.toml`: Contains all dependency declarations and project metadata
- `uv.lock`: Automatically generated lock file for reproducible installs
- `DEPENDENCIES.md`: Documentation on how to use the new system
- `requirements.txt.backup`: Backup of the old requirements file

## Quick Start

```bash
# Initial setup (creates venv and installs deps)
uv sync

# Run the server
uv run uvicorn app.main:app --reload

# Run tests
uv run pytest

# Add a new dependency
uv add package-name

# Add a dev dependency
uv add --dev package-name
```

## Why This Change?

1. **Better dependency resolution**: uv's resolver prevents conflicts
2. **Faster installations**: uv is written in Rust and significantly faster
3. **Automatic lock files**: Ensures reproducible builds
4. **Standard compliance**: Following modern Python packaging standards
5. **Single source of truth**: All config in pyproject.toml

## For Deployment

All deployment scripts have been updated:
- `Dockerfile`: Uses `uv sync --frozen`
- `runpod_startup.sh`: Uses `uv sync`
- `DEPLOYMENT_INSTRUCTIONS.md`: Updated with new commands

## Main Project vs Backend

- **Main project** (`/talkative-probes/`): Uses `scripts/ensure_env.sh` for multi-node setups
- **Web backend** (`/website/backend/`): Uses standard `uv` commands directly

The backend has its own isolated environment with web-specific dependencies.