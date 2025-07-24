# Dependency Management

This project uses `uv` for dependency management with `pyproject.toml` as the configuration file.

## Installation

### Initial Setup
```bash
# Create virtual environment and install dependencies
uv sync
```

### For Development
```bash
# Install with all optional dependencies including dev tools
uv sync --all-extras
```

## Managing Dependencies

### Adding Dependencies

#### Production Dependencies
```bash
# Add a new production dependency
uv add fastapi

# Add with version constraint
uv add "fastapi>=0.104.0"
```

#### Development Dependencies
```bash
# Add development-only dependencies
uv add --dev pytest black ruff
```

### Removing Dependencies
```bash
# Remove a dependency
uv remove package-name
```

### Updating Dependencies
```bash
# Update all dependencies to latest compatible versions
uv sync --upgrade

# Update a specific package
uv add --upgrade fastapi
```

### Viewing Dependencies
```bash
# Show installed packages
uv pip list

# Show dependency tree
uv pip tree
```

## Lock File

The `uv.lock` file is automatically managed by `uv` and ensures reproducible installations. This file should be committed to version control.

```bash
# Sync dependencies from lock file (reproducible install)
uv sync --frozen
```

## Why uv?

- **Fast**: Written in Rust, significantly faster than pip
- **Reliable**: Built-in resolver prevents dependency conflicts
- **Reproducible**: Automatic lock file generation
- **Modern**: Follows latest Python packaging standards
- **Simple**: Intuitive commands similar to other modern package managers

## Common Commands Reference

| Task | Command |
|------|---------|
| Install all deps | `uv sync` |
| Add a package | `uv add package` |
| Add dev package | `uv add --dev package` |
| Remove package | `uv remove package` |
| Update all | `uv sync --upgrade` |
| Update one package | `uv add --upgrade package` |
| Install from lock | `uv sync --frozen` |

## Legacy Note

This project previously used `requirements.txt`. We've migrated to `uv` with `pyproject.toml` for better dependency management and standards compliance.