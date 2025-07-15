# submit_with_config.sh - Extensible Config-Based Submission Script

This script provides a more extensible and maintainable approach to submitting experiments compared to the original `submit_with_dumping.sh`.

## Key Improvements

1. **Config-Driven**: All settings are extracted from the YAML config file instead of being hardcoded in the script
2. **Single Entry Point**: Just pass a config file - no need to know experiment names
3. **Minimal Bash Logic**: Uses a Python helper (`extract_config_settings.py`) to cleanly extract settings
4. **Hydra Support**: Properly handles Hydra config composition with defaults
5. **Extensible**: Adding new experiments just requires creating a new YAML config

## Usage

All arguments now use Hydra-style `key=value` syntax for cleaner command lines:

```bash
# Basic usage - config file as first positional arg (backwards compatible)
./scripts/submit_with_config.sh conf/simplestories_frozen.yaml

# Or using Hydra style (recommended)
./scripts/submit_with_config.sh config=conf/simplestories_frozen.yaml

# Force re-dump activations
./scripts/submit_with_config.sh config=conf/gpt2_frozen.yaml force_redump=true

# Resume from checkpoint
./scripts/submit_with_config.sh config=conf/simplestories_frozen.yaml resume_checkpoint=outputs/ckpt_step_1000.pt

# Resume with WandB ID
./scripts/submit_with_config.sh config=conf/simplestories_frozen.yaml \
    resume_checkpoint=outputs/ckpt_step_1000.pt \
    wandb_resume_id=abc123xyz

# Specify SLURM nodes (SLURM only)
./scripts/submit_with_config.sh config=conf/gpt2_pile.yaml nodelist=node001,node002

# Specify GPU count (non-SLURM)
./scripts/submit_with_config.sh config=conf/gpt2_pile.yaml num_gpus=4

# Add run suffix for organization
./scripts/submit_with_config.sh config=conf/gpt2_frozen.yaml run_suffix=experiment_v2

# Pass Hydra overrides to training script
./scripts/submit_with_config.sh config=conf/simplestories_frozen.yaml \
    learning_rate=1e-3 \
    batch_size=16 \
    num_epochs=10

# Combined example with multiple options
./scripts/submit_with_config.sh config=conf/gpt2_frozen.yaml \
    force_redump=true \
    run_suffix=high_lr \
    learning_rate=5e-4 \
    gumbel_temperature.schedule.start_value=2.0

# Force direct execution even on SLURM
FORCE_DIRECT=true ./scripts/submit_with_config.sh config=conf/simplestories_frozen.yaml
```

### Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `config` | Config file path (required) | - |
| `force_redump` | Force re-dump activations even if they exist | false |
| `force_retokenize` | Force re-tokenize dataset | false |
| `resume_checkpoint` | Path to checkpoint to resume from | - |
| `wandb_resume_id` | WandB run ID to resume | - |
| `nodelist` | SLURM nodes to use (comma-separated) | current hostname |
| `num_gpus` | Number of GPUs to use | auto-detected |
| `run_suffix` | Suffix to add to run name | - |
| Any other `key=value` | Passed as Hydra override to training script | - |

## How It Works

1. **Config Extraction**: The Python helper `extract_config_settings.py` uses Hydra to compose the full config and extract key settings:
   - Model name and layer
   - Dataset information
   - Activation output directories
   - Training parameters
   - Freeze schedule settings

2. **Training Script Mapping**: The helper automatically determines the appropriate training script based on:
   - Config file name (e.g., `simplestories_frozen.yaml` → `slurm_simplestories_frozen.sh`)
   - Model type and freeze settings as fallback

3. **Activation Checking**: The script checks if activations already exist in the expected locations

4. **Job Submission**: Handles both SLURM and direct execution modes automatically

## Adding New Experiments

To add a new experiment:

1. Create a new YAML config in `conf/` (can inherit from `config.yaml`)
2. Create a corresponding SLURM training script in `scripts/` following the naming convention
3. That's it! The script will automatically handle the rest

## Config Requirements

Your config should include:
- `model_name`: The HuggingFace model to use
- `layer_l`: The layer to extract activations from
- `activation_dumper`: Settings for the activation dumping process
- `freeze_schedule`: (Optional) Settings for progressive unfreezing

## Differences from Original Script

- **No hardcoded experiment logic**: All experiment-specific settings come from configs
- **Cleaner separation**: Python handles config parsing, bash handles job orchestration
- **More maintainable**: Adding experiments doesn't require modifying the submission script
- **Better error handling**: Clear error messages when configs are missing required fields

## Python Helper Details

The `extract_config_settings.py` script:
- Uses Hydra to properly compose configs with defaults
- Extracts all relevant settings into bash-friendly KEY=VALUE format
- Automatically determines the training script based on config name
- Handles model name cleaning for directory paths