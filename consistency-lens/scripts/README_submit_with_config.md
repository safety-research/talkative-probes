# submit_with_config.sh - Extensible Config-Based Submission Script

This script provides a more extensible and maintainable approach to submitting experiments compared to the original `submit_with_dumping.sh`.

## Key Improvements

1. **Config-Driven**: All settings are extracted from the YAML config file instead of being hardcoded in the script
2. **Single Entry Point**: Just pass a config file - no need to know experiment names
3. **Minimal Bash Logic**: Uses a Python helper (`extract_config_settings.py`) to cleanly extract settings
4. **Hydra Support**: Properly handles Hydra config composition with defaults
5. **Extensible**: Adding new experiments just requires creating a new YAML config

## Usage

```bash
# Basic usage - just pass a config file
./scripts/submit_with_config.sh conf/simplestories_frozen.yaml

# Force re-dump activations
./scripts/submit_with_config.sh conf/gpt2_frozen.yaml force-redump

# Resume from checkpoint
./scripts/submit_with_config.sh conf/simplestories_frozen.yaml false outputs/ckpt_step_1000.pt

# Resume with WandB ID
./scripts/submit_with_config.sh conf/simplestories_frozen.yaml false outputs/ckpt_step_1000.pt abc123xyz

# Specify SLURM nodes (SLURM only)
./scripts/submit_with_config.sh conf/gpt2_pile.yaml false "" "" node001,node002

# Specify GPU count (non-SLURM)
./scripts/submit_with_config.sh conf/gpt2_pile.yaml false "" "" "" 4

# Force direct execution even on SLURM
FORCE_DIRECT=true ./scripts/submit_with_config.sh conf/simplestories_frozen.yaml
```

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