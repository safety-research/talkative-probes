# Legacy Scripts

This folder contains scripts that have been replaced by the new extensible submission system.

## Replaced Scripts

### Submission Scripts
- `submit_with_dumping.sh` - Replaced by `submit_with_config.sh`
  - The new script is more extensible and extracts settings from YAML configs
  - No more hardcoded experiment logic

### Training Scripts  
- `slurm_simplestories_frozen.sh` - No longer needed
- `slurm_simplestories_unfreeze.sh` - No longer needed
- `slurm_gpt2_frozen.sh` - No longer needed
- `slurm_gpt2_unfreeze.sh` - No longer needed
- `slurm_gpt2_pile.sh` - No longer needed
- `slurm_gpt2_pile_unfreeze.sh` - No longer needed
  - All replaced by direct training execution in `submit_with_config.sh`

### Dumping Scripts
- `slurm_dump_activations_minimal.sh` - Replaced by `slurm_dump_activations_flexible.sh`
  - The new script automatically detects the number of GPUs instead of hardcoding 4
- `launch_multigpu_dump_optimized.sh` - Replaced by `launch_multigpu_dump.sh`
- `launch_multigpu_dump_pretokenized.sh` - Replaced by `launch_multigpu_dump.sh`
  - Merged into single script that always uses pretokenization (5x speedup)
  - No need for conditional logic since pretokenization is always beneficial

### Utility Scripts
- `train_with_compile.sh` - No longer needed
  - torch.compile is now integrated in the main training script
  - Controlled by `compile_models: true` in config (default)
  - TORCHINDUCTOR_CACHE_DIR is set automatically

## Migration Guide

To run experiments with the new system:

```bash
# Instead of:
./scripts/submit_with_dumping.sh ss-frozen

# Use:
./scripts/submit_with_config.sh conf/simplestories_frozen.yaml
```

The new system:
- Automatically handles pretokenization for 5x speedup
- Extracts all settings from YAML configs
- Generates consistent, informative names for WandB and jobs
- Works on both SLURM and non-SLURM environments
- No need to create new scripts for new experiments

## Archived: May 27, 2025