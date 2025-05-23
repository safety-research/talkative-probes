# Archived Scripts

This directory contains legacy scripts that were part of the development process but are no longer needed in the current workflow.

## Legacy Scripts Overview

### Test Scripts (Used during SLURM setup)
- `slurm_test_minimal.sh` - Basic GPU allocation test
- `slurm_minimal_test.sh` - Very basic test job  
- `test_slurm.sh` - SLURM configuration testing
- `test_submission.sh` - Submission testing

### Minimal Variants (Experimental versions)
- `slurm_simplestories_frozen_minimal.sh` - Minimal version of training script (replaced by full version)

### Alternative Launch Scripts (Replaced by optimized versions)
- `launch_multigpu_dump.sh` - Basic multi-GPU launcher (replaced by `launch_multigpu_dump_optimized.sh`)
- `launch_multigpu_dump_hydra.sh` - Hydra-only version (less flexible)
- `launch_multigpu_dump_pretokenized_hydra.sh` - Hydra-only pretokenized version

### Deprecated Submission Scripts (Replaced by submit_with_dumping.sh)
- `submit_gpt2_experiments.sh` - Old GPT-2 experiment launcher
- `submit_simplestories_experiments.sh` - Old SimpleStories experiment launcher
- `launch_gpt2_experiments.sh` - Alternative GPT-2 launcher
- `monitor_gpt2_jobs.sh` - Simple job monitoring (use `squeue -u $USER` instead)

### Old SLURM Scripts
- `slurm_dump_activations.sh` - Older activation dumping (replaced by `_minimal` version)
- `slurm.sh` - Very old script using DeepSpeed

## Current Workflow

Use these scripts instead:

```bash
# Main workflow
./submit_with_dumping.sh ss-frozen

# Production dumping  
./launch_multigpu_dump_optimized.sh
./launch_multigpu_dump_pretokenized.sh

# Individual training jobs
sbatch slurm_simplestories_frozen.sh
sbatch slurm_gpt2_frozen.sh
# etc.
```

These archived scripts are kept for reference but should not be used in production.