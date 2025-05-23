# SLURM Troubleshooting Guide

## Quick Diagnosis

Run this first to see what works:
```bash
./scripts/test_submission.sh
```

## Common Issues & Solutions

### 1. "Invalid feature specification" Error
**Cause**: `--constraint=h100` not valid on your cluster
**Fix**: Already removed from all scripts

### 2. "Invalid partition specified" Error  
**Cause**: `--partition=gpu` not valid on your cluster
**Fix**: Already removed from all scripts

### 3. Job submission returns no ID
**Cause**: SLURM rejecting the job due to invalid options
**Fix**: Use minimal scripts or test what works

## Minimal Script Versions

We've created minimal versions that should work on most clusters:

- `slurm_simplestories_frozen_minimal.sh` - Minimal training script
- `slurm_dump_activations_minimal.sh` - Minimal dumping script
- `slurm_minimal_test.sh` - Ultra-minimal test script

These only include:
- `--job-name` (required for identification)
- `--gres=gpu:N` (required for GPU allocation)
- `--time` (often required)
- `--output/--error` (for logging)

## Testing What Works

### Step 1: Test minimal submission
```bash
sbatch scripts/slurm_minimal_test.sh
```

### Step 2: If that fails, try inline
```bash
sbatch --job-name=test --gres=gpu:1 --wrap="nvidia-smi"
```

### Step 3: Check what your cluster needs
```bash
# Show sample job script from your cluster
scontrol show job <any_job_id> | grep -E "Command|Partition|QOS|Account"
```

## Customizing for Your Cluster

Once you know what works, you can:

1. **Set environment variables**:
   ```bash
   export SBATCH_PARTITION=your_partition
   export SBATCH_ACCOUNT=your_account
   ```

2. **Edit the minimal scripts** to add required options

3. **Use the wrapper with minimal scripts**:
   ```bash
   ./scripts/submit_with_dumping.sh ss-frozen
   ```

## If Nothing Works

Create the absolute minimal job:
```bash
cat > test.sh << 'EOF'
#!/bin/bash
#SBATCH --gres=gpu:1
nvidia-smi
EOF

sbatch test.sh
```

Then gradually add options until it breaks to find the problematic one.