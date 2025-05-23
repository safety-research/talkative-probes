# SLURM Setup Guide

## Quick Fix for Partition Error

The error "invalid partition specified: gpu" means your cluster doesn't have a partition named "gpu".

### Solution 1: Use Default Partition (Recommended)
The partition lines have been removed from all scripts. SLURM will use your cluster's default partition.

### Solution 2: Set Environment Variable
```bash
# Find your cluster's GPU partition name
sinfo -o "%P %a %l %D %N"

# Set it as environment variable
export SBATCH_PARTITION=<your_partition_name>

# Now run experiments
./scripts/submit_with_dumping.sh ss-frozen
```

### Solution 3: Edit Scripts
If you need specific SLURM options, add them to the scripts:
```bash
#SBATCH --partition=<your_partition>
#SBATCH --account=<your_account>
#SBATCH --qos=<your_qos>
```

## Testing Your Setup

Run the test script to check your SLURM configuration:
```bash
./scripts/test_slurm.sh
```

## Running from Correct Directory

Always run from the project root:
```bash
cd /path/to/consistency-lens
./scripts/submit_with_dumping.sh ss-frozen
```

Or from the scripts directory:
```bash
cd /path/to/consistency-lens/scripts
./submit_with_dumping.sh ss-frozen
```

## Common SLURM Configurations

### For NERSC/Perlmutter
```bash
#SBATCH --partition=gpu
#SBATCH --constraint=gpu
#SBATCH --account=<your_account>
```

### For Stanford Sherlock
```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
```

### For Generic Clusters
```bash
# Often no partition needed - uses default
#SBATCH --gres=gpu:1
```

## Debugging Submission Errors

1. **Check available partitions:**
   ```bash
   sinfo
   ```

2. **Check your default account:**
   ```bash
   sacctmgr show user $USER
   ```

3. **Test submission with simple job:**
   ```bash
   sbatch --wrap="hostname" --time=00:01:00
   ```

## Modified Scripts

All SLURM scripts now have:
- ✅ No hardcoded partition
- ✅ Project root directory handling
- ✅ Better error messages
- ✅ GPU allocation via `--gpus=N`

The scripts will work with your cluster's defaults. Add cluster-specific options only if needed.