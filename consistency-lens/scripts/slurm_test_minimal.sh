#!/bin/bash
#SBATCH --job-name=test-gpu
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=330702be7061

# Minimal test script to verify SLURM GPU allocation

echo "=== SLURM Test Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"

# Check GPU allocation
echo -e "\n=== GPU Information ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
else
    echo "nvidia-smi not found"
fi

echo -e "\n=== Environment Variables ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"

echo -e "\n=== Test Complete ==="