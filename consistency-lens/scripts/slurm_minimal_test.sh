#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:1
#SBATCH --nodelist=330702be7061

echo "Test job running on $(hostname)"
nvidia-smi