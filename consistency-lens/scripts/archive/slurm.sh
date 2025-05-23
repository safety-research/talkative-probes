#!/bin/bash
#SBATCH --job-name=lens
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=330702be7061
#SBATCH --constraint=h100

# module load cuda/12.4

deepspeed --num_gpus 8 01_train.py --deepspeed_config ../config/ds_stage2.yaml --config ../config/lens.yaml