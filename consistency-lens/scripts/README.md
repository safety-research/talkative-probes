# Consistency Lens Scripts

This directory contains all scripts for running Consistency Lens experiments, from activation extraction to model training and evaluation.

## Quick Start

```bash
# Run complete experiment with automatic dependency management
./submit_with_dumping.sh ss-frozen

# Or manually: 
# 1. Extract activations (8 GPUs)
sbatch slurm_dump_activations_minimal.sh

# 2. Train model (1 GPU) 
sbatch slurm_simplestories_frozen_minimal.sh
```

## Available Experiments

We provide 6 pre-configured experiments:

| Experiment | Config | Dataset | Base Model | Training |
|------------|--------|---------|------------|----------|
| `ss-frozen` | `simplestories_frozen.yaml` | SimpleStories | Frozen throughout | 10 epochs |
| `ss-unfreeze` | `simplestories_unfreeze.yaml` | SimpleStories | Unfreeze after epoch 1 | 30 epochs |
| `gpt2-frozen` | `gpt2_frozen.yaml` | OpenWebText | Frozen throughout | Standard |
| `gpt2-unfreeze` | `gpt2_unfreeze.yaml` | OpenWebText | Unfreeze after 10k steps | Extended |
| `gpt2-pile-frozen` | `gpt2_pile_frozen.yaml` | The Pile | Frozen throughout | Standard |
| `gpt2-pile-unfreeze` | `gpt2_pile_unfreeze.yaml` | The Pile | Unfreeze after epoch 1 | Extended |

## Script Categories

### 🎯 Core Scripts
- `00_dump_activations.py` - Single-GPU activation extraction
- `00_dump_activations_multigpu.py` - Multi-GPU parallel extraction (8x faster)
- `01_train.py` - Main training script with Hydra configuration
- `02_eval.py` - Model evaluation and metrics computation

### 🚀 SLURM Submission
- `submit_with_dumping.sh` - Smart wrapper with dependency management
- `slurm_dump_activations_minimal.sh` - 8-GPU activation dumping job
- `slurm_simplestories_*.sh` - SimpleStories experiment jobs
- `slurm_gpt2_*.sh` - GPT-2 experiment jobs

### 🔧 Utilities
- `pretokenize_dataset.py` - Pre-tokenize for 5x faster dumping
- `launch_multigpu_dump_optimized.sh` - CPU-optimized multi-GPU launcher
- `train_with_compile.sh` - Training with torch.compile wrapper

## Workflow Details

### Step 1: Activation Extraction (8 GPUs, ~30 min)

```bash
# Automatic with smart wrapper
./submit_with_dumping.sh experiment-name

# Or manual submission
sbatch slurm_dump_activations_minimal.sh
```

**Output**: Sharded `.pt` files in `data/activations/DATASET/MODEL/layer_X/`

### Step 2: Model Training (1 GPU, 4-8 hours)

Training automatically starts after dumping completes if using the wrapper.

**Key features**:
- torch.compile for 2x speedup (after initial compilation)
- Automatic checkpointing every 100 steps
- WandB integration for real-time monitoring
- Progressive unfreezing support

### Step 3: Evaluation

```bash
python scripts/02_eval.py \
    --checkpoint outputs/checkpoints/RUN_NAME/best_checkpoint.pt \
    --save_results
```

## Performance Optimization

### Pretokenization (5x Speedup)

```bash
# One-time preprocessing
python scripts/pretokenize_dataset.py --config_path conf/experiment.yaml

# Then use pretokenized data
PRETOKENIZE=true sbatch slurm_simplestories_frozen.sh
```

### CPU Threading (Critical for H100 Nodes)

The optimized launcher automatically configures threading:
```bash
./launch_multigpu_dump_optimized.sh config.yaml output_dir
```

## Monitoring & Troubleshooting

### Check Job Status
```bash
squeue -u $USER
tail -f logs/job_name_*.out
grep "wandb.ai" logs/job_name_*.err  # Get WandB URL
```

### Common Issues

1. **Slow initial training steps**: Normal - torch.compile optimization
2. **Module not found errors**: Already handled - scripts work without module system
3. **Missing activations**: Use `submit_with_dumping.sh` for automatic handling
4. **Hydra warnings**: Fixed - configs include `_self_` directive

### Resource Usage

| Task | GPUs | Time | Memory |
|------|------|------|--------|
| Activation Dumping (5M model) | 8 | ~30 min | 20GB/GPU |
| Training (frozen base) | 1 | 4-6 hours | 40GB |
| Training (unfreezing) | 1 | 8-12 hours | 60GB |

## Advanced Usage

### Custom Configurations

```bash
# Override any parameter
python scripts/01_train.py \
    --config-name=simplestories_frozen \
    learning_rate=5e-4 \
    t_text=15 \
    +wandb.name="custom_experiment"
```

### Resume Training

```bash
python scripts/01_train.py \
    resume=outputs/checkpoint.pt \
    wandb_resume_id=run_id
```

## File Organization

```
scripts/
├── Core functionality (*.py)
├── SLURM jobs (slurm_*.sh)
├── Launchers (launch_*.sh)
├── Smart wrappers (submit_*.sh)
└── Documentation (README.md)
```

For implementation details, see the main project [README](../README.md).