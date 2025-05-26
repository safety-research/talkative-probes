# Consistency Lens Scripts

This directory contains all scripts for running Consistency Lens experiments, from activation extraction to model training and evaluation.

## Quick Start

The `submit_with_dumping.sh` script works on both SLURM and non-SLURM environments:

```bash
# Start new experiment (auto-detects environment and GPUs)
./submit_with_dumping.sh ss-frozen

# Resume from checkpoint (find path in outputs/checkpoints/)
./submit_with_dumping.sh ss-frozen false outputs/checkpoints/run_name/checkpoint_step5000.pt

# Resume with WandB run ID (get from WandB dashboard)
./submit_with_dumping.sh ss-frozen false outputs/checkpoints/run_name/checkpoint_step5000.pt abc123xyz

# SLURM only: Use specific nodes
./submit_with_dumping.sh ss-frozen false "" "" node001,node002

# Non-SLURM: Specify GPU count (auto-detected if omitted)
./submit_with_dumping.sh ss-frozen false "" "" "" 4

# Force direct execution even on SLURM systems
FORCE_DIRECT=true ./submit_with_dumping.sh ss-frozen

# Manual SLURM submission (deprecated - use wrapper above):
sbatch slurm_dump_activations_minimal.sh conf/config.yaml
sbatch slurm_simplestories_frozen.sh
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

### 🚀 Job Management
- `submit_with_dumping.sh` - Smart wrapper (SLURM + non-SLURM) with dependency management
- `slurm_dump_activations_minimal.sh` - 8-GPU activation dumping job (SLURM only)
- `slurm_simplestories_*.sh` - SimpleStories experiment jobs (SLURM only)
- `slurm_gpt2_*.sh` - GPT-2 experiment jobs (SLURM only)

### 🔧 Utilities
- `pretokenize_dataset.py` - Pre-tokenize for 5x faster dumping
- `launch_multigpu_dump_optimized.sh` - CPU-optimized multi-GPU launcher
- `train_with_compile.sh` - Training with torch.compile wrapper

## Workflow Details

### Step 1: Activation Extraction

```bash
# Automatic with smart wrapper (works on both SLURM and non-SLURM)
./submit_with_dumping.sh experiment-name

# Non-SLURM: Direct execution with specified GPUs
./submit_with_dumping.sh experiment-name false "" "" "" 8

# SLURM: Manual submission (legacy approach)
sbatch slurm_dump_activations_minimal.sh conf/config.yaml
```

**Performance**: 
- **SLURM**: 8 GPUs, ~30 min via job queue
- **Non-SLURM**: N GPUs, runs immediately in foreground

**Output**: Sharded `.pt` files in `data/activations/DATASET/MODEL/layer_X/`

### Step 2: Model Training

Training automatically starts after dumping completes if using the wrapper.

**Environment Differences**:
- **SLURM**: Jobs submitted with dependencies, run when resources available  
- **Non-SLURM**: Runs sequentially in foreground, immediate execution

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
# Resume via wrapper script (recommended)
./submit_with_dumping.sh ss-frozen false outputs/checkpoint.pt wandb_run_id

# Or manually
python scripts/01_train.py \
    resume=outputs/checkpoint.pt \
    wandb_resume_id=run_id

# SLURM environment variables (used by wrapper)
export RESUME_CHECKPOINT="outputs/checkpoint.pt"
export WANDB_RESUME_ID="abc123xyz"
sbatch slurm_simplestories_frozen.sh
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