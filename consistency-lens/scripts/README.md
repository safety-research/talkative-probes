# Consistency Lens Scripts

This directory contains all scripts for running Consistency Lens experiments, from activation extraction to model training and evaluation.

## Quick Start

The new `submit_with_config.sh` script provides a unified interface for all experiments:

```bash
# Start new experiment - just pass a config file!
./scripts/submit_with_config.sh conf/simplestories_frozen.yaml

# Resume from checkpoint
./scripts/submit_with_config.sh conf/simplestories_frozen.yaml false outputs/checkpoint.pt

# Resume with WandB run ID
./scripts/submit_with_config.sh conf/simplestories_frozen.yaml false outputs/checkpoint.pt abc123xyz

# SLURM only: Use specific nodes
./scripts/submit_with_config.sh conf/gpt2_frozen.yaml false "" "" node001,node002

# Non-SLURM: Specify GPU count (auto-detected if omitted)
./scripts/submit_with_config.sh conf/gpt2_frozen.yaml false "" "" "" 4

# Force direct execution even on SLURM systems
FORCE_DIRECT=true ./scripts/submit_with_config.sh conf/simplestories_frozen.yaml
```

## Available Configurations

All experiments are now config-driven. Just pass the appropriate YAML file:

| Config File | Dataset | Model | Training Type | Duration |
|------------|---------|--------|---------------|----------|
| `conf/simplestories_frozen.yaml` | SimpleStories | 5M | Frozen throughout | 10 epochs |
| `conf/simplestories_unfreeze.yaml` | SimpleStories | 5M | Unfreeze after epoch 1 | 30 epochs |
| `conf/gpt2_frozen.yaml` | OpenWebText | GPT-2 | Frozen throughout | Standard |
| `conf/gpt2_unfreeze.yaml` | OpenWebText | GPT-2 | Unfreeze after 10k steps | Extended |
| `conf/gpt2_pile_frozen.yaml` | The Pile | GPT-2 | Frozen throughout | Standard |
| `conf/gpt2_pile_unfreeze.yaml` | The Pile | GPT-2 | Unfreeze after epoch 1 | Extended |

## Script Categories

### 🎯 Core Scripts
- `00_dump_activations.py` - Single-GPU activation extraction
- `00_dump_activations_multigpu.py` - Multi-GPU parallel extraction (8x faster)
- `01_train.py` - Main training script with Hydra configuration
- `02_eval.py` - Model evaluation and metrics computation

### 🚀 Job Management
- **`submit_with_config.sh`** - NEW! Config-driven submission (SLURM + non-SLURM)
- `slurm_dump_activations_flexible.sh` - Flexible GPU activation dumping (auto-detects GPUs)

### 🔧 Utilities
- `pretokenize_dataset.py` - Pre-tokenize for 5x faster dumping
- `extract_config_settings.py` - Helper to extract settings from YAML configs
- `launch_multigpu_dump_optimized.sh` - CPU-optimized multi-GPU launcher
- `launch_multigpu_dump_pretokenized.sh` - Launcher for pre-tokenized data
- `train_with_compile.sh` - Training with torch.compile wrapper

### 📁 Legacy Scripts
Scripts in `scripts/legacy/` have been replaced by the new extensible system. See [legacy/README_LEGACY.md](legacy/README_LEGACY.md) for migration guide.

## Workflow Details

### Complete Pipeline (Automatic)

```bash
# The script handles everything: pretokenization → dumping → training
./scripts/submit_with_config.sh conf/your_experiment.yaml
```

**What happens automatically**:
1. **Pretokenization** (if needed) - 5x speedup for dumping
2. **Activation dumping** - Multi-GPU extraction
3. **Training** - With proper dependencies and resource allocation

### Manual Steps (if needed)

#### Step 1: Pretokenization (Optional but Recommended)
```bash
python scripts/pretokenize_dataset.py --config-path=conf --config-name=your_experiment
```

#### Step 2: Activation Extraction
```bash
# Multi-GPU dumping
./scripts/launch_multigpu_dump_pretokenized.sh conf/your_experiment.yaml

# Or with SLURM
sbatch scripts/slurm_dump_activations_flexible.sh conf/your_experiment.yaml
```

#### Step 3: Training
```bash
python scripts/01_train.py --config-path=conf --config-name=your_experiment
```

#### Step 4: Evaluation
```bash
python scripts/02_eval.py \
    checkpoint=outputs/RUN_NAME/checkpoint.pt \
    evaluation.save_results=true
```

## Key Improvements in New System

### 1. Config-Driven Everything
- No more hardcoded experiment logic
- All settings extracted from YAML configs
- Easy to add new experiments

### 2. Consistent Naming
- WandB runs: `SS_5M_L5_frozen_lr1e-3_t10_20ep_0527_1258`
- Job names: `SimpleStories_5M_L5_frozen`
- Informative and sortable

### 3. Automatic Resource Detection
- GPU count auto-detected
- Works on both SLURM and non-SLURM
- Flexible activation dumping adapts to available GPUs

### 4. Always Optimized
- Pretokenization enabled by default (5x speedup)
- Proper CPU threading configuration
- torch.compile enabled for training

## Performance Optimization

### Resource Usage

| Task | GPUs | Time | Memory |
|------|------|------|--------|
| Pretokenization | 0 (CPU) | ~5 min | 10GB |
| Activation Dumping (5M model) | 8 | ~30 min | 20GB/GPU |
| Activation Dumping (GPT-2) | 8 | ~1 hour | 40GB/GPU |
| Training (frozen base) | 1 | 4-6 hours | 40GB |
| Training (unfreezing) | 1 | 8-12 hours | 60GB |

### CPU Threading

The scripts automatically configure optimal threading:
```bash
# Automatic in submit_with_config.sh
# Manual if needed:
export OMP_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=true
```

## Monitoring & Troubleshooting

### Check Job Status
```bash
# SLURM
squeue -u $USER
tail -f logs/*_1234.out  # Replace 1234 with job ID

# Non-SLURM
# Output appears directly in terminal
```

### Common Issues

1. **"DependencyNeverSatisfied"**: Previous job failed - check logs
2. **Missing activations**: Script will automatically pretokenize and dump
3. **Out of memory**: Reduce batch size in config
4. **Slow first steps**: Normal - torch.compile optimization

### Finding Logs

```bash
# Pretokenization logs
ls -ltr logs/pretokenize_*.out

# Dumping logs  
ls -ltr logs/dump_*.out

# Training logs
ls -ltr logs/*_train_*.out

# Get WandB URL from training log
grep "wandb.ai" logs/*_train_*.out
```

## Advanced Usage

### Custom Configurations

```bash
# Create your own config inheriting from base
cat > conf/my_experiment.yaml << EOF
defaults:
  - config
  - _self_

model_name: "gpt2-medium"
learning_rate: 5e-4
t_text: 15
num_train_epochs: 5
EOF

# Run it!
./scripts/submit_with_config.sh conf/my_experiment.yaml
```

### Override Parameters

```bash
# Via command line (manual training)
python scripts/01_train.py \
    --config-path=conf \
    --config-name=simplestories_frozen \
    learning_rate=5e-4 \
    t_text=15
```

### Resume Training

The submit script handles resumption automatically:
```bash
# Find checkpoint
ls outputs/*/checkpoint_*.pt

# Resume
./scripts/submit_with_config.sh conf/simplestories_frozen.yaml \
    false \
    outputs/SS_5M_L5_frozen_lr1e-3_t10_20ep_0527_1258/checkpoint_step5000.pt \
    abc123xyz  # Optional: WandB run ID
```

## File Organization

```
scripts/
├── Core functionality (*.py)
├── Job submission (submit_*.sh)
├── SLURM templates (slurm_*.sh)  
├── Launchers (launch_*.sh)
├── Utilities (extract_*.py, *.sh)
├── Documentation (README*.md)
└── legacy/                        # Old scripts for reference
    ├── submit_with_dumping.sh     # Replaced by submit_with_config.sh
    ├── slurm_simplestories_*.sh   # No longer needed
    ├── slurm_gpt2_*.sh           # No longer needed
    └── README_LEGACY.md          # Migration guide
```

For implementation details, see the main project [README](../README.md).