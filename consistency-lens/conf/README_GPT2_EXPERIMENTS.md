# GPT-2 Consistency Lens Experiments

## Quick Start - SLURM Submission (Recommended)

### Use Smart Submission Wrapper
```bash
# Automatically handles activation dumping + training with dependencies
./scripts/submit_with_dumping.sh gpt2-frozen   # OpenWebText dataset
./scripts/submit_with_dumping.sh gpt2-pile     # The Pile dataset

# Force re-dump activations even if they exist
./scripts/submit_with_dumping.sh gpt2-frozen force-redump
```

**Benefits:**
- ✅ Checks if activations exist
- ✅ Submits 8-GPU dumping job if needed
- ✅ Submits 1-GPU training with dependency
- ✅ No GPU idle time - training waits for dumping

### Monitor Jobs
```bash
# Check job status
squeue -u $USER

# Monitor specific job output
tail -f logs/gpt2_frozen_*.out

# Check dumping progress
tail -f logs/dump_activations_*.out
```

## Manual Method (Not Recommended)

### Step 1: Dump Activations (if needed)
```bash
# Check if activations exist
ls data/activations/openwebtext_train/openai-community/gpt2/layer_6/train/

# If not, dump them (8 GPUs, ~2-4 hours)
sbatch scripts/slurm_dump_activations.sh conf/gpt2_frozen.yaml 6
```

### Step 2: Submit Training
```bash
# Will fail if activations don't exist
sbatch scripts/slurm_gpt2_frozen.sh
sbatch scripts/slurm_gpt2_pile.sh
```

## Manual Steps

### Step 1: Dump Activations (One-time)
```bash
# For OpenWebText
python scripts/00_dump_activations.py --config_path conf/gpt2_frozen.yaml

# For The Pile
python scripts/00_dump_activations.py --config_path conf/gpt2_pile_frozen.yaml

# Or use multi-GPU for faster dumping
torchrun --nproc_per_node=8 scripts/00_dump_activations_multigpu.py --config_path conf/gpt2_frozen.yaml
```

### Step 2: Train
```bash
# Single GPU (development) - Using Hydra config overrides
python scripts/01_train.py --config-name=gpt2_frozen

# Multi-GPU (production - 8xH100) 
torchrun --nproc_per_node=8 scripts/01_train.py --config-name=gpt2_frozen

# With specific overrides
python scripts/01_train.py --config-name=gpt2_frozen batch_size=2048 learning_rate=5e-5
```

### Step 3: Evaluate
```bash
python scripts/02_eval.py \
    --checkpoint outputs/checkpoints/gpt2_frozen_step10000_epoch1.pt \
    --save_results
```

## Experiment Configurations

### 1. **gpt2_frozen.yaml**
- Base model: Frozen throughout training
- Dataset: OpenWebText (5M samples)
- Good baseline to test if interpretability works

### 2. **gpt2_unfreeze.yaml**
- Base model: Frozen first 10k steps, then unfrozen
- Dataset: Same as frozen (for fair comparison)
- Tests if fine-tuning improves interpretability

### 3. **gpt2_pile_frozen.yaml**
- Base model: Frozen throughout
- Dataset: The Pile (10M samples, more diverse)
- Tests generalization across domains

## Key Hyperparameters

- **Learning Rate**: 1e-4 (projections), 1e-5 (unfrozen base model)
- **Batch Size**: 1024 total (128 per GPU × 8 GPUs)
- **Layer**: 6 (middle of GPT-2's 12 layers)
- **Decoder Tokens**: 10 tokens of explanation
- **Unfreeze Step**: 10,000 (for unfreezing experiment)

## Expected Timeline (8xH100)

1. **Activation Dumping**: ~2-4 hours per dataset
2. **Training**: 
   - Frozen: ~6-8 hours (2 epochs)
   - Unfreezing: ~8-10 hours (3 epochs)
3. **Total per experiment**: ~10-14 hours

## Monitoring

- WandB Project: `consistency-lens-gpt2` or `consistency-lens-gpt2-pile`
- Key metrics: `val_loss`, `kl_loss`, `lm_loss`
- Verbose samples logged every 1000 steps

## Scaling to Larger Models

After GPT-2 success:
1. GPT-2 Medium (355M): Change `model_name: "openai-community/gpt2-medium"`
2. GPT-2 Large (774M): Change `model_name: "openai-community/gpt2-large"`
3. Llama 2 7B: New config with `model_name: "meta-llama/Llama-2-7b-hf"`

## Troubleshooting

- **OOM**: Reduce `per_device_train_batch_size` or enable gradient checkpointing
- **Slow dumping**: Use pre-tokenization (see CLAUDE.md)
- **Diverging loss**: Reduce learning rate or increase warmup steps