# SLURM Experiment Scripts

## Quick Start (Recommended)

```bash
# Submit SimpleStories experiments (handles everything automatically)
./scripts/submit_with_dumping.sh ss-frozen
./scripts/submit_with_dumping.sh ss-unfreeze

# Submit GPT-2 experiments
./scripts/submit_with_dumping.sh gpt2-frozen
./scripts/submit_with_dumping.sh gpt2-pile
```

The wrapper script automatically:
- Checks if activations exist
- Submits 8-GPU dumping job if needed
- Submits 1-GPU training job with dependency
- Ensures GPU doesn't sit idle waiting

## Architecture: Two-Step Process

Due to different GPU requirements:

1. **Activation Dumping** (8 GPUs) - Fast parallel extraction
2. **Training** (1 GPU) - Sequential training

## Step 1: Dump Activations (If Needed)

### For SimpleStories
```bash
# Check if you already have activations
ls data/activations/SimpleStories_train/SimpleStories/SimpleStories-5M/layer_5/train/

# If not, dump them (8 GPUs, ~30 minutes)
sbatch scripts/slurm_dump_activations.sh conf/config.yaml 5
```

### For GPT-2
```bash
# Dump OpenWebText activations (8 GPUs, ~2-4 hours)
sbatch scripts/slurm_dump_activations.sh conf/gpt2_frozen.yaml 6

# Dump The Pile activations (8 GPUs, ~4-6 hours)
sbatch scripts/slurm_dump_activations.sh conf/gpt2_pile_frozen.yaml 6
```

## Recommended: Use the Smart Submission Wrapper

The wrapper script automatically handles dependencies:

```bash
# Automatically checks for activations and submits both jobs if needed
./scripts/submit_with_dumping.sh ss-frozen
./scripts/submit_with_dumping.sh ss-unfreeze
./scripts/submit_with_dumping.sh gpt2-frozen
./scripts/submit_with_dumping.sh gpt2-pile

# Force re-dump even if activations exist
./scripts/submit_with_dumping.sh ss-frozen force-redump
```

**Benefits:**
- ✅ GPU won't sit idle - training waits for dumping to complete
- ✅ Automatically checks if activations exist
- ✅ Submits dumping job only if needed
- ✅ Sets up proper SLURM dependencies
- ✅ Single command does everything

## Manual Method (Not Recommended)

### Step 2: Run Training Experiments

Training scripts will check for activations and exit with instructions if not found.

## SimpleStories Experiments
```bash
./scripts/submit_simplestories_experiments.sh all
```

### Individual Experiments
```bash
# Frozen base model (12 hours)
sbatch scripts/slurm_simplestories_frozen.sh

# Progressive unfreezing (18 hours)
sbatch scripts/slurm_simplestories_unfreeze.sh

# Force re-dump activations
FORCE_REDUMP=true sbatch scripts/slurm_simplestories_frozen.sh
```

### Key Configuration
- **Model**: SimpleStories-5M
- **Layer**: 5
- **t_text**: 10 (width-10 explanations)
- **Frozen**: Base model always frozen
- **Unfreeze**: Frozen for 1st epoch, then unfrozen with 0.1x LR
- **Dataset**: Already dumped for layer 5

## GPT-2 Experiments

### Submit All GPT-2 Experiments
```bash
./scripts/submit_gpt2_experiments.sh all
```

### Individual Experiments
```bash
# Frozen with OpenWebText (24 hours)
sbatch scripts/slurm_gpt2_frozen.sh

# Progressive unfreezing (30 hours)
sbatch scripts/slurm_gpt2_unfreeze.sh

# Frozen with The Pile (36 hours)
sbatch scripts/slurm_gpt2_pile.sh
```

### Key Configuration
- **Model**: openai-community/gpt2 (124M)
- **Layer**: 6 (middle layer)
- **t_text**: 10
- **Batch size**: 256 (reduced for single GPU)
- **Datasets**: OpenWebText (5M) or The Pile (10M)

## Monitoring Jobs

```bash
# Check all jobs
squeue -u $USER

# Monitor specific experiment
tail -f logs/simplestories_frozen_*.out

# Check for errors
grep -i error logs/*.err

# Monitor all GPT-2 jobs
./scripts/monitor_gpt2_jobs.sh
```

## Pretokenization Details

All scripts now:
1. Check if pretokenized data exists
2. If not, pretokenize the dataset using 16 CPU cores
3. Use pretokenized data for 5x faster activation dumping

Pretokenized data locations:
- SimpleStories: `./data/corpus/SimpleStories/pretokenized/`
- OpenWebText: `./data/corpus/openwebtext/pretokenized/`
- The Pile: `./data/corpus/pile/pretokenized/`

## Force Re-dump

To force re-dumping of activations (useful if model changed):
```bash
# SimpleStories
FORCE_REDUMP=true sbatch scripts/slurm_simplestories_frozen.sh

# GPT-2
FORCE_REDUMP=true sbatch scripts/slurm_gpt2_frozen.sh
```

## Expected Timeline

### SimpleStories (per experiment)
1. Pretokenization: ~10 minutes (if needed)
2. Activation dumping: ~30 minutes (8 GPUs)
3. Training: ~10-16 hours (1 GPU)

### GPT-2 (per experiment)
1. Pretokenization: ~30-60 minutes (if needed)
2. Activation dumping: ~2-4 hours (8 GPUs)
3. Training: ~20-30 hours (1 GPU)

## Resource Usage

- **Activation Dumping**: 8 H100 GPUs (fast)
- **Training**: 1 H100 GPU (slower, but works)
- **CPU**: 16 cores for pretokenization
- **Memory**: ~200GB for The Pile pretokenization

## Notes

1. **Single GPU Training**: The training script doesn't support distributed training yet, so we use 1 GPU with reduced batch size (256 instead of 1024).

2. **Tau Schedule**: For SimpleStories unfreezing, tau is kept at 1.0 (no decay implemented yet due to epoch-based scheduling complexity).

3. **Learning Rates**: 
   - Projections: 1e-3 (SimpleStories), 1e-4 (GPT-2)
   - Unfrozen base model: 0.1x projection LR
   - Unfrozen output head: 0.5x projection LR

4. **WandB Projects**:
   - SimpleStories: `consistency-lens-simplestories`
   - GPT-2: `consistency-lens-gpt2` or `consistency-lens-gpt2-pile`