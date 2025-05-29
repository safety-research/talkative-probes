# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Consistency Lens** is a scalable method for interpreting Large Language Models by forcing internal neural states through a human-readable textual bottleneck. The system trains an autoencoder that converts LLM activations to text explanations and back, testing whether reconstructed activations preserve functional behavior.

### Training Objective

- **Primary Goal**: KL divergence loss measures functional preservation - ensuring reconstructed activations maintain the same behavioral properties as original activations (fixed weight)
- **Secondary Goal**: Language modeling loss encourages linguistic fluency - keeping explanations coherent and human-readable (ramped up via alpha schedule)  
- **Training Strategy**: KL loss provides the constant core learning signal, while LM loss weight is gradually increased from 0 to introduce linguistic constraints without overwhelming the reconstruction objective

## Development Commands

### Setup
```bash
cd consistency-lens
make  # One-time setup: installs uv and configures shared cache
```

#### Multi-Node Environment Setup (RunPod/SLURM)

This project uses `uv` for seamless multi-node Python environment management.

**How it works**:
- `uv run` automatically manages Python environments per-project
- Virtual environments stored in node-local locations via `UV_PROJECT_ENVIRONMENT`:
  - Compute nodes: `$SLURM_TMPDIR/uv-venv-consistency-lens` (fast local SSD)
  - Login nodes: `~/.cache/uv/envs/consistency-lens` (persistent)
- Downloaded packages cached in shared `/workspace/.uv_cache` (fast with hardlinks)
- No manual venv activation needed - just prefix commands with `uv run`

**Initial Setup** (once total, on any node):
```bash
make  # Installs uv to ~/.local/bin and sets up shared cache
source ~/.cargo/env  # Add uv to PATH (or restart shell)
```

**That's it!** No need to run setup on each node. When you first use `uv run` on a new node, it will automatically create the environment for that node.

**Usage Examples**:
```bash
# Training
uv run python scripts/01_train.py

# Evaluation  
uv run python scripts/02_eval.py checkpoint=outputs/ckpt.pt

# Running tests
uv run pytest tests/

# Using torchrun for distributed training
uv run torchrun --nproc_per_node=8 scripts/01_train_distributed.py
```

**Key Benefits**:
- **True node isolation**: Each node has its own venv (no symlink conflicts!)
- **Fast installs**: Shared package cache with hardlinking
- **No activation needed**: Just use `uv run` prefix
- **Optimized storage**: Compute nodes use fast local SSD, login nodes use persistent cache
- **Self-healing**: Environments recreated if corrupted

**Performance Note**: First `uv run` on a new node will be slower (~1-2 min) as it installs PyTorch and other packages. Subsequent runs are instant.

### Main Workflow
```bash
# 1. Extract activations from corpus
# Option A: Single GPU (slower)
uv run python scripts/00_dump_activations.py --config config/lens_simple.yaml

# Option B: Multi-GPU (recommended for production)
./scripts/launch_multigpu_dump.sh

# Option C: Pre-tokenize first for maximum speed
uv run python scripts/pretokenize_dataset.py  # One-time preprocessing
./scripts/launch_multigpu_dump.sh  # Uses pretokenized data automatically

# 2. Train the lens  
export TORCHINDUCTOR_CACHE_DIR="${HOME}/.cache/torchinductor"
uv run python scripts/01_train.py
# Training creates run-specific output directories like: outputs/5M_L5_SimpleStories_lr1e-4_t10_1222_1430/

# 3. Evaluate trained model
uv run python scripts/02_eval.py checkpoint=outputs/RUN_NAME/ckpt_step_X.pt evaluation.save_results=true
# Evaluation results saved to: outputs/evaluations/RUN_NAME_stepX_TIMESTAMP/
```

### Evaluation Examples
```bash
# Basic evaluation
uv run python scripts/02_eval.py checkpoint=outputs/checkpoints/SimpleStories-5M_L5_S_lr1e-3_t5_resume_0523_0009/checkpoint_step7000_epoch4.pt

# With custom parameters
uv run python scripts/02_eval.py \
    checkpoint=outputs/checkpoints/SimpleStories-5M_L5_S_lr1e-3_t5_resume_0523_0009/checkpoint_step7000_epoch4.pt \
    evaluation.verbose_samples=5 \
    evaluation.batch_size=8 \
    evaluation.num_batches=50 \
    evaluation.save_results=true

# Save results to specific directory
uv run python scripts/02_eval.py \
    checkpoint=outputs/checkpoints/SimpleStories-5M_L5_S_lr1e-3_t5_resume_0523_0009/checkpoint_step7000_epoch4.pt \
    evaluation.output_dir=outputs/my_evaluation \
    evaluation.save_results=true

# Override activation directory
uv run python scripts/02_eval.py \
    checkpoint=outputs/checkpoints/SimpleStories-5M_L5_S_lr1e-3_t5_resume_0523_0009/checkpoint_step7000_epoch4.pt \
    evaluation.activation_dir=./data/activations/different_dataset_test
```

### Resuming Training
```bash
# Resume from checkpoint (automatically resumes WandB run if available)
uv run python scripts/01_train.py \
    resume=outputs/ckpt_step_1000.pt

# Resume with explicit WandB run ID override
uv run python scripts/01_train.py \
    resume=outputs/ckpt_step_1000.pt \
    wandb_resume_id=abc123xyz
```

### Example Learning Rate Schedules
```yaml
# Linear decay with warmup
lr_scheduler:
  type: "linear"
  warmup_steps: 1000
  warmup_start_factor: 0.1
  end_factor: 0.0

# Cosine annealing
lr_scheduler:
  type: "cosine"
  eta_min: 1.0e-5

# Cosine with warm restarts
lr_scheduler:
  type: "cosine_with_restarts"
  T_0: 500
  T_mult: 2
  eta_min: 1.0e-5
```

### Testing
```bash
# Run smoke tests
uv run python tests/smoke_train.py
uv run python tests/test_dataset.py
uv run python tests/test_swap_hook.py
```

### SLURM Cluster Usage

For HPC clusters with SLURM, use the provided submission scripts that handle dependencies automatically:

#### Quick Start (Recommended)
The `submit_with_config.sh` script now works on both SLURM and non-SLURM environments:

```bash
# Submit new experiments with automatic activation dumping
# Uses Hydra-style key=value syntax for all arguments
./scripts/submit_with_config.sh config=conf/simplestories_frozen.yaml      # SimpleStories frozen
./scripts/submit_with_config.sh config=conf/simplestories_unfreeze.yaml    # SimpleStories progressive unfreeze
./scripts/submit_with_config.sh config=conf/gpt2_frozen.yaml               # GPT-2 with OpenWebText
./scripts/submit_with_config.sh config=conf/gpt2_unfreeze.yaml             # GPT-2 with OpenWebText + unfreeze
./scripts/submit_with_config.sh config=conf/gpt2_pile_frozen.yaml          # GPT-2 with The Pile
./scripts/submit_with_config.sh config=conf/gpt2_pile_unfreeze.yaml        # GPT-2 with The Pile + unfreeze

# Resume from checkpoint (find checkpoint in outputs/ directory)
./scripts/submit_with_config.sh config=conf/simplestories_frozen.yaml \
    resume_checkpoint=outputs/checkpoints/run_name/checkpoint_step5000.pt

# Resume with specific WandB run ID (get ID from WandB dashboard)
./scripts/submit_with_config.sh config=conf/simplestories_frozen.yaml \
    resume_checkpoint=outputs/checkpoints/run_name/checkpoint_step5000.pt \
    wandb_resume_id=abc123xyz

# Use specific SLURM nodes (optional - defaults to current node, SLURM only)
./scripts/submit_with_config.sh config=conf/gpt2_frozen.yaml nodelist=node001,node002

# Specify number of GPUs for non-SLURM environments (auto-detected if not specified)
./scripts/submit_with_config.sh config=conf/gpt2_frozen.yaml num_gpus=4

# Force re-dump activations even if they exist
./scripts/submit_with_config.sh config=conf/simplestories_frozen.yaml force_redump=true

# Pass Hydra overrides to training script
./scripts/submit_with_config.sh config=conf/gpt2_frozen.yaml \
    learning_rate=1e-3 \
    batch_size=16 \
    gumbel_temperature.schedule.start_value=2.0

# Force direct execution even on SLURM systems
FORCE_DIRECT=true ./scripts/submit_with_config.sh config=conf/simplestories_frozen.yaml
```

The wrapper script automatically:
- Detects SLURM vs non-SLURM environments
- Checks if activations exist
- **Pre-tokenizes datasets for 5x faster dumping** (all experiments)
- On SLURM: Submits jobs with proper dependencies
- On non-SLURM: Runs dumping and training sequentially in foreground
- Auto-detects available GPUs (configurable)
- Ensures efficient resource utilization

#### Available Experiments
- **SimpleStories**: 5M parameter model, faster experiments (~12-18 hours)
  - `ss-frozen`: Base model frozen throughout
  - `ss-unfreeze`: Base model unfrozen after 1st epoch
- **GPT-2**: 124M parameters, longer experiments (~24-36 hours)
  - `gpt2-frozen`: Frozen model with OpenWebText
  - `gpt2-pile`: Frozen model with The Pile dataset

#### Manual Submission (Not Recommended)
```bash
# Step 1: Dump activations (8 GPUs)
sbatch scripts/slurm_dump_activations.sh conf/config.yaml 5

# Step 2: Submit training (1 GPU) - will fail if activations don't exist
sbatch scripts/slurm_simplestories_frozen.sh
```

#### Monitoring Jobs
```bash
# Check job status
squeue -u $USER

# Monitor specific job output
tail -f logs/simplestories_frozen_*.out

# Check for errors
grep -i error logs/*.err
```

### Multi-GPU Training

The project now supports distributed data-parallel training across multiple GPUs for faster training:

#### Quick Start with Multi-GPU
```bash
# Train with 8 GPUs (auto-enables distributed mode)
./scripts/submit_with_config.sh config=conf/gpt2_frozen.yaml num_gpus_train=8

# Train with 4 GPUs for training, 8 for dumping
./scripts/submit_with_config.sh config=conf/gpt2_frozen.yaml num_gpus=8 num_gpus_train=4

# Force distributed mode even with 1 GPU (for testing)
./scripts/submit_with_config.sh config=conf/simplestories_frozen.yaml use_distributed=true

# Multi-GPU with other options
./scripts/submit_with_config.sh config=conf/gpt2_frozen.yaml \
    num_gpus_train=8 \
    learning_rate=5e-4 \
    batch_size=4  # Per-GPU batch size
```

#### Distributed Training Details
- **Data Parallelism**: Each GPU processes different data batches
- **Automatic Scaling**: Effective batch size = `batch_size * num_gpus * gradient_accumulation_steps`
- **Efficient Communication**: Uses NCCL backend for GPU-to-GPU communication
- **Rank-Aware**: Only rank 0 saves checkpoints and logs to WandB
- **DistributedSampler**: Ensures each GPU sees different data

#### Launching Distributed Training Manually
```bash
# Using torchrun (PyTorch's distributed launcher)
torchrun --nproc_per_node=8 scripts/01_train_distributed.py \
    --config-path=../conf --config-name=gpt2_frozen

# Using the provided launcher script
./scripts/launch_distributed_train.sh \
    --config-path=../conf \
    --config-name=gpt2_frozen \
    --num-gpus=8

# In SLURM environment (auto-detected)
sbatch --gres=gpu:8 scripts/launch_distributed_train.sh \
    --config-path=../conf --config-name=gpt2_frozen
```

#### Configuration for Distributed Training
Add to your config YAML to customize distributed behavior:
```yaml
# Include distributed defaults
defaults:
  - distributed

# Override specific settings
distributed:
  backend: nccl  # or gloo for CPU
  find_unused_parameters: false
  gradient_sync_interval: ${gradient_accumulation_steps}
  mixed_precision:
    enabled: true
    dtype: bfloat16  # Better for A100/H100
```

#### Performance Considerations
- **Linear Scaling**: With proper batch sizes, expect near-linear speedup
- **Batch Size**: Keep per-GPU batch size reasonable (4-8 for large models)
- **Memory**: Each GPU needs full model copy (no model parallelism yet)
- **Network**: Fast interconnect (NVLink/InfiniBand) improves scaling

### Performance Configuration

#### torch.compile Setup
Model compilation is enabled by default for better performance:

```yaml
# In config/lens_simple.yaml
compile_models: true  # Enable torch.compile for Decoder & Encoder
```

If you encounter cache permission issues:
```bash
# Set cache directory before training
export TORCHINDUCTOR_CACHE_DIR="${HOME}/.cache/torchinductor"
python scripts/01_train.py

# Or use the provided wrapper script
./scripts/train_with_compile.sh
```

#### TensorFloat32 (TF32)
Automatically enabled for NVIDIA Ampere GPUs (A100/H100):
- Provides up to 10x speedup for matrix operations
- Maintains training stability with 19-bit precision
- No configuration needed - enabled by default in training script

## Architecture

### Core Components
- **Original Model** (`lens/models/orig.py`): Frozen HuggingFace LLM with activation replacement hooks
- **Decoder** (`lens/models/decoder.py`): Converts activations → text via Gumbel-Softmax + STE  
- **Encoder** (`lens/models/encoder.py`): Converts text embeddings → reconstructed activations
  - Configurable output layer selection via `output_layer` parameter (-1 = last layer, 0 = embeddings, 1..n = specific layer)
- **Training Loop** (`lens/training/loop.py`): Composite loss with L_LM + L_KL + entropy regularization

### Data Pipeline
- **Activation Cache**: Pre-computed tuples `(A, A', input_ids, ...)` stored as sharded `.pt` files
- **Dataset** (`lens/data/dataset.py`): Handles both legacy single-file and modern sharded formats
- **Collation** (`lens/data/collate.py`): Efficient batching with padding and tensor management

### Configuration System
All training controlled via YAML configs in `conf/`:
- `config.yaml`: Main training and evaluation configuration
- `ds_stage2.yaml`: DeepSpeed distributed settings  
- `wandb.yaml`: Experiment tracking

## Key Implementation Details

### WandB Verbose Sample Logging
During training, verbose samples are automatically logged to WandB tables for easy inspection:
- Shows input text, chosen token, decoder explanation, and top predictions
- Includes autoregressive continuations when enabled
- Compares original vs reconstructed model predictions
- Updated incrementally during training with workaround for WandB table limitations

The table includes columns for:
- Input text with context
- Token being analyzed and its position
- Decoder's textual explanation
- Top predicted tokens with probabilities
- Model continuation from the analyzed position
- Original vs reconstructed model predictions

### Training Strategy
- **Dual-Path Autoencoding**: Trains on paired activations simultaneously
- **Functional Evaluation**: Uses KL divergence on actual LLM behavior, not just reconstruction loss
- **Scheduled Training**: Configurable Gumbel temperature and loss weight schedules
- **Learning Rate Scheduling**: Supports constant, linear, cosine, polynomial, and exponential schedules with optional warmup

### Memory Optimizations
- **DeepSpeed ZeRO-2 + FSDP**: Distributed parameter sharding
- **8-bit Quantization**: Optional `bitsandbytes` for frozen models
- **Gradient Checkpointing**: Reduced memory during backpropagation
- **Flash Attention**: Optimized attention computation

### Performance Optimizations
- **torch.compile**: Enable model compilation for faster training (up to 2x speedup)
- **TensorFloat32 (TF32)**: Automatic mixed precision for Ampere GPUs (A100/H100)
- **Distributed Training**: Multi-GPU support with optimized data loading

### Scalability Design
- Built for 8xH100 GPU environments with billion-parameter models
- Heavy reliance on proven libraries (HuggingFace, DeepSpeed, Flash Attention)
- `torch.compile` support for additional performance gains

## Evaluation Configuration

The evaluation script `02_eval.py` uses Hydra configuration with the following available parameters:

### Required Parameters
- `checkpoint`: Path to checkpoint file (required)

### Optional Evaluation Parameters
- `evaluation.activation_dir`: Override activation directory for evaluation
- `evaluation.batch_size`: Batch size for evaluation (default: 4)
- `evaluation.num_batches`: Number of batches to evaluate (default: 25)
- `evaluation.verbose_samples`: Number of verbose samples to print (default: 3)
- `evaluation.top_n_analysis`: Number of top predictions to show (default: 3)
- `evaluation.val_fraction`: Validation split fraction (default: uses main config)
- `evaluation.split_seed`: Random seed for splits (default: uses main config)
- `evaluation.output_dir`: Output directory for results (default: auto-generated)
- `evaluation.save_results`: Save JSON results file (default: false)

### Usage Notes
- Use Hydra syntax: `key=value` instead of `--flag value`
- Use dot notation for nested parameters: `evaluation.verbose_samples=5`
- All evaluation parameters are optional except `checkpoint`
- Results automatically saved to timestamped directories when `evaluation.save_results=true`

## Important Patterns

### Code Organization
- Models are clones of original LLM rather than separate architectures
- Extensive use of dataclasses for configuration and data structures
- Type hints throughout for better development experience
- Modular design with clear separation between data/models/training/evaluation

### Development Practices
- Configuration-driven development - avoid hardcoded parameters
- Comprehensive logging with structured console output and W&B integration
- Checkpointing every 100 steps with automatic cleanup of old checkpoints
- WandB run resumption - checkpoints store run IDs for seamless experiment continuation
- Testing via smoke tests that verify core functionality without full training runs

### Training Experience
- **Progress Bar**: Real-time training progress with ETA, loss, learning rate, and throughput
- **Performance Metrics**: Tracks samples/second, tokens/second, and step times
- **System Monitoring**: GPU utilization, memory usage, and temperature logged to W&B
- **Clean Logging**: Reduced console clutter with informative periodic updates
- **Automatic Metrics**: Performance stats calculated and logged without manual intervention
- **Verbose Sample Tables**: Training samples automatically logged to W&B tables for inspection
- **SLURM Integration**: SLURM job ID and node info automatically logged to W&B for easy tracking

### Run Organization
- **Automatic Run Naming**: Training runs are named with format `{dataset}_{model}_{layer}_{learning_rate}_{timestamp}`
- **Structured Outputs**: All outputs saved to timestamped directories under `outputs/`
- **Evaluation Tracking**: Evaluation results automatically organized by checkpoint and timestamp

## Future Architecture Considerations

### Runtime Activation Computation

The current system pre-dumps activations to disk before training. An alternative approach would be to compute activations at runtime from pre-tokenized datasets.

**Current approach (pre-dumping):**
- Pros: One-time LLM forward pass per sequence, consistent activations, storage of intermediate results
- Cons: Large storage requirements (multi-GB), two-stage workflow complexity

**Alternative approach (runtime computation):**
- Pros: No storage overhead, single-stage workflow, more flexible layer/position changes
- Cons: Need LLM + decoder/encoder in memory simultaneously

**Key insight**: Training already requires 2-3 LLM forward passes per step for KL loss computation, so the compute overhead argument for pre-dumping is weaker than initially assumed.

This alternative could be valuable for:
- Single-shot training experiments
- Storage-constrained environments  
- Research scenarios prioritizing workflow simplicity
- Cases where memory is more abundant than storage