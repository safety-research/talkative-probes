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

# Test differentiable generation equivalence
uv run python tests/test_differentiable_generation.py
```

### Flash Attention Support (Optional)

Flash Attention 2 provides optimized attention computation for faster training. When combined with KV caching, it offers O(n) complexity with highly optimized kernels.

#### Installation
```bash
# Install Flash Attention (requires CUDA, takes 5-10 minutes to compile)
make flash-attention
```

This command will:
1. Check CUDA availability
2. Add flash-attn to project dependencies if needed
3. Build and install Flash Attention with proper flags
4. Verify the installation

#### Usage
Enable Flash Attention in your config:
```yaml
decoder:
  use_flash_attention: true  # Automatically uses Flash + KV cache
```

Or use the pre-configured example:
```bash
uv run python scripts/01_train.py --config conf/gpt2_frozen_flash.yaml
```

#### Troubleshooting
If installation fails:
- Ensure CUDA toolkit is installed (`nvcc --version`)
- Check available disk space (compilation needs ~5GB)
- Verify PyTorch CUDA version matches system CUDA
- The code automatically falls back to standard KV cache if Flash Attention is unavailable

#### Performance Benefits
- 2-5x speedup for attention computation
- Lower memory usage during generation
- Maintains full differentiability for training

#### Environment Setup for Scripts
When running scripts outside of the standard workflow, source the environment helper:
```bash
source scripts/ensure_env.sh
# This sets up the uv_run function and environment variables
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

# Resume with smooth LR transition (useful when changing learning rate)
./scripts/submit_with_config.sh config=conf/simplestories_frozen.yaml \
    resume_checkpoint=outputs/checkpoints/run_name/checkpoint_step5000.pt \
    learning_rate=5e-4 \
    smooth_lr_transition.enabled=true \
    smooth_lr_transition.transition_steps=1000s

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

(Rest of the file remains unchanged)

## Flash Attention Integration

The project now includes Flash Attention v2 support for memory-efficient training:

- **Implementation**: `lens/models/flash_kv_cache_v2.py` provides a clean, direct implementation
- **Installation**: `make flash-attention` (requires CUDA-capable GPU)
- **Usage**: Set `use_flash_attention: true` in decoder config
- **Benefits**: Up to 86% memory reduction during training, enabling larger batch sizes
- **Testing**: Comprehensive tests in `tests/test_flash_v2.py` verify functional equivalence

Note: Small numerical differences (< 0.2) are expected and normal. Use tau > 0 for consistent token generation.

## Memories

- Always source `scripts/ensure_env.sh` and then use `uv run` to execute scripts in this repository.
- Flash Attention is fully integrated and tested - use it for memory-constrained training scenarios.