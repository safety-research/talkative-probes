# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Consistency Lens** is a scalable method for interpreting Large Language Models by forcing internal neural states through a human-readable textual bottleneck. The system trains an autoencoder that converts LLM activations to text explanations and back, testing whether reconstructed activations preserve functional behavior.

## Development Commands

### Setup
```bash
cd consistency-lens
uv venv
uv pip install -e .
```

### Main Workflow
```bash
# 1. Extract activations from corpus
python scripts/00_dump_activations.py --config config/lens_simple.yaml

# 2. Train the lens  
python scripts/01_train.py --config config/lens_simple.yaml
# Training creates run-specific output directories like: outputs/5M_L5_SimpleStories_lr1e-4_t10_1222_1430/

# 3. Evaluate trained model
python scripts/02_eval.py --checkpoint outputs/RUN_NAME/ckpt_step_X.pt --save_results
# Evaluation results saved to: outputs/evaluations/RUN_NAME_stepX_TIMESTAMP/
```

### Resuming Training
```bash
# Resume from checkpoint (automatically resumes WandB run if available)
python scripts/01_train.py --config config/lens_simple.yaml --resume outputs/ckpt_step_1000.pt

# Resume with explicit WandB run ID override
python scripts/01_train.py --config config/lens_simple.yaml --resume outputs/ckpt_step_1000.pt --wandb_resume_id abc123xyz
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
python tests/smoke_train.py
python tests/test_dataset.py
python tests/test_swap_hook.py
```

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
python scripts/01_train.py --config config/lens_simple.yaml

# Or use the provided wrapper script
./scripts/train_with_compile.sh --config config/lens_simple.yaml
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
- **Training Loop** (`lens/training/loop.py`): Composite loss with L_LM + L_KL + entropy regularization

### Data Pipeline
- **Activation Cache**: Pre-computed tuples `(A, A', input_ids, ...)` stored as sharded `.pt` files
- **Dataset** (`lens/data/dataset.py`): Handles both legacy single-file and modern sharded formats
- **Collation** (`lens/data/collate.py`): Efficient batching with padding and tensor management

### Configuration System
All training controlled via YAML configs in `config/`:
- `lens_simple.yaml`: Main training configuration
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

### Run Organization
- **Automatic Run Naming**: Training runs are named with format `{model}_{layer}_{dataset}_lr{lr}_t{t_text}_{timestamp}`
- **Run-Specific Outputs**: Each training run gets its own checkpoint directory under `outputs/`
- **Evaluation Results**: Evaluation outputs include JSON metrics and human-readable summaries
- **Dataset-Aware**: Scripts automatically extract model/layer/dataset info from activation paths