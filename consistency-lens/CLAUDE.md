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

# 3. Evaluate trained model
python scripts/02_eval.py --checkpoint outputs/ckpt_step_X.pt
```

### Testing
```bash
# Run smoke tests
python tests/smoke_train.py
python tests/test_dataset.py
python tests/test_swap_hook.py
```

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

### Training Strategy
- **Dual-Path Autoencoding**: Trains on paired activations simultaneously
- **Functional Evaluation**: Uses KL divergence on actual LLM behavior, not just reconstruction loss
- **Scheduled Training**: Configurable Gumbel temperature and loss weight schedules

### Memory Optimizations
- **DeepSpeed ZeRO-2 + FSDP**: Distributed parameter sharding
- **8-bit Quantization**: Optional `bitsandbytes` for frozen models
- **Gradient Checkpointing**: Reduced memory during backpropagation
- **Flash Attention**: Optimized attention computation

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
- Testing via smoke tests that verify core functionality without full training runs