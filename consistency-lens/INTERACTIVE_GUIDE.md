# Interactive Lens Experimentation Guide

This guide helps you experiment with the encoder/decoder architecture interactively.

## Quick Start

### 1. Get GPU Allocation

```bash
# Option A: Quick 2-hour session
./scripts/get_interactive_gpu.sh

# Option B: Longer session with more memory
./scripts/get_interactive_gpu.sh --time 4:00:00 --mem 64G

# Option C: Manual SLURM command
salloc -p gpu --gres=gpu:1 --time=2:00:00 --mem=32G
```

### 2. Run Interactive Script

```bash
# With default GPT-2 config
uv run python scripts/interactive_lens.py

# With specific config
uv run python scripts/interactive_lens.py --config conf/simplestories_frozen.yaml

# With checkpoint
uv run python scripts/interactive_lens.py --config conf/gpt2_frozen.yaml \
    --checkpoint outputs/checkpoints/your_checkpoint.pt
```

### 3. VSCode Interactive Usage

1. Open `scripts/interactive_lens.py` in VSCode
2. Make sure Python extension is installed
3. Use one of these methods:
   - **Interactive Window**: Select code and press `Shift+Enter`
   - **Jupyter-style**: The script has `# %%` markers for cell execution
   - **Python Terminal**: Right-click and "Run Python File in Terminal"

## Key Functions Available

### Model Operations
- `extract_activations()` - Get activations from teacher model at any layer/position
- `generate_text_from_activation()` - Full pipeline: activation → text → reconstructed activation
- Generation methods available:
  - `generate_soft` - Standard differentiable generation with Gumbel-Softmax
  - `generate_soft_kv_cached` - O(n) generation with KV caching
  - `generate_soft_kv_flash` - Flash Attention with KV caching
  - `generate_soft_chkpt` - Gradient checkpointing for memory efficiency

### Activation Manipulation
- `patch_activations()` - Interpolate between two activations
- `analyze_activations()` - Compare two activation tensors (cosine sim, L2, etc.)
- `activation_steering()` - Steer one activation towards another

### Example Experiments

```python
# 1. Basic activation extraction and generation
text = "The weather today is"
activation = extract_activations(teacher, text, tokenizer)
generated_text, gen, reconstructed = generate_text_from_activation(
    decoder, encoder, activation, tokenizer, max_length=20, tau=1.0
)

# 2. Activation interpolation
act1 = extract_activations(teacher, "Happy story:", tokenizer)
act2 = extract_activations(teacher, "Sad story:", tokenizer)
blended = patch_activations(act1, act2, alpha=0.5)
result, _, _ = generate_text_from_activation(decoder, encoder, blended, tokenizer)

# 3. Try different generation methods
for method in ["generate_soft", "generate_soft_kv_cached", "generate_soft_kv_flash"]:
    gen_text, _, _ = generate_text_from_activation(
        decoder, encoder, activation, tokenizer, 
        decode_method=method, max_length=20, tau=1.0
    )
    print(f"{method}: {gen_text}")

# 4. Activation steering
style_act = extract_activations(teacher, "In the style of Shakespeare:", tokenizer)
content_act = extract_activations(teacher, "The meaning of life is", tokenizer)
steered = activation_steering(content_act, style_act, strength=0.3)
result, _, _ = generate_text_from_activation(decoder, encoder, steered, tokenizer)
```

## Configuration Files

Available configs in `conf/`:
- `gpt2_frozen.yaml` - GPT-2 with frozen transformer
- `gpt2_unfreeze.yaml` - GPT-2 with progressive unfreezing
- `simplestories_frozen.yaml` - SimpleStories dataset
- `gpt2_pile_frozen.yaml` - GPT-2 with The Pile dataset

## Tips

1. **Memory Management**: Clear cache periodically with `torch.cuda.empty_cache()`
2. **Batch Processing**: Most functions support batched inputs
3. **Temperature Control**: Lower temperature = more deterministic
4. **Gumbel Tau**: Lower tau = more discrete, higher tau = more smooth
5. **Checkpoint Loading**: Checkpoints contain decoder weights and training state

## Common Issues

- **CUDA OOM**: Reduce sequence length or use smaller batches
- **Import Errors**: Make sure to run with `uv run` prefix
- **SLURM Timeout**: Request longer sessions with `--time`
- **No GPU**: Check allocation with `nvidia-smi`

## Advanced Usage

For more complex experiments, you can:
1. Modify the decoder architecture in the config
2. Implement custom generation strategies
3. Add new activation manipulation functions
4. Export results for further analysis

Remember to save any important results or modified code!