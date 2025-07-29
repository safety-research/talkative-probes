# Best-of-K Sweep Evaluation System

This document describes the K-sweep evaluation system for Talkative Autoencoders, including distributed execution and vector caching.

## Overview

The K-sweep evaluation (`03_best_of_k_sweep.py`) evaluates how variance recovery improves with different K values in best-of-K sampling. It supports:

- **Distributed evaluation** across multiple GPUs
- **Vector caching** to avoid re-extracting activations
- **SLURM submission** with automatic resource allocation
- **Flexible configuration** via Hydra

## Quick Start

### Single-GPU Evaluation
```bash
./scripts/submit_k_sweep.sh \
  +eval.checkpoint_path=/path/to/checkpoint.pt \
  +eval.k_values=[1,2,4,8,16,32]
```

### Multi-GPU Evaluation
```bash
./scripts/submit_k_sweep.sh num_gpus=4 \
  +eval.checkpoint_path=/path/to/checkpoint.pt \
  +eval.k_values=[1,2,4,8,16,32,64]
```

### With Vector Caching
```bash
./scripts/submit_k_sweep.sh \
  +eval.checkpoint_path=/path/to/checkpoint.pt \
  +eval.k_values=[1,2,4,8,16,32] \
  +eval.load_store=true
```

## Distributed Execution

The script supports distributed evaluation using PyTorch's DistributedDataParallel (DDP):

1. **Automatic Setup**: When `num_gpus>1`, the submission script automatically uses `torchrun`
2. **Data Sharding**: Each GPU processes a unique subset of the validation data
3. **Result Gathering**: Statistics are computed globally across all GPUs
4. **Rank-Specific Caching**: Each process saves its own cache file to avoid conflicts

### Resource Allocation

- **1-7 GPUs**: Allocated on a single node
- **8 GPUs**: Requests exclusive node access
- **Multi-node**: Not supported (single-node only for optimal performance)

## Vector Caching

Vector caching dramatically speeds up repeated experiments by saving extracted activations:

### How It Works

1. **Cache Key**: Based on model name, layer, dataset config, and num_positions
2. **Rank-Specific Files**: Each GPU saves `rank_N_vectors.pkl`
3. **Smart Loading**: Can use a subset of cached vectors (e.g., cache 50k, use 10k)
4. **Dataloader Optimization**: Skips extraction when cache exists and `num_positions=1`

### Caching Workflow

1. **Initial Cache Creation** (one-time, can be slow):
```bash
./scripts/submit_k_sweep.sh num_gpus=8 time=24:00:00 \
  +eval.checkpoint_path=/path/to/checkpoint.pt \
  +eval.k_values=[1] \
  +eval.load_store=true \
  +eval.max_batches=null  # Cache entire dataset
```

2. **Run Experiments Using Cache** (fast):
```bash
# Temperature sweep
for temp in 0.8 1.0 1.2; do
  ./scripts/submit_k_sweep.sh num_gpus=4 \
    +eval.checkpoint_path=/path/to/checkpoint.pt \
    +eval.k_values=[1,2,4,8,16,32,64] \
    +eval.temperature=$temp \
    +eval.load_store=true \
    +eval.output_file=k_sweep_temp_${temp}.json
done
```

### Cache Management

- **Location**: Default `vector_cache/`, configurable with `+eval.cache_dir=`
- **Structure**: `cache_dir/{hash}/rank_N_vectors.pkl`
- **Clearing**: `rm -rf vector_cache/` to force regeneration

## Configuration Options

### Submission Script Options

| Option | Description | Default |
|--------|-------------|---------|
| `num_gpus=N` | Number of GPUs to use | 1 |
| `time=HH:MM:SS` | SLURM time limit | 24:00:00 |
| `partition=NAME` | SLURM partition | gpu |
| `job_name=NAME` | Job name | k_sweep |
| `nice=true/N` | Low-priority requeueable job | false |
| `force_direct=true` | Skip SLURM, run directly | false |

### Evaluation Options (via Hydra)

| Option | Description | Example |
|--------|-------------|---------|
| `+eval.checkpoint_path` | Model checkpoint | `/path/to/checkpoint.pt` |
| `+eval.k_values` | List of K values | `[1,2,4,8,16,32]` |
| `+eval.max_batches` | Limit batches (null=all) | `100` |
| `+eval.temperature` | Generation temperature | `1.0` |
| `+eval.load_store` | Enable caching | `true` |
| `+eval.cache_dir` | Cache directory | `vector_cache` |
| `+eval.output_dir` | Results directory | `eval_results` |
| `+eval.output_file` | Results filename | `k_sweep_results.json` |
| `+eval.do_bootstrap` | Bootstrap confidence intervals | `true` |

## Output and Visualization

### Results Format

Results are saved as JSON in `eval_results/` with:
- Variance recovery for each K
- Confidence intervals
- MSE and R-squared metrics
- Configuration used

### Plotting Results

```bash
source scripts/ensure_env.sh && uv run python eval_results/plot_k_sweep.py
```

This creates:
- `gemma3_variance_recovery.png`: Main plot
- `gemma3_variance_recovery.pdf`: Publication-quality version

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `+eval.max_generation_batch_size` (default: 32)
   - Reduce `+eval.vector_extraction_batch_size` (default: 8)

2. **Slow First Run**
   - Expected when building cache
   - Use more GPUs for initial caching
   - Check `+eval.load_store=true` is set for subsequent runs

3. **Distributed Errors**
   - Ensure all GPUs are on same node
   - Check NCCL environment variables
   - Try different master port if conflicts

4. **Cache Not Used**
   - Verify cache exists: `ls -la vector_cache/`
   - Check model name and layer match
   - Ensure `+eval.load_store=true`

### Monitoring

- **SLURM Jobs**: `squeue -u $USER`
- **Live Logs**: `tail -f logs/k_sweep_*.out`
- **GPU Usage**: `nvidia-smi -l 1`
- **Cancel Job**: `scancel <job_id>`

## Advanced Usage

### Custom Configurations

Create a YAML config in `conf/eval/`:
```yaml
# conf/eval/my_k_sweep.yaml
checkpoint_path: /path/to/checkpoint.pt
k_values: [1, 2, 4, 8, 16, 32, 64, 128]
max_batches: 1000
temperature: 1.2
load_store: true
cache_dir: my_cache
do_bootstrap: true
```

Then run:
```bash
./scripts/submit_k_sweep.sh +eval=my_k_sweep
```

### Batch Experiments

```bash
# Run multiple checkpoints
for ckpt in outputs/*/checkpoint_*.pt; do
  ./scripts/submit_k_sweep.sh num_gpus=4 \
    +eval.checkpoint_path=$ckpt \
    +eval.k_values=[1,4,16,64] \
    +eval.output_file=$(basename $ckpt .pt)_k_sweep.json
done
```

## Implementation Details

### Distributed Algorithm

1. Each rank loads the dataset with `FastDistributedSampler`
2. Vectors are extracted/loaded independently per rank
3. Best-of-K generation happens on each rank's subset
4. Results are gathered via `dist.all_gather`
5. Global statistics computed on rank 0, broadcast to all

### Cache Implementation

- **Key Components**: model name, layer, dataset, num_positions
- **Not Included**: max_batches (allows subset reuse), checkpoint path
- **Format**: Pickle files with CPU tensors
- **Metadata**: Saved but currently unused (future validation)

### Performance Considerations

- **Vector Extraction**: ~5-10 min for 1000 samples on 8 GPUs
- **Best-of-K Generation**: Scales linearly with K
- **Memory Usage**: ~2GB per 10k vectors (model-dependent)
- **Optimal Batch Sizes**: Depends on GPU memory and model size