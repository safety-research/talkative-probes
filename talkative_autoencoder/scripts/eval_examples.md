# Evaluation Examples

## Running Evaluation

The evaluation scripts support three types of evaluation:

### 1. Standard Validation (`02_eval.py`)
Uses the same validation logic as training. Provides comprehensive metrics including MSE, KL divergence, language modeling loss, variance explained, correlation, and intervention analysis.

### 2. Best-of-N Sampling (`02_eval.py`)
Uses LensAnalyzer to find the best explanation among N candidates. Only provides MSE metrics but uses a more thorough search for optimal explanations.

### 3. Best-of-K Sweep Analysis (`03_best_of_k_sweep.py`)
Evaluates how variance recovery improves with different K values in best-of-K sampling. Provides MSE, variance recovery, and R² metrics for each K value.

### Basic Usage

```bash
# Standard validation (reuses training validation logic)
./scripts/submit_eval.sh +eval=default eval.checkpoint_path=/path/to/checkpoint.pt

# Quick validation on subset
./scripts/submit_eval.sh +eval=quick eval.checkpoint_path=/path/to/checkpoint.pt

# Best-of-N evaluation
./scripts/submit_eval.sh +eval=best_of_n eval.checkpoint_path=/path/to/checkpoint.pt

# Custom settings
./scripts/submit_eval.sh eval.checkpoint_path=/path/to/checkpoint.pt \
    eval.use_best_of_n=true \
    eval.best_of_n=20 \
    eval.max_batches=50 \
    eval.output_file=custom_results.json
```

### SLURM Options

```bash
# Run on multiple GPUs
GPUS_PER_NODE=4 ./scripts/submit_eval.sh +eval=default eval.checkpoint_path=/path/to/checkpoint.pt

# Longer time limit
TIME=4:00:00 ./scripts/submit_eval.sh +eval=best_of_n eval.checkpoint_path=/path/to/checkpoint.pt

# Different partition
PARTITION=gpu-large ./scripts/submit_eval.sh +eval=default eval.checkpoint_path=/path/to/checkpoint.pt
```

### Direct Python Execution

```bash
# Single GPU
python scripts/02_eval.py +eval=default eval.checkpoint_path=/path/to/checkpoint.pt

# Multi-GPU with torchrun
torchrun --nproc_per_node=4 scripts/02_eval.py +eval=default eval.checkpoint_path=/path/to/checkpoint.pt
```

## Configuration Options

Key parameters in `config/eval/*.yaml`:

- `checkpoint_path`: Path to model checkpoint
- `use_best_of_n`: Enable best-of-N sampling mode
- `best_of_n`: Number of samples per position (default: 5)
- `max_batches`: Limit number of validation batches
- `batch_size`: Batch size for evaluation
- `analyzer_batch_size`: Internal batch size for LensAnalyzer
- `n_groups_per_rollout`: Positions processed together in best-of-N
- `save_detailed_results`: Save per-example results
- `calculate_salience`: Calculate token importance scores
- `output_dir`: Directory for results
- `output_file`: Results filename

## Best-of-K Sweep Analysis

### Running K-Sweep

The K-sweep analysis evaluates how model performance improves with different K values:

```bash
# Full K-sweep (K = 1, 2, 4, 8, 16, 32, 64)
./scripts/run_k_sweep.sh eval.checkpoint_path=/path/to/checkpoint.pt

# Quick test with fewer K values (1, 4, 16)
CONFIG=+eval=k_sweep_quick ./scripts/run_k_sweep.sh eval.checkpoint_path=/path/to/checkpoint.pt

# Custom K values
python scripts/03_best_of_k_sweep.py eval.checkpoint_path=/path/to/checkpoint.pt \
    eval.k_values=[1,5,10,20,50] \
    eval.output_file=custom_k_sweep.json
```

### K-Sweep Output

The K-sweep produces a JSON file with:
- Results for each K value including:
  - Average MSE
  - Variance recovery (fraction of variance explained)
  - R-squared
  - Total positions evaluated
- Summary arrays for easy plotting

### Plotting K-Sweep Results

```python
import json
import matplotlib.pyplot as plt

# Load results
with open('eval_results/k_sweep_results.json', 'r') as f:
    data = json.load(f)

summary = data['summary']
k_values = summary['k_values']
var_recovery = summary['variance_recovery_values']
mse_values = summary['mse_values']

# Plot variance recovery vs K
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(k_values, var_recovery, 'o-', linewidth=2, markersize=8)
ax1.set_xlabel('K (number of samples)')
ax1.set_ylabel('Variance Recovery')
ax1.set_title('Variance Recovery vs Best-of-K')
ax1.set_xscale('log')
ax1.grid(True, alpha=0.3)

# Plot MSE vs K
ax2.plot(k_values, mse_values, 'o-', linewidth=2, markersize=8, color='red')
ax2.set_xlabel('K (number of samples)')
ax2.set_ylabel('Average MSE')
ax2.set_title('MSE vs Best-of-K')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('k_sweep_analysis.png', dpi=150)
plt.show()
```

## Output Formats

### Standard Validation Output
Results include only average MSE (full metrics are logged to console/wandb):
```json
{
  "avg_mse": 0.123456,
  "mode": "standard_validation",
  "checkpoint_path": "/path/to/checkpoint.pt",
  "eval_config": {...}
}
```

### Best-of-N Output
```json
{
  "avg_mse": 0.098765,
  "total_samples": 50000,
  "mode": "best_of_n",
  "checkpoint_path": "/path/to/checkpoint.pt",
  "eval_config": {...}
}
```

### K-Sweep Output
```json
{
  "checkpoint_path": "/path/to/checkpoint.pt",
  "results_by_k": [
    {
      "k": 1,
      "avg_mse": 0.123456,
      "variance_recovery": 0.4567,
      "r_squared": 0.4321,
      "total_positions": 100000
    },
    ...
  ],
  "summary": {
    "k_values": [1, 2, 4, 8, 16, 32, 64],
    "variance_recovery_values": [...],
    "mse_values": [...]
  }
}
```