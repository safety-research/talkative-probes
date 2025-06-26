# Cache Cycling Fix Summary

## Changes Made

### 1. Fixed Silent Data Repetition Issue

**File**: `lens/data/on_the_fly_datasets.py`

Added comprehensive logging to detect and warn when the dataset cycles:

- **Pre-regeneration warning**: When cache size exceeds shard size, warns about expected cycling with cycle count estimate
- **During regeneration**: Logs when dataset exhausts and starts a new cycle  
- **Post-regeneration summary**: Reports total cycles completed and partial cycle info

**Key additions**:
- Tracks `cycle_count` and `samples_in_current_cycle`
- Warns on first cycle, then every 5 cycles to avoid log spam
- Clear final summary with exact cycle information

### 2. Added WandB Metrics Tracking

**File**: `scripts/01_train_distributed.py`

Added metrics logging for cache regeneration statistics:
- `cache/cycles_per_regeneration`: How many times data cycles per regeneration
- `cache/total_regenerations`: Total number of regenerations so far
- `cache/estimated_total_cycles`: Estimated total dataset cycles
- `cache/shard_size_per_rank`: Size of each rank's data shard

### 3. Created Validation Tests

**Files**: 
- `tests/test_cache_regeneration.py` - Full validation with model loading (requires GPU)
- `tests/test_cache_regeneration_simple.py` - Lightweight validation with mocks

Tests verify:
- Cycling detection works correctly
- Warning messages appear when expected
- Exact cycle counting is accurate
- Edge cases handled properly

## Example Output

When cache size (250) exceeds shard size (100):

```
[WARNING] [Rank 0] WARNING: Cache size (250) exceeds shard size (100). Will cycle through dataset ~2.5 times. This may lead to overfitting on repeated data.
[WARNING] [Rank 0] Dataset shard exhausted after 100 samples. Starting cycle #2...
[INFO] [Rank 0] Cache regeneration complete. Generated 250 samples with 2 complete cycles through the dataset shard (plus 50 samples from partial cycle).
```

## Benefits

1. **Transparency**: Users now know when their model is seeing repeated data
2. **Debugging**: Easy to identify if poor performance is due to excessive data repetition
3. **Monitoring**: WandB metrics track cycling behavior across training
4. **Awareness**: Helps users choose appropriate `samples_per_regeneration_cycle` values

## Usage Recommendations

For 8xH100 setup with large datasets:
- Set `samples_per_regeneration_cycle` < (total_dataset_size / 8) to avoid cycling
- Monitor `cache/cycles_per_regeneration` metric in WandB
- If seeing high cycle counts, consider:
  - Reducing `samples_per_regeneration_cycle`
  - Using a larger pretokenized dataset
  - Adjusting regeneration frequency