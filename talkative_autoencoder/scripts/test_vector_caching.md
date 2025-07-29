# Vector Caching Test

To test the vector caching functionality in the K-sweep script:

## 1. Run with caching enabled to cache a large number of vectors:
```bash
source scripts/ensure_env.sh && CUDA_VISIBLE_DEVICES=4 uv run python scripts/03_best_of_k_sweep.py \
    +eval.checkpoint_path=outputs/5M_L5_SimpleStories_lr1e-4_t10_1222_1430/JL_B14_L5_SIMPLE_OJ_IT_E_D_WILD_OTF_dist8/checkpoint_step399999_epoch240_final.pt \
    +eval.k_values=[1] \
    +eval.max_batches=20 \
    +eval.load_store=true \
    +eval.cache_dir=vector_cache_test
```

## 2. Run again requesting fewer samples (will use subset of cached vectors):
```bash
source scripts/ensure_env.sh && CUDA_VISIBLE_DEVICES=4 uv run python scripts/03_best_of_k_sweep.py \
    +eval.checkpoint_path=outputs/5M_L5_SimpleStories_lr1e-4_t10_1222_1430/JL_B14_L5_SIMPLE_OJ_IT_E_D_WILD_OTF_dist8/checkpoint_step399999_epoch240_final.pt \
    +eval.k_values=[1,2,4,8,16,32] \
    +eval.max_batches=5 \
    +eval.load_store=true \
    +eval.cache_dir=vector_cache_test
```

The second run should print something like:
"Cache has 20 vectors (20 samples), using first 5 vectors (5 samples)"

## 3. Check cache structure:
```bash
ls -la vector_cache_test/
```

You should see a directory with a hash name containing rank-specific pickle files.

## Key points:
- Cache key is based on: model name, layer, dataset config, and num_positions (NOT max_batches)
- This allows reusing cached vectors even when evaluating different numbers of samples
- If you cached 50000 vectors but only need 10000, it will use the first 10000
- Each rank saves its own cache file (rank_0_vectors.pkl, rank_1_vectors.pkl, etc.)
- Cached data includes: A vectors, positions, and token IDs
- Cache is only used when +eval.load_store=true

## Example use case:
1. Cache vectors from entire validation set (e.g., max_batches=None or large number)
2. Run multiple K-sweep experiments with different max_batches values
3. All experiments will reuse the same cached vectors, just using the subset they need