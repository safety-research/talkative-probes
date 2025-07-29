#!/bin/bash
# Test script to verify vector caching efficiency

echo "=== Vector Caching Efficiency Test ==="
echo "This test verifies that when cache exists, we skip vector extraction in the dataloader"
echo ""

# Clean up any existing cache
rm -rf vector_cache_test_efficiency

# Test 1: First run - should extract vectors
echo "Test 1: First run (no cache) - should extract vectors"
echo "Watch for: 'Best-of-1 Eval' progress bar (indicates extraction)"
source scripts/ensure_env.sh && CUDA_VISIBLE_DEVICES=4 uv run python scripts/03_best_of_k_sweep.py \
    +eval.checkpoint_path=outputs/5M_L5_SimpleStories_lr1e-4_t10_1222_1430/JL_B14_L5_SIMPLE_OJ_IT_E_D_WILD_OTF_dist8/checkpoint_step399999_epoch240_final.pt \
    +eval.k_values=[1] \
    +eval.max_batches=2 \
    +eval.load_store=true \
    +eval.cache_dir=vector_cache_test_efficiency \
    +eval.num_positions=1

echo ""
echo "Test 2: Second run (with cache) - should skip extraction"
echo "Watch for:"
echo "  - 'Cache exists at ... will skip vector extraction in dataloader'"
echo "  - 'Using cached vectors from ...'"
echo "  - NO 'Best-of-1 Eval' progress bar (extraction skipped)"
source scripts/ensure_env.sh && CUDA_VISIBLE_DEVICES=4 uv run python scripts/03_best_of_k_sweep.py \
    +eval.checkpoint_path=outputs/5M_L5_SimpleStories_lr1e-4_t10_1222_1430/JL_B14_L5_SIMPLE_OJ_IT_E_D_WILD_OTF_dist8/checkpoint_step399999_epoch240_final.pt \
    +eval.k_values=[1,2,4] \
    +eval.max_batches=2 \
    +eval.load_store=true \
    +eval.cache_dir=vector_cache_test_efficiency \
    +eval.num_positions=1

# Clean up
rm -rf vector_cache_test_efficiency