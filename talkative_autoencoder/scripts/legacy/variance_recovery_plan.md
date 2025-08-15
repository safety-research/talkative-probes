# Plan: Variance Recovery vs Best-of-K Analysis

## Objective
Create a comprehensive analysis showing how variance recovery improves as we increase K in best-of-K sampling for the talkative autoencoder evaluation.

## Current Issues to Fix First

### 1. Evaluation Script Issues
Based on Gemini's review, we need to fix:
- The `validate_distributed` function only returns `avg_mse`, not full metrics
- DDP initialization uses wrong device identifier
- Missing variance recovery calculation in best-of-N mode

### 2. Variance Recovery Calculation
Currently, the best-of-N mode only tracks MSE. We need to:
- Modify `LensAnalyzer` to return activations (A and A_hat) OR
- Calculate variance recovery directly within the best-of-N evaluation loop
- Ensure we calculate full-batch variance (not per-example)

## Implementation Plan

### Step 1: Fix Core Issues
1. Modify the evaluation script to correctly handle metrics from `validate_distributed`
2. Fix DDP initialization issue
3. Add a simplified variance recovery calculation for best-of-N

### Step 2: Add Variance Recovery to Best-of-N
Since `LensAnalyzer.analyze_all_tokens()` doesn't return activations, we need to:
1. Create a custom function that:
   - Uses the analyzer's encoder/decoder directly
   - Generates K explanations per position
   - Selects the best based on MSE
   - Returns both MSE and the activations for variance calculation
2. Accumulate activations across the entire validation set
3. Calculate variance recovery on the full batch

### Step 3: Create K-Sweep Configuration
1. Create config files for different K values:
   - `config/eval/best_of_k_sweep.yaml` - base config
   - K values to test: 1, 2, 4, 8, 16, 32, 64
2. Create a bash script to run all K values sequentially
3. Collect results and format for plotting

### Step 4: Output Format
Results should include:
- K value
- Average MSE
- Variance recovery (fraction of variance explained)
- R-squared
- Total samples evaluated
- Time taken

### Step 5: Documentation
Update `eval_examples.md` with:
- Instructions for running the K-sweep
- Expected output format
- Example plotting code (matplotlib/seaborn)

## Technical Details

### Variance Recovery Formula
```
variance_recovery = 1 - MSE / Var(A)
```
Where:
- MSE = mean((A - A_hat)^2) across all positions and examples
- Var(A) = variance of original activations
- A = original activations from the model
- A_hat = reconstructed activations from best-of-K

### Implementation Approach
Instead of modifying `LensAnalyzer`, we'll:
1. Access the analyzer's encoder and decoder directly
2. Implement our own best-of-K loop that preserves activations
3. Calculate variance recovery on accumulated data

## Files to Create/Modify

1. **scripts/02_eval.py** - Fix issues and add variance recovery calculation
2. **scripts/03_best_of_k_sweep.py** - New script specifically for K-sweep analysis
3. **config/eval/k_sweep/k_*.yaml** - Config files for each K value
4. **scripts/run_k_sweep.sh** - Bash script to run all K values
5. **scripts/eval_examples.md** - Update with K-sweep instructions

## Validation Steps

1. Test single K value to ensure variance recovery is calculated correctly
2. Compare variance recovery from standard validation vs best-of-1
3. Run full K-sweep on a small subset to verify trends
4. Generate plot to visualize results