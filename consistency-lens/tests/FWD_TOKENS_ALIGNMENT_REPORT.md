# fwd_tokens Alignment Report

## Summary

This report documents the alignment between `fwd_tokens` and the autoregressive generation methods (`generate_soft`, `generate_soft_kv_cached`, `generate_soft_kv_cached_nondiff`) in the consistency-lens decoder.

## Key Findings

### 1. Bug Identified in fwd_tokens

There is a bug in the current implementation of `fwd_tokens` at line 1676 in `lens/models/decoder.py`:

```python
# Current (buggy) code:
outputs = main_base(inputs_embeds=seq_embs)

# Should be:
outputs = main_base(inputs_embeds=seq_embs, output_hidden_states=True)
```

Without `output_hidden_states=True`, the base model returns logits instead of hidden states, causing a shape mismatch when applying the output head.

### 2. Alignment Verification (with Fix)

With the bug fix applied, `fwd_tokens` correctly aligns with the generation methods:

- **Single-layer patching**: Max probability difference ~2.4e-5 (acceptable numerical precision)
- **Multi-layer patching**: Max probability difference ~9.5e-6 (excellent alignment)

### 3. Multi-layer Patching Support

`fwd_tokens` correctly implements multi-layer patching:

- When `patch_all_layers=True`, it uses `_patched_forward` to apply per-layer projections
- When `patch_all_layers=False`, it uses standard forward pass
- The patching mechanism is consistent with the generation methods

### 4. Code Structure in fwd_tokens

```python
if self.config.patch_all_layers:
    # Custom forward pass with activation patching at all layers
    hidden_states, _ = self._patched_forward(
        main_base=main_base,
        seq_embs=seq_embs,
        activation_input_modified=activation_input_modified,
        use_projection=use_projection,
        do_patching=True,
        prompt_left_emb=prompt_left_emb,
    )
    logits = main_out(hidden_states)
else:
    # Standard forward pass (needs fix)
    outputs = main_base(inputs_embeds=seq_embs, output_hidden_states=True)  # Fix needed
    hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs.hidden_states[-1]
    logits = main_out(hidden_states)
```

## Use Case: Reinforcement Learning

With the bug fix, `fwd_tokens` can be used effectively for RL applications:

1. **Trajectory Generation**: Use `generate_soft_kv_cached` or `generate_soft_kv_cached_nondiff` for fast O(n) generation
2. **Policy Evaluation**: Use `fwd_tokens` to compute log probabilities for policy gradient updates
3. **Consistency**: Both methods use the same patching mechanism (single or multi-layer)

Example RL workflow:
```python
# Generate trajectory
with torch.no_grad():
    trajectory = decoder.generate_soft_kv_cached(activation, max_length=100, gumbel_tau=1.0)
    actions = trajectory.hard_token_ids[0]

# Compute policy probabilities for gradient update
log_probs, entropies = decoder.fwd_tokens(
    activation_input=activation,
    input_tokens=actions
)

# Compute policy gradient loss
rewards = compute_rewards(actions)  # Your reward function
loss = -(log_probs.log() * rewards).mean()
loss.backward()
```

## Recommendations

1. **Fix the bug**: Add `output_hidden_states=True` to line 1676 in `decoder.py`
2. **Test thoroughly**: After fixing, run the alignment tests to ensure correctness
3. **Document the fix**: Update the method documentation to clarify the alignment with generation methods

## Test Files

1. `test_fwd_tokens_debug.py` - Debugged the shape mismatch issue
2. `test_fwd_tokens_workaround.py` - Demonstrated alignment with the fix
3. `test_fwd_tokens_alignment.py` - Attempted comprehensive alignment test (blocked by bug)

## Conclusion

Once the identified bug is fixed, `fwd_tokens` correctly aligns with all generation methods and properly supports both single-layer and multi-layer patching. This enables its use for policy gradient methods in RL applications where fast generation (via KV caching) and accurate probability computation (via fwd_tokens) are both required.