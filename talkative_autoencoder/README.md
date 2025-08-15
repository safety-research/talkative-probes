# Consistency Lenses: Scalable LLM Interpretation via textual bottlenecks

This document outlines the architectural plan for training talkative probes as autoencoders to interpretthe internal states of LLMs by forcing them through a textual, human-readable bottleneck.

```bash
make
```
The main entrypoint is scripts/submit_with_config.sh, which torchruns 01_distributed_train.py. We configure hyperparameters via `conf/` using hydra.

Default is data parallel.

## Environment setup (uv)

```bash
make           # installs uv if needed, locks and syncs env (default+dev groups)
make sync-env  # re-locks and syncs with consistent prerelease/index flags
```

Notes:
- We use `uv lock --index-strategy unsafe-best-match --prerelease allow` and then `uv sync --group dev --index-strategy unsafe-best-match --prerelease allow` to keep prerelease/index behavior consistent and to avoid pulling vLLM into the base env.
- vLLM serving stack is installed on-demand by the Slurm script via `uv pip install --pre vllm==0.10.1+gptoss ...` to avoid solver conflicts in the base env.

## Serving gpt-oss-20b via vLLM (8× H100, with thinking)

Primary path (single node, TP=8):
```bash
sbatch talkative_autoencoder/scripts/run_vllm_gpt-oss.sh
```
- The script:
  - sources `scripts/ensure_env.sh`
  - locks/syncs env
  - ensures build tools (ninja/cmake) for FlashInfer JIT
  - starts vLLM with `--tensor-parallel-size 8`, waits for readiness
  - runs `talkative_autoencoder/scripts/generate_thinking_dataset.py`

Generation script:
- Stores messages-only (chat format) with native `thinking` for gpt-oss
- Prints first server output and finalized example
- Logs token length stats (prompt, prompt+system, thinking, content)

Pretokenize the generated dataset (preserves `thinking` for gpt-oss):
```bash
uv run python -m talkative_autoencoder.scripts.pretokenize_dataset \
  activation_dumper.hf_dataset_name=/abs/path/data/gpt_oss_generated/train \
  orig_tokenizer_name=openai/gpt-oss-20b \
  activation_dumper.seq_len=4096 \
  pretokenize.force=true
```

For multi-node options and Ray-based TP, see the top-of-file docstring in `scripts/generate_thinking_dataset.py`.