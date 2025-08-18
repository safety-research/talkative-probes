#!/bin/bash
#SBATCH -J vllm_oss20b_tp8
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH -t 08:00:00
set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "$SLURM_SUBMIT_DIR"
fi
source talkative_autoencoder/scripts/ensure_env.sh


uv remove triton triton-kernels
# Ensure we're in the project directory so uv uses the correct pyproject
cd "$UV_PROJECT_ROOT"

# Sync env with correct pre-release/index strategy (updates env; avoids later solver during run)
uv sync --index-strategy unsafe-best-match --prerelease allow
# Ensure build tools for FlashInfer JIT and vLLM are present (avoid pip inside venv)
uv pip install -q ninja cmake || true
uv pip install --pre vllm==0.10.1+gptoss \
  --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
  --index-strategy unsafe-best-match

# Ensure Triton components are aligned with vLLM's MXFP4 requirements
uv pip install --pre 'triton_kernels @ https://github.com/triton-lang/triton/releases/download/v3.0.0/triton_kernels-3.0.0-py3-none-any.whl' || true

"$UV_PROJECT_ENVIRONMENT/bin/vllm" serve openai/gpt-oss-20b \
  --dtype auto --max-model-len 4096 \
  --tensor-parallel-size 8 \
  --port 8000 --gpu-memory-utilization 0.90 &
READY=0
for i in $(seq 1 120); do
  if curl -sf http://localhost:8000/v1/models >/dev/null; then
    echo "vLLM server is ready"
    READY=1; break
  fi
  echo "Waiting for vLLM server... ($i)"; sleep 2
done

if [ "$READY" -ne 1 ]; then
  echo "ERROR: vLLM server did not become ready after waiting. Exiting."
  exit 1
fi
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/generate_thinking_dataset.py \
  --input-dataset "allenai/WildChat-1M" --input-split train \
  --server-urls http://localhost:8000/v1 \
  --model openai/gpt-oss-20b \
  --output-path /workspace/kitf/talkative-probes/talkative_autoencoder/data/gpt_oss_generated/train \
  --target-total-tokens 4000 --anchor-prompt-tokens 3000 \
  --max-concurrency 64 --reasoning-level medium
