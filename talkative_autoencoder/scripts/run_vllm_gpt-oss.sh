#!/bin/bash
#SBATCH -J vllm_oss20b_tp4
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -t 08:00:00
set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "$SLURM_SUBMIT_DIR"
fi
source talkative_autoencoder/scripts/ensure_env.sh

# Ensure we're in the project directory so uv uses the correct pyproject
cd "$UV_PROJECT_ROOT"

# Sync env with correct pre-release/index strategy (updates env; avoids later solver during run)
uv sync --index-strategy unsafe-best-match --prerelease allow
# Ensure build tools for FlashInfer JIT are present
"$UV_PROJECT_ENVIRONMENT/bin/python" -m pip install --upgrade pip >/dev/null 2>&1 || true
"$UV_PROJECT_ENVIRONMENT/bin/python" -m pip install ninja cmake -q || true

"$UV_PROJECT_ENVIRONMENT/bin/vllm" serve openai/gpt-oss-20b \
  --dtype auto --max-model-len 4096 \
  --tensor-parallel-size 4 \
  --port 8000 --gpu-memory-utilization 0.90 &
for i in $(seq 1 120); do
  if curl -sf http://localhost:8000/v1/models >/dev/null; then
    echo "vLLM server is ready"
    break
  fi
  echo "Waiting for vLLM server... ($i)"; sleep 2
done
"$UV_PROJECT_ENVIRONMENT/bin/python" -m talkative_autoencoder.scripts.generate_thinking_dataset \
  --input-dataset "allenai/WildChat-1M" --input-split train \
  --server-urls http://localhost:8000/v1 \
  --model openai/gpt-oss-20b \
  --output-path /workspace/kitf/talkative-probes/talkative_autoencoder/data/gpt_oss_generated/train \
  --target-total-tokens 4000 --anchor-prompt-tokens 3000 \
  --max-concurrency 64 --reasoning-level medium