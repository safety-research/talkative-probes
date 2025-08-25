#!/bin/bash
# Local (non-Slurm) launcher for gpt-oss generation
# - Uses all visible GPUs on the current node (tensor-parallel = GPU count)
# - Starts a local vLLM server and runs the generator with pause/resume
#
# Example:
#   OUTPUT_PATH=/workspace/kitf/talkative-probes/talkative_autoencoder/data/gpt_oss_generated2/train \
#   INPUT_DATASET="allenai/WildChat-1M" \
#   bash talkative_autoencoder/scripts/run_local_vllm_gpt-oss.sh
#
# Optional envs:
#   MODEL_NAME (default openai/gpt-oss-20b)
#   VLLM_MAX_LEN (default 4096)
#   VLLM_DTYPE (default auto)
#   VLLM_GPU_UTIL (default 0.90)
#   PORT (default 8000)
#   NICE_LEVEL_SERVER (default 10)
#   TARGET_TOTAL_TOKENS (default 4000)
#   ANCHOR_PROMPT_TOKENS (default 3000)
#   MAX_CONCURRENCY (default 64)
#   REASONING_LEVEL (default medium)
#   MAX_SAMPLES (optional)

set -euo pipefail

# Find repo root containing talkative_autoencoder
find_repo_root() {
  local d
  d="$(pwd)"
  while [ "$d" != "/" ]; do
    if [ -d "$d/talkative_autoencoder" ]; then
      echo "$d"; return 0
    fi
    d="$(dirname "$d")"
  done
  return 1
}

REPO_ROOT="$(find_repo_root || true)"
if [ -z "$REPO_ROOT" ]; then
  echo "ERROR: Could not locate repo root from $(pwd)" >&2
  exit 1
fi
cd "$REPO_ROOT"

source talkative_autoencoder/scripts/ensure_env.sh

MODEL_NAME=${MODEL_NAME:-openai/gpt-oss-20b}
VLLM_MAX_LEN=${VLLM_MAX_LEN:-4096}
VLLM_DTYPE=${VLLM_DTYPE:-auto}
VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.90}
PORT=${PORT:-8000}
NICE_LEVEL_SERVER=${NICE_LEVEL_SERVER:-10}

INPUT_DATASET=${INPUT_DATASET:-"allenai/WildChat-1M"}
INPUT_SPLIT=${INPUT_SPLIT:-train}
OUTPUT_PATH=${OUTPUT_PATH:-/workspace/kitf/talkative-probes/talkative_autoencoder/data/gpt_oss_generated2/train}
TARGET_TOTAL_TOKENS=${TARGET_TOTAL_TOKENS:-4000}
ANCHOR_PROMPT_TOKENS=${ANCHOR_PROMPT_TOKENS:-3000}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-64}
REASONING_LEVEL=${REASONING_LEVEL:-medium}
GEN_TIMEOUT=${GEN_TIMEOUT:-120}

# Detect GPU count
detect_gpu_count() {
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    # Count comma-separated entries
    IFS=',' read -r -a devs <<< "$CUDA_VISIBLE_DEVICES"
    echo "${#devs[@]}"
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local n
    n=$(nvidia-smi -L 2>/dev/null | wc -l | awk '{print $1}')
    if [ -n "$n" ] && [ "$n" -gt 0 ]; then echo "$n"; return 0; fi
  fi
  echo 1
}

TP_SIZE=$(detect_gpu_count)
echo "Detected GPUs: $TP_SIZE (tensor-parallel-size=$TP_SIZE)"

# Local registry dir
REGISTRY_DIR=${REGISTRY_DIR:-$REPO_ROOT/talkative_autoencoder/data/vllm_registry}
mkdir -p "$REGISTRY_DIR"
URL_FILE="$REGISTRY_DIR/server_local_$$.url"
echo "http://localhost:${PORT}/v1" > "$URL_FILE"

cleanup() {
  if [ -n "${LOCAL_PID:-}" ]; then
    kill "$LOCAL_PID" 2>/dev/null || true
  fi
  rm -f "$URL_FILE" || true
}
trap cleanup EXIT

# Start isolated vLLM server using uv
export TAE_SERVE_ISOLATED=1
SERVE_DIR="${SLURM_TMPDIR:-$HOME/.cache}/vllm-serve-local-$$"
mkdir -p "$SERVE_DIR"
cd "$SERVE_DIR"

uv venv --python 3.12
source .venv/bin/activate
uv pip install -q ninja cmake || true
uv pip install --pre vllm==0.10.1+gptoss \
  --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
  --index-strategy unsafe-best-match

if [ -n "$NICE_LEVEL_SERVER" ]; then
  NICE_PREFIX=(nice -n "$NICE_LEVEL_SERVER")
else
  NICE_PREFIX=()
fi

"${NICE_PREFIX[@]}" vllm serve "$MODEL_NAME" \
  --dtype "$VLLM_DTYPE" \
  --max-model-len "$VLLM_MAX_LEN" \
  --tensor-parallel-size "$TP_SIZE" \
  --port "$PORT" \
  --gpu-memory-utilization "$VLLM_GPU_UTIL" \
  > vllm_local_$$.log 2>&1 &

LOCAL_PID=$!

# Wait for readiness
READY=0
for i in $(seq 1 600); do
  if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null; then
    READY=1; break
  fi
  sleep 1
done

if [ "$READY" -ne 1 ]; then
  echo "Local vLLM server failed to start" >&2
  exit 1
fi

# Run generator from the repo root to ensure module resolution
cd "$REPO_ROOT"

# Optional max-samples argument
if [ -n "${MAX_SAMPLES:-}" ]; then
  MS_ARGS=(--max-samples "$MAX_SAMPLES")
else
  MS_ARGS=()
fi

uv run python -m talkative_autoencoder.scripts.generate_thinking_dataset \
  --input-dataset "$INPUT_DATASET" --input-split "$INPUT_SPLIT" \
  --server-urls "http://localhost:${PORT}/v1" \
  --model "$MODEL_NAME" \
  --output-path "$OUTPUT_PATH" \
  --target-total-tokens "$TARGET_TOTAL_TOKENS" --anchor-prompt-tokens "$ANCHOR_PROMPT_TOKENS" \
  --max-concurrency "$MAX_CONCURRENCY" --reasoning-level "$REASONING_LEVEL" \
  --timeout "$GEN_TIMEOUT" --resume \
  "${MS_ARGS[@]}"

echo "Done. Output at: $OUTPUT_PATH"


