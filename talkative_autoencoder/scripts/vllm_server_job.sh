#!/bin/bash
#SBATCH -J vllm_oss20b_srv
#SBATCH -t 72:00:00
# One GPU per server by default; override via --export GPUS_PER_SERVER
#SBATCH --gres=gpu:1

set -euo pipefail

#
# Per-node vLLM server job for gpt-oss
# - Launches a single vLLM server (default 1 GPU, TP=1; override via --export)
# - Writes its URL to REGISTRY_DIR/server_${SLURM_JOB_ID}.url
# - Coordinator reads registry.txt and the generator routes requests dynamically
#
# Quick submit example (manual scale-out):
#   MODEL_NAME=openai/gpt-oss-20b TP_SIZE=1 VLLM_PORT=8077 \
#   REGISTRY_DIR=/workspace/kitf/talkative-probes/talkative_autoencoder/data/vllm_registry \
#   sbatch -N 1 --gres=gpu:1 talkative_autoencoder/scripts/vllm_server_job.sh
#
# Environment variables (override with --export or sbatch --export):
# - MODEL_NAME: HF model id (default openai/gpt-oss-20b)
# - TP_SIZE: tensor parallel size (default 1)
# - VLLM_PORT: server port (default 8000)
# - VLLM_MAX_LEN: max model len (default 4096)
# - VLLM_GPU_UTIL: GPU memory util (default 0.90)
# - VLLM_DTYPE: dtype (default auto)
# - REGISTRY_DIR: shared registry directory (default repo data/vllm_registry)
# - GPUS_PER_SERVER: requested GPUs (default 1; controlled by --gres as well)
#
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "$SLURM_SUBMIT_DIR"
fi

# If coordinator passed its submit dir, use it to locate scripts on shared FS
COORD_SUBMIT_DIR=${COORD_SUBMIT_DIR:-}
if [ -n "$COORD_SUBMIT_DIR" ]; then
  cd "$COORD_SUBMIT_DIR"
fi

source talkative_autoencoder/scripts/ensure_env.sh

MODEL_NAME=${MODEL_NAME:-openai/gpt-oss-20b}
TP_SIZE=${TP_SIZE:-1}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_MAX_LEN=${VLLM_MAX_LEN:-4096}
VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.90}
VLLM_DTYPE=${VLLM_DTYPE:-auto}
REGISTRY_DIR=${REGISTRY_DIR:-/workspace/kitf/talkative-probes/talkative_autoencoder/data/vllm_registry}
GPUS_PER_SERVER=${GPUS_PER_SERVER:-1}
NICE_LEVEL_SERVER=${NICE_LEVEL_SERVER:-}

export TAE_SERVE_ISOLATED=1

# Create an isolated working dir and uv venv (per upstream guidance)
if [ -n "${SLURM_TMPDIR:-}" ]; then
  SERVE_DIR="$SLURM_TMPDIR/vllm-serve-$SLURM_JOB_ID"
else
  SERVE_DIR="$HOME/.cache/vllm-serve-$SLURM_JOB_ID"
fi
mkdir -p "$SERVE_DIR"
cd "$SERVE_DIR"

uv venv --python 3.12
source .venv/bin/activate

uv pip install -q ninja cmake || true
uv pip install --pre vllm==0.10.1+gptoss \
  --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
  --index-strategy unsafe-best-match

# Expose URL file for coordinator
mkdir -p "$REGISTRY_DIR"
URL_FILE="$REGISTRY_DIR/server_${SLURM_JOB_ID}.url"
echo "http://$(hostname -f):${VLLM_PORT}/v1" > "$URL_FILE"

# Launch server (Slurm constrains visible GPUs via --gres)
if [ -n "$NICE_LEVEL_SERVER" ]; then
  NICE_PREFIX=(nice -n "$NICE_LEVEL_SERVER")
else
  NICE_PREFIX=()
fi

"${NICE_PREFIX[@]}" vllm serve "$MODEL_NAME" \
  --dtype "$VLLM_DTYPE" \
  --max-model-len "$VLLM_MAX_LEN" \
  --tensor-parallel-size "$TP_SIZE" \
  --port "$VLLM_PORT" \
  --gpu-memory-utilization "$VLLM_GPU_UTIL" \
  > vllm_${SLURM_JOB_ID}.log 2>&1 &

PID=$!

# Wait until ready or timeout
READY=0
for i in $(seq 1 600); do
  if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null; then
    READY=1; break
  fi
  sleep 1
done

if [ "$READY" -ne 1 ]; then
  echo "Server failed to start" >&2
  kill "$PID" 2>/dev/null || true
  rm -f "$URL_FILE" || true
  exit 1
fi

echo "Server ${SLURM_JOB_ID} ready on port ${VLLM_PORT}"

# Keep job alive until server exits; cleanup URL file on exit
wait "$PID" || true
rm -f "$URL_FILE" || true


