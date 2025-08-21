#!/bin/bash
#SBATCH -J vllm_oss20b_coord
#SBATCH -N 1
#SBATCH -t 72:00:00
# Coordinator is CPU-only; do not request GPUs

#
# Coordinator for gpt-oss data generation (CPU-only)
# - Submits NUM_SERVERS independent GPU server jobs (each cancellable)
# - Maintains a live registry of healthy servers
# - Runs the generator with pause/resume enabled against the registry
# - Supports auto scale-in/out (refill missing servers) or manual scaling
#
# Default: submit 4 servers, each on an 8× GPU node (TP=8). Not all may start immediately.
#   OUTPUT_PATH=/workspace/kitf/talkative-probes/talkative_autoencoder/data/gpt_oss_generated2/train \
#   INPUT_DATASET="allenai/WildChat-1M" \
#   sbatch talkative_autoencoder/scripts/run_vllm_gpt-oss.sh
#
# Alternate: many 1× GPU servers
#   NUM_SERVERS=16 GPUS_PER_SERVER=1 TP_SIZE_PER_SERVER=1 \
#   OUTPUT_PATH=/path/to/out INPUT_DATASET=... sbatch talkative_autoencoder/scripts/run_vllm_gpt-oss.sh
#
# Pause/resume:
# - To pause: scancel <COORDINATOR_JOB_ID>
# - To resume: resubmit the same command. The generator reads <OUTPUT_PATH>/data.jsonl and
#   skips completed indices, then rewrites the HF dataset under <OUTPUT_PATH>.
# - Set AUTO_TEARDOWN=0 to keep servers running when the coordinator exits.
#
# Scale in/out while running:
# - Manual scale-out: submit extra server jobs; they will join the registry automatically:
#     MODEL_NAME=openai/gpt-oss-20b TP_SIZE=1 VLLM_PORT=8077 \
#     REGISTRY_DIR=/workspace/kitf/talkative-probes/talkative_autoencoder/data/vllm_registry \
#     sbatch -N 1 --gres=gpu:1 talkative_autoencoder/scripts/vllm_server_job.sh
# - Manual scale-in: scancel <SERVER_JOB_ID> and the server URL will be removed automatically.
# - Auto-maintain desired count: MAINTAIN_DESIRED=1 (default). Set MAINTAIN_DESIRED=0 to disable.
#
# Monitor:
# - Registry: ${REGISTRY_DIR:-...}/registry.txt (live server URLs)
# - Server job IDs: /workspace/kitf/talkative-probes/.coord/server_jobs.txt
# - Logs: tail -f slurm-<JOBID>.out (coordinator and server jobs)
# - Progress: <OUTPUT_PATH>/data.jsonl (streaming), <OUTPUT_PATH>/progress.json (stats)
#
# Standalone generator (against a live registry):
#   uv run python -m talkative_autoencoder.scripts.generate_thinking_dataset \
#     --input-dataset "allenai/WildChat-1M" --input-split train \
#     --server-registry /workspace/kitf/talkative-probes/talkative_autoencoder/data/vllm_registry/registry.txt \
#     --model openai/gpt-oss-20b \
#     --output-path <OUTPUT_PATH> \
#     --target-total-tokens 4000 --anchor-prompt-tokens 3000 \
#     --max-concurrency 64 --reasoning-level medium --max-samples 10000 \
#     --resume
#
# Optional envs:
# - MODEL_NAME, VLLM_MAX_LEN, VLLM_DTYPE, VLLM_GPU_UTIL, GEN_TIMEOUT, BASE_PORT,
#   REGISTRY_DIR, AUTO_TEARDOWN, EXTRA_SBATCH_FLAGS, MAX_WAIT_SECS (default 86400),
#   NICE_LEVEL_SERVER

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "$SLURM_SUBMIT_DIR"
fi

# Anchor helper scripts to the submission working directory (shared FS)
SERVER_SCRIPT="$PWD/talkative_autoencoder/scripts/vllm_server_job.sh"
ENSURE_ENV_SCRIPT="$PWD/talkative_autoencoder/scripts/ensure_env.sh"

if [ ! -f "$SERVER_SCRIPT" ]; then
  echo "ERROR: Cannot find server script at $SERVER_SCRIPT" >&2
  exit 1
fi
if [ ! -f "$ENSURE_ENV_SCRIPT" ]; then
  echo "ERROR: Cannot find ensure_env script at $ENSURE_ENV_SCRIPT" >&2
  exit 1
fi

chmod +x "$SERVER_SCRIPT" 2>/dev/null || true
source "$ENSURE_ENV_SCRIPT"

# ---------- User-configurable parameters ----------
# Number of vLLM server jobs to launch (default: 4 servers)
NUM_SERVERS=${NUM_SERVERS:-4}
# GPUs per server job (default: 8 GPUs)
GPUS_PER_SERVER=${GPUS_PER_SERVER:-8}
# Tensor-parallel size per server (default: 8)
TP_SIZE_PER_SERVER=${TP_SIZE_PER_SERVER:-8}
# Optional extra sbatch flags (advanced), e.g. "--constraint=h100 --qos=high"
EXTRA_SBATCH_FLAGS=${EXTRA_SBATCH_FLAGS:-}

# Model and vLLM args
MODEL_NAME=${MODEL_NAME:-openai/gpt-oss-20b}
VLLM_MAX_LEN=${VLLM_MAX_LEN:-4096}
VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.90}
VLLM_DTYPE=${VLLM_DTYPE:-auto}

# Registry path for live servers (shared FS)
REGISTRY_DIR=${REGISTRY_DIR:-/workspace/kitf/talkative-probes/talkative_autoencoder/data/vllm_registry}
REGISTRY_TXT="$REGISTRY_DIR/registry.txt"
mkdir -p "$REGISTRY_DIR"

# Generator args
INPUT_DATASET=${INPUT_DATASET:-"allenai/WildChat-1M"}
INPUT_SPLIT=${INPUT_SPLIT:-train}
OUTPUT_PATH=${OUTPUT_PATH:-/workspace/kitf/talkative-probes/talkative_autoencoder/data/gpt_oss_generated2/train}
TARGET_TOTAL_TOKENS=${TARGET_TOTAL_TOKENS:-4000}
ANCHOR_PROMPT_TOKENS=${ANCHOR_PROMPT_TOKENS:-3000}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-64}
REASONING_LEVEL=${REASONING_LEVEL:-medium}
MAX_SAMPLES=${MAX_SAMPLES:-}
GEN_TIMEOUT=${GEN_TIMEOUT:-120}

# Optional niceness levels
NICE_LEVEL_SERVER=${NICE_LEVEL_SERVER:-10}

# Max time to wait for at least one server (seconds). Default: one day
MAX_WAIT_SECS=${MAX_WAIT_SECS:-86400}

# Automanage servers to maintain NUM_SERVERS; set 0 to skip auto-scale-out
MAINTAIN_DESIRED=${MAINTAIN_DESIRED:-1}
# Cancel servers on exit
AUTO_TEARDOWN=${AUTO_TEARDOWN:-1}

# Coordinator working dir
COORD_DIR=${COORD_DIR:-/workspace/kitf/talkative-probes/.coord}
mkdir -p "$COORD_DIR"
JOBLIST_FILE="$COORD_DIR/server_jobs.txt"
> "$JOBLIST_FILE"
> "$REGISTRY_TXT"

submit_server_job() {
  local port="$1"
  local sbatch_opts=("--parsable" "-N" "1" "--gres=gpu:${GPUS_PER_SERVER}")
  if [ -n "$EXTRA_SBATCH_FLAGS" ]; then
    # Safely split EXTRA_SBATCH_FLAGS into array words
    read -r -a _extra_flags <<< "$EXTRA_SBATCH_FLAGS"
    sbatch_opts+=("${_extra_flags[@]}")
  fi
  local export_vars="ALL,MODEL_NAME=$MODEL_NAME,TP_SIZE=$TP_SIZE_PER_SERVER,VLLM_PORT=$port,VLLM_MAX_LEN=$VLLM_MAX_LEN,VLLM_GPU_UTIL=$VLLM_GPU_UTIL,VLLM_DTYPE=$VLLM_DTYPE,REGISTRY_DIR=$REGISTRY_DIR,GPUS_PER_SERVER=$GPUS_PER_SERVER,NICE_LEVEL_SERVER=$NICE_LEVEL_SERVER,COORD_SUBMIT_DIR=$PWD"
  local jid
  jid=$(sbatch "${sbatch_opts[@]}" --export="$export_vars" "$SERVER_SCRIPT")
  echo "$jid" | tee -a "$JOBLIST_FILE"
}

scancel_joblist() {
  if [ -f "$JOBLIST_FILE" ]; then
    while read -r jid; do
      [ -n "${jid:-}" ] || continue
      scancel "$jid" || true
    done < "$JOBLIST_FILE"
  fi
}

cleanup() {
  if [ "${AUTO_TEARDOWN}" = "1" ]; then
    echo "Teardown: cancelling server jobs"
    scancel_joblist || true
  fi
  if [ -n "${MAINTAIN_PID:-}" ] && kill -0 "$MAINTAIN_PID" 2>/dev/null; then
    kill "$MAINTAIN_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

update_registry() {
  # Build registry.txt from per-job url files and health check
  local tmp="$REGISTRY_DIR/registry.tmp"
  : > "$tmp"
  for f in "$REGISTRY_DIR"/server_*.url; do
    [ -f "$f" ] || continue
    local url
    url=$(cat "$f" | head -n1 || true)
    if [ -z "$url" ]; then continue; fi
    if curl -sf "${url}/models" >/dev/null; then
      echo "$url" >> "$tmp"
    fi
  done
  sort -u "$tmp" > "$REGISTRY_TXT" || true
  rm -f "$tmp" || true
}

echo "Launching ${NUM_SERVERS} server jobs..."
# Choose distinct ports to avoid potential reuse if multiple servers land on same node
BASE_PORT=${BASE_PORT:-8000}
for i in $(seq 0 $((NUM_SERVERS-1))); do
  submit_server_job $((BASE_PORT + i))
done

echo "Waiting for at least 1 server to become ready (up to ${MAX_WAIT_SECS}s)..."
READY=0
start_ts=$(date +%s)
i=0
while true; do
  now_ts=$(date +%s)
  elapsed=$((now_ts - start_ts))
  if [ "$elapsed" -ge "$MAX_WAIT_SECS" ]; then
    break
  fi
  update_registry
  if [ -s "$REGISTRY_TXT" ]; then READY=1; break; fi
  # No auto-resubmit here; we may be pending due to no GPUs available
  i=$((i+1))
  echo "Waiting for servers... ($i) elapsed=${elapsed}s"; sleep 2
done

if [ "$READY" -ne 1 ]; then
  echo "ERROR: No servers became ready. Exiting."
  exit 1
fi

echo "Servers ready. Starting generator (pause/resume enabled)."

# Start background maintenance: refresh registry and optionally maintain desired count
maintain_loop() {
  while true; do
    update_registry
    if [ "$MAINTAIN_DESIRED" = "1" ]; then
      # Maintain based on Slurm active jobs (PD/R/etc.), not registry, to avoid over-submitting
      if [ -s "$JOBLIST_FILE" ]; then
        ids_csv=$(paste -sd, "$JOBLIST_FILE")
        if [ -n "$ids_csv" ]; then
          active_count=$(squeue -h -j "$ids_csv" | wc -l | awk '{print $1}')
          # Also prune JOBLIST_FILE to active IDs only
          active_ids=$(squeue -h -j "$ids_csv" -o "%i" | sort -u)
          : > "$JOBLIST_FILE.tmp"
          if [ -n "$active_ids" ]; then
            while read -r ajid; do
              [ -n "$ajid" ] && echo "$ajid" >> "$JOBLIST_FILE.tmp"
            done <<< "$active_ids"
          fi
          mv "$JOBLIST_FILE.tmp" "$JOBLIST_FILE"
        else
          active_count=0
        fi
      else
        active_count=0
      fi

      if [ "$active_count" -lt "$NUM_SERVERS" ]; then
        missing=$((NUM_SERVERS - active_count))
        for j in $(seq 1 $missing); do
          submit_server_job $((BASE_PORT + RANDOM % 1000 + 8000))
        done
      fi
    fi
    sleep 5
  done
}
maintain_loop &
MAINTAIN_PID=$!

# Optional max-samples argument
if [ -n "${MAX_SAMPLES:-}" ]; then
  MAX_SAMPLES_ARG=(--max-samples "$MAX_SAMPLES")
else
  MAX_SAMPLES_ARG=()
fi

uv run python -m talkative_autoencoder.scripts.generate_thinking_dataset \
  --input-dataset "$INPUT_DATASET" --input-split "$INPUT_SPLIT" \
  --server-registry "$REGISTRY_TXT" \
  --model "$MODEL_NAME" \
  --output-path "$OUTPUT_PATH" \
  --target-total-tokens "$TARGET_TOTAL_TOKENS" --anchor-prompt-tokens "$ANCHOR_PROMPT_TOKENS" \
  --max-concurrency "$MAX_CONCURRENCY" --reasoning-level "$REASONING_LEVEL" \
  --timeout "$GEN_TIMEOUT" --resume \
  "${MAX_SAMPLES_ARG[@]}"

echo "Generator finished. Registry remains available at: $REGISTRY_TXT"