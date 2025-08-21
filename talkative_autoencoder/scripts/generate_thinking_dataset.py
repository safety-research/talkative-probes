"""
Generate ~4k-token chat data by regenerating one assistant turn with thinking via vLLM.

Quick start (full instructions below):
  - Start one or more vLLM servers serving `openai/gpt-oss-20b`
  - Run this script pointing to your dataset and server URL(s)
  - First server response and finalized example are printed in full

Model card: openai/gpt-oss-20b (install/serve guidance)
  https://huggingface.co/openai/gpt-oss-20b

1) Install vLLM (MXFP4-native) per node
  source scripts/ensure-env.sh
  uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

2) Start vLLM server(s)
  Main (one 8× H100 node): single endpoint with tensor-parallel size 8
    vllm serve openai/gpt-oss-20b \
      --dtype auto \
      --max-model-len 4096 \
      --tensor-parallel-size 8 \
      --port 8000 \
      --gpu-memory-utilization 0.90

    Slurm single-node example (8× H100):
      #!/bin/bash
      #SBATCH -J vllm_oss20b_tp8
      #SBATCH -N 1
      #SBATCH --gres=gpu:8
      #SBATCH -t 08:00:00
      #SBATCH -p gpu
      set -euo pipefail
      source scripts/ensure-env.sh
      vllm serve openai/gpt-oss-20b \
        --dtype auto --max-model-len 4096 \
        --tensor-parallel-size 8 \
        --port 8000 --gpu-memory-utilization 0.90 &
      sleep 5
      uv run python -m talkative_autoencoder.scripts.generate_thinking_dataset \
        --input-dataset some_org/some_chat_dataset --input-split train \
        --server-urls http://localhost:8000/v1 \
        --model openai/gpt-oss-20b \
        --output-path /abs/path/data/gpt_oss_generated/train \
        --target-total-tokens 4000 --anchor-prompt-tokens 3000 \
        --max-concurrency 64 --reasoning-level high

  Alternate A) Many independent servers (e.g., multiple nodes, 1 GPU each)
    Run one server per node and pass their URLs comma-separated to --server-urls.

  Alternate B) Multi-node single endpoint via Ray (tensor parallel across nodes)
    # head node
    ray start --head --port 6379
    # each worker node
    ray start --address 'HEAD_NODE_IP:6379'
    # then on head
    vllm serve openai/gpt-oss-20b \
      --dtype auto \
      --max-model-len 4096 \
      --tensor-parallel-size 8 \
      --distributed-executor-backend ray \
      --ray-address auto \
      --port 8000 \
      --gpu-memory-utilization 0.90

  Sanity check (local):
    curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
      "model": "openai/gpt-oss-20b",
      "messages": [{"role":"system","content":"Reasoning: high"},{"role":"user","content":"Say hi"}],
      "max_tokens": 64
    }' | jq

3) Run generator (this script)
  Single server:
    uv run python -m talkative_autoencoder.scripts.generate_thinking_dataset \
      --input-dataset some_org/some_chat_dataset --input-split train \
      --server-urls http://localhost:8000/v1 \
      --model openai/gpt-oss-20b \
      --output-path /abs/path/data/gpt_oss_generated/train \
      --target-total-tokens 4000 --anchor-prompt-tokens 3000 \
      --max-concurrency 8 --reasoning-level high

  Multiple servers (round-robin):
    uv run python -m talkative_autoencoder.scripts.generate_thinking_dataset \
      --input-dataset some_org/some_chat_dataset --input-split train \
      --server-urls http://node1:8000/v1,http://node2:8000/v1,http://node3:8000/v1 \
      --model openai/gpt-oss-20b \
      --output-path /abs/path/data/gpt_oss_generated/train \
      --target-total-tokens 4000 --anchor-prompt-tokens 3000 \
      --max-concurrency 64 --reasoning-level high

  Notes:
    - Native Harmony/thinking is preserved for gpt-oss models end-to-end.
    - If your dataset is a JSONL file, pass --input-dataset /abs/path/data.jsonl

4) Pre-tokenize the generated dataset
  uv run python -m talkative_autoencoder.scripts.pretokenize_dataset \
    activation_dumper.hf_dataset_name=/abs/path/data/gpt_oss_generated/train \
    orig_tokenizer_name=openai/gpt-oss-20b \
    activation_dumper.seq_len=4096 \
    pretokenize.force=true

  - The tokenizer keeps native 'thinking' for gpt-oss; for other models it inlines as <think>...</think>.

5) Slurm sbatch examples
  A) Launch independent vLLM servers (one per node) and generate via multi-URL round-robin
    #!/bin/bash
    #SBATCH -J vllm_oss20b
    #SBATCH -N 8                 # adjust
    #SBATCH -n 8                 # one task per node
    #SBATCH --gres=gpu:1         # one GPU per node
    #SBATCH -t 08:00:00
    #SBATCH -p gpu
    set -euo pipefail
    source scripts/ensure-env.sh

    # Start a server on each node (same port per node is fine)
    srun --ntasks=$SLURM_NTASKS --ntasks-per-node=1 bash -lc '
      CUDA_VISIBLE_DEVICES=0 vllm serve openai/gpt-oss-20b \
        --dtype auto --max-model-len 4096 --port 8000 --gpu-memory-utilization 0.90 \
        > vllm_${SLURMD_NODENAME}.log 2>&1 &
      echo $! > vllm_${SLURMD_NODENAME}.pid
      sleep 2
    '

    # Build comma-separated server URL list from allocated nodes
    URLS=$(scontrol show hostnames $SLURM_JOB_NODELIST | awk '{print "http://"$1":8000/v1"}' | paste -sd, -)
    echo "Using server URLs: ${URLS}"

    # Generate data (run on the first node)
    srun -N 1 -n 1 bash -lc '
      uv run python -m talkative_autoencoder.scripts.generate_thinking_dataset \
        --input-dataset some_org/some_chat_dataset --input-split train \
        --server-urls "'"'${URLS}'"'" \
        --model openai/gpt-oss-20b \
        --output-path /abs/path/data/gpt_oss_generated/train \
        --target-total-tokens 4000 --anchor-prompt-tokens 3000 \
        --max-concurrency 64 --reasoning-level high
    '

    # Optional: stop servers
    srun --ntasks=$SLURM_NTASKS --ntasks-per-node=1 bash -lc '
      if [[ -f vllm_${SLURMD_NODENAME}.pid ]]; then kill $(cat vllm_${SLURMD_NODENAME}.pid) || true; fi
    '

  B) Multi-node single endpoint via Ray tensor-parallel
    #!/bin/bash
    #SBATCH -J vllm_ray_oss20b
    #SBATCH -N 8
    #SBATCH --gres=gpu:1
    #SBATCH -t 08:00:00
    #SBATCH -p gpu
    set -euo pipefail
    source scripts/ensure-env.sh

    HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)

    srun -N 1 -w ${HEAD_NODE} bash -lc '
      ray start --head --port 6379
    '

    srun -N $((SLURM_NNODES-1)) -r 1 bash -lc '
      ray start --address "'"'${HEAD_NODE}:6379'"'"
    '

    # Start single logical vLLM server on head (TP across nodes managed by Ray)
    srun -N 1 -w ${HEAD_NODE} bash -lc '
      vllm serve openai/gpt-oss-20b \
        --dtype auto --max-model-len 4096 \
        --tensor-parallel-size 8 \
        --distributed-executor-backend ray --ray-address auto \
        --port 8000 --gpu-memory-utilization 0.90 \
        > vllm_head.log 2>&1 &
      sleep 5
    '

    # Generate data against the single endpoint
    srun -N 1 -w ${HEAD_NODE} bash -lc '
      uv run python -m talkative_autoencoder.scripts.generate_thinking_dataset \
        --input-dataset some_org/some_chat_dataset --input-split train \
        --server-urls http://'"'${HEAD_NODE}'"':8000/v1 \
        --model openai/gpt-oss-20b \
        --output-path /abs/path/data/gpt_oss_generated/train \
        --target-total-tokens 4000 --anchor-prompt-tokens 3000 \
        --max-concurrency 64 --reasoning-level high
    '

    # Optional teardown
    srun -N 1 -w ${HEAD_NODE} bash -lc 'ray stop'
    srun -N $((SLURM_NNODES-1)) -r 1 bash -lc 'ray stop'

  C) Submit the prebuilt script in this repo
    # Edit GPUs/TP/partition as needed inside:
    #   talkative_autoencoder/scripts/run_vllm_gpt-oss.sh
    # Submit:
    sbatch talkative_autoencoder/scripts/run_vllm_gpt-oss.sh

    # Optional: specify account/partition, capture JobID, and monitor
    JOBID=$(sbatch --parsable talkative_autoencoder/scripts/run_vllm_gpt-oss.sh)
    echo "Submitted as $JOBID" && squeue -j $JOBID
    # Tail logs (default slurm-%j.out in the submission CWD)
    tail -f slurm-$JOBID.out

    # To adapt the script for 8× H100, set:
    #   #SBATCH --gres=gpu:8
    #   --tensor-parallel-size 8
    # and adjust #SBATCH -p / -A per your cluster
    # Note: The first line of the script must be a shebang, e.g. '#!/bin/bash'.

6) Notes
  - This script prints the first raw server response and the first finalized example.
  - Dataset contains only 'messages'. Use pretokenize on the saved directory path. It preserves native 'thinking'.
  - Harmony format with 'thinking' is supported natively by gpt-oss.
  - Reference: https://huggingface.co/openai/gpt-oss-20b
"""

#
# Usage summary (pause/resume + dynamic servers)
# - To use dynamic server discovery, pass --server-registry <path/to/registry.txt>.
#   The file should contain one base URL per line (e.g., http://host:8000/v1). This
#   script reloads the file periodically and routes around dropped servers.
# - Pause/resume: run with --resume. The script writes streaming outputs to
#   <output-path>/data.jsonl (one record per line, including the original data index).
#   Cancelling the job and re-running with --resume will skip completed indices
#   and finalize the HF dataset again under <output-path>.
# - Standalone example:
#   uv run python -m talkative_autoencoder.scripts.generate_thinking_dataset \
#     --input-dataset "allenai/WildChat-1M" --input-split train \
#     --server-registry /abs/dir/registry.txt \
#     --model openai/gpt-oss-20b \
#     --output-path /abs/path/data/gpt_oss_generated/train \
#     --target-total-tokens 4000 --anchor-prompt-tokens 3000 \
#     --max-concurrency 64 --reasoning-level medium --max-samples 10000 \
#     --resume


import argparse
import asyncio
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from datasets import Dataset, DatasetDict, load_dataset

from transformers import AutoTokenizer

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate chat data with thinking via vLLM.")
    parser.add_argument("--input-dataset", type=str, required=True, help="HF dataset ID or a JSONL file path")
    parser.add_argument("--input-split", type=str, default="train", help="Dataset split (if HF dataset)")
    parser.add_argument(
        "--input-column",
        type=str,
        default=None,
        help="Column name containing chat messages (list[dict]). Auto-detected if omitted.",
    )
    parser.add_argument("--output-path", type=str, required=True, help="Directory to save the resulting split")
    parser.add_argument(
        "--server-urls",
        type=str,
        default="http://localhost:8000/v1",
        help="Comma-separated OpenAI-compatible base URLs for DDP-style scaling",
    )
    parser.add_argument(
        "--server-registry",
        type=str,
        default=None,
        help="Optional path to a text file containing one base URL per line. The file can change at runtime; servers will be added/removed dynamically.",
    )
    parser.add_argument(
        "--registry-refresh-interval",
        type=float,
        default=5.0,
        help="Seconds between reloads of --server-registry for dynamic scale-in/out.",
    )
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b", help="Model name served by vLLM")
    parser.add_argument("--target-total-tokens", type=int, default=4000, help="Approx target total tokens")
    parser.add_argument("--anchor-prompt-tokens", type=int, default=3000, help="Prompt length to target")
    parser.add_argument("--max-completion-tokens", type=int, default=1536, help="Upper bound for generation")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--max-concurrency", type=int, default=8, help="Concurrent requests to vLLM server")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--reasoning-level", type=str, default="high", help="Reasoning level in system prompt")
    parser.add_argument("--system-prompt", type=str, default=None, help="Custom system prompt to prepend")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP request timeout in seconds")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set, resume from partial outputs saved under --output-path. Writes one JSONL line per example for robustness.",
    )
    parser.add_argument(
        "--partial-jsonl",
        type=str,
        default=None,
        help="Override path for partial JSONL (defaults to <output-path>/data.jsonl)",
    )
    return parser.parse_args()


def load_input_dataset(args: argparse.Namespace) -> Dataset:
    if Path(args.input_dataset).exists():
        rows: List[Dict[str, Any]] = []
        with open(args.input_dataset, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
        return Dataset.from_list(rows)
    else:
        dset_dict: DatasetDict = load_dataset(args.input_dataset)
        if args.input_split not in dset_dict:
            raise ValueError(f"Split '{args.input_split}' not found in dataset. Available: {list(dset_dict.keys())}")
        return dset_dict[args.input_split]


def detect_chat_column(dataset: Dataset) -> str:
    if hasattr(dataset, "column_names"):
        for col in dataset.column_names:
            if len(dataset) == 0:
                continue
            sample = dataset[0].get(col)
            if isinstance(sample, list) and sample:
                first = sample[0]
                if isinstance(first, dict) and all(k in first for k in ["role", "content"]):
                    return col
    raise ValueError("Could not find a chat column (list[dict] with 'role' and 'content').")


def merge_assistant_thinking_inline(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("thinking"):
            thinking_text = msg.get("thinking") or ""
            content_text = msg.get("content") or ""
            merged_content = f"<think>{thinking_text}</think>\n{content_text}".strip()
            new_msg = {k: v for k, v in msg.items() if k != "thinking"}
            new_msg["content"] = merged_content
            merged.append(new_msg)
        else:
            merged.append(msg)
    return merged


def compute_prompt_tokens_for_assistant_index(
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, Any]],
    assistant_index: int,
) -> int:
    prompt_messages = messages[:assistant_index]
    rendered = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    tokenized = tokenizer(
        rendered,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return len(tokenized["input_ids"]) if isinstance(tokenized["input_ids"], list) else len(tokenized["input_ids"][0])


def count_prompt_tokens_with_system(
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, Any]],
) -> int:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    tokenized = tokenizer(
        rendered,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return len(tokenized["input_ids"]) if isinstance(tokenized["input_ids"], list) else len(tokenized["input_ids"][0])


def count_text_tokens(tokenizer: AutoTokenizer, text: Optional[str]) -> int:
    if not text:
        return 0
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    ids = enc["input_ids"]
    return len(ids) if isinstance(ids, list) else len(ids[0])


def summarize(values: List[int]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    vals = sorted(values)
    n = len(vals)

    def q(p: float) -> int:
        if n == 1:
            return vals[0]
        idx = int(round(p * (n - 1)))
        return vals[idx]

    return {
        "count": n,
        "min": vals[0],
        "p50": q(0.50),
        "p90": q(0.90),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": vals[-1],
        "mean": float(sum(vals) / n),
    }


def choose_assistant_turn_index(
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, Any]],
    anchor_prompt_tokens: int,
    target_total_tokens: int,
    min_completion_tokens: int = 16,
) -> Tuple[int, int]:
    assistant_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if not assistant_indices:
        raise ValueError("No assistant messages found in conversation.")

    best_idx = assistant_indices[0]
    best_prompt_len = compute_prompt_tokens_for_assistant_index(tokenizer, messages, best_idx)
    best_diff = abs(best_prompt_len - anchor_prompt_tokens)

    for idx in assistant_indices:
        prompt_len = compute_prompt_tokens_for_assistant_index(tokenizer, messages, idx)
        if prompt_len + min_completion_tokens >= target_total_tokens:
            continue
        diff = abs(prompt_len - anchor_prompt_tokens)
        if diff < best_diff:
            best_idx = idx
            best_prompt_len = prompt_len
            best_diff = diff
    return best_idx, best_prompt_len


def ensure_system_prompt(
    messages: List[Dict[str, Any]], reasoning_level: str, system_prompt_override: Optional[str]
) -> List[Dict[str, Any]]:
    sys_content = system_prompt_override or f"Reasoning: {reasoning_level}"
    if messages and messages[0].get("role") == "system":
        return [{"role": "system", "content": sys_content}] + messages[1:]
    return [{"role": "system", "content": sys_content}] + messages


def extract_thinking_and_content(message: Dict[str, Any]) -> Tuple[Optional[str], str]:
    thinking = message.get("thinking")
    content = message.get("content") or ""
    if thinking:
        return thinking, content
    rc = message.get("reasoning_content")
    if rc:
        return rc, content
    if "<think>" in content and "</think>" in content:
        before, after = content.split("</think>", 1)
        thinking_text = before.replace("<think>", "").strip()
        return thinking_text, after.strip()
    return None, content


async def generate_one(
    client: httpx.AsyncClient,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    # Sanitize messages to avoid non-JSON-serializable fields
    def _sanitize_messages(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Only pass role/content to the server to avoid 400s from unknown fields
        allowed_keys = {"role", "content"}
        cleaned: List[Dict[str, Any]] = []
        for m in msgs:
            nm: Dict[str, Any] = {}
            for k in allowed_keys:
                if k in m:
                    v = m[k]
                    if isinstance(v, (datetime, date)):
                        v = v.isoformat()
                    elif not isinstance(v, (str, type(None))):
                        v = str(v)
                    nm[k] = v if v is not None else ""
            nm.setdefault("role", "user")
            nm.setdefault("content", "")
            cleaned.append(nm)
        return cleaned

    safe_messages = _sanitize_messages(messages)
    payload = {
        "model": model,
        "messages": safe_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    resp = await client.post("/chat/completions", json=payload)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("choices"):
        raise RuntimeError("No choices returned from server")
    msg = data["choices"][0].get("message", {})
    thinking, content = extract_thinking_and_content(msg)
    return {"thinking": thinking, "content": content, "raw": data}


async def main_async(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dset = load_input_dataset(args)
    in_col = args.input_column or detect_chat_column(dset)
    log.info(f"Using input column: {in_col}")

    # Output/Resume setup
    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_jsonl_path = Path(args.partial_jsonl) if args.partial_jsonl else out_dir / "data.jsonl"
    progress_path = out_dir / "progress.json"

    processed_indices: set[int] = set()
    if args.resume and partial_jsonl_path.exists():
        try:
            with open(partial_jsonl_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and "index" in obj:
                            processed_indices.add(int(obj["index"]))
                    except Exception:
                        # Skip malformed lines; keep going
                        continue
            log.info(f"Resume enabled: found {len(processed_indices)} completed examples in {partial_jsonl_path}")
        except FileNotFoundError:
            pass

    # HTTP clients (support multiple servers; optionally dynamic registry)
    limits = httpx.Limits(max_keepalive_connections=args.max_concurrency, max_connections=args.max_concurrency)
    timeout = httpx.Timeout(args.timeout)

    def _clean_urls(urls: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for u in urls:
            s = u.strip()
            if not s:
                continue
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    static_urls = _clean_urls(args.server_urls.split(",")) if args.server_urls else []
    registry_file: Optional[Path] = Path(args.server_registry) if args.server_registry else None

    def _read_registry_urls() -> List[str]:
        if not registry_file or not registry_file.exists():
            return []
        try:
            with open(registry_file, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            return _clean_urls(lines)
        except Exception:
            return []

    current_urls: List[str] = _clean_urls(static_urls + _read_registry_urls())
    clients_by_url: Dict[str, httpx.AsyncClient] = {
        url: httpx.AsyncClient(base_url=url, timeout=timeout, limits=limits) for url in current_urls
    }
    clients_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.max_concurrency)

    async def _ensure_some_client_available():
        nonlocal current_urls
        while True:
            async with clients_lock:
                if len(current_urls) > 0:
                    return
            # Try to load from registry if available
            await asyncio.sleep(max(1.0, args.registry_refresh_interval))
            new_urls = _clean_urls(static_urls + _read_registry_urls())
            if new_urls:
                async with clients_lock:
                    for u in new_urls:
                        if u not in clients_by_url:
                            clients_by_url[u] = httpx.AsyncClient(base_url=u, timeout=timeout, limits=limits)
                    for u in list(clients_by_url.keys()):
                        if u not in new_urls:
                            await clients_by_url[u].aclose()
                            del clients_by_url[u]
                    current_urls = list(clients_by_url.keys())

    async def _refresh_registry_periodically():
        if not registry_file:
            return
        last_mtime: Optional[float] = None
        while True:
            try:
                if registry_file.exists():
                    mtime = registry_file.stat().st_mtime
                    if last_mtime is None or mtime > last_mtime:
                        last_mtime = mtime
                        new_urls = _clean_urls(static_urls + _read_registry_urls())
                        async with clients_lock:
                            # Add new clients
                            for u in new_urls:
                                if u not in clients_by_url:
                                    clients_by_url[u] = httpx.AsyncClient(base_url=u, timeout=timeout, limits=limits)
                            # Remove missing clients
                            for u in list(clients_by_url.keys()):
                                if u not in new_urls:
                                    try:
                                        await clients_by_url[u].aclose()
                                    finally:
                                        del clients_by_url[u]
                            # Update current_urls snapshot
                            nonlocal current_urls
                            current_urls = list(clients_by_url.keys())
                await asyncio.sleep(max(1.0, args.registry_refresh_interval))
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(max(1.0, args.registry_refresh_interval))

    async def _pick_client_for_index(idx: int) -> httpx.AsyncClient:
        # Wait until at least one client is available
        await _ensure_some_client_available()
        async with clients_lock:
            urls = list(current_urls)
            if not urls:
                # Fallback shouldn't happen due to ensure above
                raise RuntimeError("No available servers to route request")
            url = urls[idx % len(urls)]
            return clients_by_url[url]

    # Start registry refresher
    refresh_task = asyncio.create_task(_refresh_registry_periodically())

    results: List[Dict[str, Any]] = []
    model_supports_thinking = "gpt-oss" in args.model

    orig_prompt_tokens_stats: List[int] = []
    with_sys_prompt_tokens_stats: List[int] = []
    thinking_tokens_stats: List[int] = []
    content_tokens_stats: List[int] = []

    write_lock = asyncio.Lock()

    async def _append_jsonl(obj: Dict[str, Any]) -> None:
        # Append one line to partial JSONL with flush and fsync for robustness
        import os
        line = json.dumps(obj, ensure_ascii=False)
        async with write_lock:
            with open(partial_jsonl_path, "a") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())

    async def process_row(row_idx: int, row: Dict[str, Any]) -> None:
        try:
            if row_idx in processed_indices:
                return
            messages: List[Dict[str, Any]] = row[in_col]
            if not model_supports_thinking:
                messages = merge_assistant_thinking_inline(messages)

            target_idx, prompt_tokens = choose_assistant_turn_index(
                tokenizer=tokenizer,
                messages=messages,
                anchor_prompt_tokens=args.anchor_prompt_tokens,
                target_total_tokens=args.target_total_tokens,
            )

            prompt_messages = messages[:target_idx]
            prompt_messages = ensure_system_prompt(prompt_messages, args.reasoning_level, args.system_prompt)
            with_sys_len = count_prompt_tokens_with_system(tokenizer, prompt_messages)

            # Back off to earlier assistant turns if no generation budget remains
            remaining_tokens = args.target_total_tokens - with_sys_len
            if remaining_tokens < 1:
                backoff_attempts = 0
                while remaining_tokens < 1 and backoff_attempts < 8:
                    prev_idx = None
                    for i2 in range(target_idx - 1, -1, -1):
                        if messages[i2].get("role") == "assistant":
                            prev_idx = i2
                            break
                    if prev_idx is None:
                        break
                    target_idx = prev_idx
                    prompt_messages = messages[:target_idx]
                    prompt_messages = ensure_system_prompt(prompt_messages, args.reasoning_level, args.system_prompt)
                    with_sys_len = count_prompt_tokens_with_system(tokenizer, prompt_messages)
                    remaining_tokens = args.target_total_tokens - with_sys_len
                    backoff_attempts += 1

            if remaining_tokens < 1:
                log.warning(f"Row {row_idx}: prompt exceeds target_total_tokens; skipping")
                return

            max_tokens = max(1, min(args.max_completion_tokens, remaining_tokens))

            # Stats
            orig_len_final = compute_prompt_tokens_for_assistant_index(tokenizer, messages, target_idx)
            orig_prompt_tokens_stats.append(orig_len_final)
            with_sys_prompt_tokens_stats.append(with_sys_len)

            async def _try_generate() -> Dict[str, Any]:
                attempts = max(3, len(current_urls) * 2 or 3)
                last_err: Optional[Exception] = None
                for a in range(attempts):
                    try:
                        client = await _pick_client_for_index(row_idx + a)
                        async with semaphore:
                            return await generate_one(
                                client=client,
                                model=args.model,
                                messages=prompt_messages,
                                max_tokens=max_tokens,
                                temperature=args.temperature,
                                top_p=args.top_p,
                            )
                    except Exception as e:
                        last_err = e
                        await asyncio.sleep(0.5)
                        continue
                if last_err:
                    raise last_err
                raise RuntimeError("Generation failed with unknown error")

            gen = await _try_generate()

            assistant_msg = {"role": "assistant", "content": gen["content"]}
            if gen["thinking"]:
                assistant_msg["thinking"] = gen["thinking"]
            thinking_tokens_stats.append(count_text_tokens(tokenizer, gen["thinking"]))
            content_tokens_stats.append(count_text_tokens(tokenizer, gen["content"]))

            final_messages = messages[:target_idx] + [assistant_msg]

            # Stream to JSONL immediately for pause/resume
            await _append_jsonl({"index": row_idx, "messages": final_messages})
            results.append({"messages": final_messages})

            if len(results) == 1:
                log.info("First server output (raw):\n" + json.dumps(gen["raw"], indent=2))
                log.info(
                    "First finalized example (messages):\n" + json.dumps(final_messages, indent=2, ensure_ascii=False)
                )
        except httpx.HTTPStatusError as e:
            try:
                err_text = e.response.text
            except Exception:
                err_text = str(e)
            log.error(f"Row {row_idx} HTTP {e.response.status_code if hasattr(e, 'response') else '??'}: {err_text}")
        except Exception as e:
            log.error(f"Row {row_idx} failed: {e}")

    total = len(dset) if args.max_samples is None else min(args.max_samples, len(dset))
    # Submit work in streaming batches to avoid creating millions of tasks
    batch_submit = max(args.max_concurrency * 4, 64)
    for start in range(0, total, batch_submit):
        print(f"{start}/{total} completed")
        end = min(start + batch_submit, total)
        tasks: List[asyncio.Task] = []
        for i in range(start, end):
            row = dset[i]
            tasks.append(asyncio.create_task(process_row(i, row)))
        # Process in waves limited by max_concurrency
        for j in range(0, len(tasks), args.max_concurrency):
            await asyncio.gather(*tasks[j : j + args.max_concurrency])

    # Close clients and refresher
    try:
        refresh_task.cancel()
    except Exception:
        pass
    async with clients_lock:
        await asyncio.gather(*[c.aclose() for c in clients_by_url.values()])

    # Finalize: write HF dataset snapshot to output directory
    # Build from the partial JSONL (includes previously completed examples if resume)
    # Only include examples within the requested total range
    jsonl_rows: List[Dict[str, Any]] = []
    try:
        with open(partial_jsonl_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    idx = int(obj.get("index", -1))
                    if idx < 0:
                        continue
                    if idx >= len(dset):
                        continue
                    if args.max_samples is not None and idx >= args.max_samples:
                        continue
                    jsonl_rows.append({"messages": obj.get("messages", [])})
                except Exception:
                    continue
    except FileNotFoundError:
        pass

    if jsonl_rows:
        out_dset = Dataset.from_list(jsonl_rows)
        out_dset.save_to_disk(str(out_dir))
        log.info(f"Saved {len(out_dset)} records to {out_dir}")
    else:
        log.info("No records to save (partial JSONL empty)")

    stats = {
        "original_prompt_tokens": summarize(orig_prompt_tokens_stats),
        "prompt_tokens_with_system": summarize(with_sys_prompt_tokens_stats),
        "thinking_tokens": summarize(thinking_tokens_stats),
        "content_tokens": summarize(content_tokens_stats),
    }
    # Persist progress snapshot
    try:
        with open(progress_path, "w") as f:
            json.dump({"stats": stats, "completed": len(jsonl_rows)}, f, indent=2)
    except Exception:
        pass
    log.info("Token length stats (tokens):\n" + json.dumps(stats, indent=2))


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
