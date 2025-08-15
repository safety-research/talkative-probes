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

import argparse
import asyncio
import json
import logging
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
    payload = {
        "model": model,
        "messages": messages,
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

    # HTTP clients (support multiple servers)
    limits = httpx.Limits(max_keepalive_connections=args.max_concurrency, max_connections=args.max_concurrency)
    timeout = httpx.Timeout(args.timeout)
    base_urls = [u.strip() for u in args.server_urls.split(",") if u.strip()]
    clients: List[httpx.AsyncClient] = [
        httpx.AsyncClient(base_url=base, timeout=timeout, limits=limits) for base in base_urls
    ]
    semaphore = asyncio.Semaphore(args.max_concurrency)

    results: List[Dict[str, Any]] = []
    model_supports_thinking = "gpt-oss" in args.model

    orig_prompt_tokens_stats: List[int] = []
    with_sys_prompt_tokens_stats: List[int] = []
    thinking_tokens_stats: List[int] = []
    content_tokens_stats: List[int] = []

    async def process_row(row_idx: int, row: Dict[str, Any]) -> None:
        messages: List[Dict[str, Any]] = row[in_col]
        if not model_supports_thinking:
            messages = merge_assistant_thinking_inline(messages)

        target_idx, prompt_tokens = choose_assistant_turn_index(
            tokenizer=tokenizer,
            messages=messages,
            anchor_prompt_tokens=args.anchor_prompt_tokens,
            target_total_tokens=args.target_total_tokens,
        )

        max_tokens = args.target_total_tokens - prompt_tokens
        max_tokens = max(16, min(args.max_completion_tokens, max_tokens))

        prompt_messages = messages[:target_idx]
        orig_prompt_tokens_stats.append(prompt_tokens)
        prompt_messages = ensure_system_prompt(prompt_messages, args.reasoning_level, args.system_prompt)
        with_sys_prompt_tokens_stats.append(count_prompt_tokens_with_system(tokenizer, prompt_messages))

        async with semaphore:
            client = clients[row_idx % len(clients)]
            gen = await generate_one(
                client=client,
                model=args.model,
                messages=prompt_messages,
                max_tokens=max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

        assistant_msg = {"role": "assistant", "content": gen["content"]}
        if gen["thinking"]:
            assistant_msg["thinking"] = gen["thinking"]
        thinking_tokens_stats.append(count_text_tokens(tokenizer, gen["thinking"]))
        content_tokens_stats.append(count_text_tokens(tokenizer, gen["content"]))

        final_messages = messages[:target_idx] + [assistant_msg]

        results.append({"messages": final_messages})

        if len(results) == 1:
            log.info("First server output (raw):\n" + json.dumps(gen["raw"], indent=2))
            log.info("First finalized example (messages):\n" + json.dumps(final_messages, indent=2, ensure_ascii=False))

    tasks: List[asyncio.Task] = []
    total = len(dset) if args.max_samples is None else min(args.max_samples, len(dset))
    for i in range(total):
        row = dset[i]
        tasks.append(asyncio.create_task(process_row(i, row)))

    for i in range(0, len(tasks), args.max_concurrency):
        batch = tasks[i : i + args.max_concurrency]
        await asyncio.gather(*batch)

    # Close clients
    await asyncio.gather(*[c.aclose() for c in clients])

    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dset = Dataset.from_list(results)
    out_dset.save_to_disk(str(out_dir))
    log.info(f"Saved {len(out_dset)} records to {out_dir}")

    stats = {
        "original_prompt_tokens": summarize(orig_prompt_tokens_stats),
        "prompt_tokens_with_system": summarize(with_sys_prompt_tokens_stats),
        "thinking_tokens": summarize(thinking_tokens_stats),
        "content_tokens": summarize(content_tokens_stats),
    }
    log.info("Token length stats (tokens):\n" + json.dumps(stats, indent=2))


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
