from __future__ import annotations
import asyncio
import dataclasses
from pathlib import Path
from simple_parsing import ArgumentParser
import matplotlib.pyplot as plt

from safetytooling.utils.experiment_utils import ExperimentConfigBase
from safetytooling.apis.batch_api import BatchInferenceAPI
from safetytooling.data_models import Prompt, ChatMessage, MessageRole
from tqdm.asyncio import tqdm_asyncio
@dataclasses.dataclass(kw_only=True)
class ExperimentConfig(ExperimentConfigBase):
    exp_name: str
    models: list[str]
    texts: list[str] = dataclasses.field(default_factory=lambda: ["The quick brown fox jumps over the lazy dog."])
    n_repeats: int = 1  # number of repetitions per text
    use_batch: bool = True  # whether to use BatchInferenceAPI or single InferenceAPI

    def get_exp_subdir(self) -> Path:
        return self.output_dir / self.exp_name

def rot_n(s: str, n: int) -> str:
    def shift(c: str) -> str:
        if 'a' <= c <= 'z': return chr((ord(c) - ord('a') + n) % 26 + ord('a'))
        if 'A' <= c <= 'Z': return chr((ord(c) - ord('A') + n) % 26 + ord('A'))
        return c
    return ''.join(shift(c) for c in s)

async def main(cfg: ExperimentConfig):
    exp_dir = cfg.get_exp_subdir()
    exp_dir.mkdir(parents=True, exist_ok=True)
    raw_records: list[dict] = []
    summary: dict[str, list[dict]] = {m: [] for m in cfg.models}
    # summarize planned prompts and API calls
    num_texts = len(cfg.texts)
    num_models = len(cfg.models)
    num_ns = 26-1
    num_prompts_per_model = num_texts * cfg.n_repeats * num_ns
    num_requests_per_model = num_prompts_per_model if not cfg.use_batch else num_ns
    total_requests = num_requests_per_model * num_models
    print(f"Running ROT-N eval: models={cfg.models}")
    print(f"Texts={num_texts}, repeats={cfg.n_repeats}, use_batch={cfg.use_batch}")
    print(f"Total prompts: {num_prompts_per_model * num_models}")
    print(f"Total API requests: {total_requests}")
    # init batch API if needed
    batch_api = BatchInferenceAPI(use_redis=cfg.use_redis, no_cache=not cfg.enable_cache) if cfg.use_batch else None
    for model in cfg.models:
        # Build all prompts for all n values for this model
        all_prompts = []
        all_expected = []
        n_indices = []  # Track which n each prompt belongs to
        
        for n in range(1,num_ns):
            for _ in range(cfg.n_repeats):
                for text in cfg.texts:
                    all_prompts.append(
                        Prompt(messages=[ChatMessage(content=f"Apply a ROT{n} cipher to the following text and output only the converted text: \"{text}\"", role=MessageRole.user)])
                    )
                    all_expected.append(rot_n(text, n))
                    n_indices.append(n)
        
        # Get all responses for this model in a single batch
        if cfg.use_batch:
            responses_list, _ = await batch_api(
                model_id=model,
                prompts=all_prompts,
                log_dir=cfg.prompt_history_dir,
                use_cache=False,
                max_tokens=max(len(t) for t in cfg.texts) + 10,
            )
        else:
            tasks = [
                cfg.api(
                    model_id=model,
                    prompt=prompt,
                    print_prompt_and_response=False,
                    max_tokens=len(expected) + 10,
                    temperature=0,
                    use_cache=False,
                )
                for prompt, expected in zip(all_prompts, all_expected)
            ]
            results = await tqdm_asyncio.gather(*tasks, desc=f"Model: {model}")
            responses_list = [res[0] for res in results]
        
        # Process results
        for n_val, response, expected in zip(n_indices, responses_list, all_expected):
            out = response.completion.strip()
            matched = sum(1 for a, b in zip(out, expected) if a == b)
            char_acc = matched / len(expected) if expected else 0.0
            exact = 1 if out == expected else 0
            raw_records.append({"model": model, "text": expected, "n": n_val, "output": out, "expected": expected, "exact": exact, "char_acc": char_acc})
        
        # Compute summary statistics for this model
        for n in range(1,26):
            records = [r for r in raw_records if r["model"] == model and r["n"] == n]
            exact_sum = sum(r["exact"] for r in records)
            char_acc_sum = sum(r["char_acc"] for r in records)
            k = len(records)
            summary[model].append({"n": n, "avg_exact": exact_sum / k, "avg_char_acc": char_acc_sum / k})

    import jsonlines
    with jsonlines.open(exp_dir / "rot_raw_results.jsonl", "w") as writer:
        for rec in raw_records:
            writer.write(rec)
    with jsonlines.open(exp_dir / "rot_summary.jsonl", "w") as writer:
        for model, recs in summary.items():
            for r in recs:
                writer.write({"model": model, **r})

    plt.figure()
    for model in cfg.models:
        xs = [r["n"] for r in summary[model]]
        ys = [r["avg_exact"] for r in summary[model]]
        plt.plot(xs, ys, marker="o", label=model)
    plt.title("Avg exact match accuracy vs ROT N")
    plt.xlabel("N")
    plt.ylabel("Avg exact match")
    plt.xticks(range(26))
    plt.ylim(-0.01, 1.01)
    plt.legend()
    plt.grid(True)
    plt.savefig(exp_dir / "rot_exact_accuracy.png")
    plt.close()
    plt.figure()
    for model in cfg.models:
        xs = [r["n"] for r in summary[model]]
        ys = [r["avg_char_acc"] for r in summary[model]]
        plt.plot(xs, ys, marker="o", label=model)
    plt.title("Avg char-level accuracy vs ROT N")
    plt.xlabel("N")
    plt.ylabel("Avg char-level accuracy")
    plt.xticks(range(26))
    plt.ylim(-0.01, 1.01)
    plt.legend()
    plt.grid(True)
    plt.savefig(exp_dir / "rot_char_accuracy.png")
    plt.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="config")
    args = parser.parse_args()
    cfg: ExperimentConfig = args.config

    cfg.setup_experiment(log_file_prefix=cfg.exp_name)
    asyncio.run(main(cfg)) 