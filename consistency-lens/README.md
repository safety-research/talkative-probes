# Consistency Lenses: An Architectural Plan for Scalable LLM Interpretation

This document outlines the architectural plan for training Consistency Lenses, a method to interpret the internal states of Large Language Models (LLMs) by forcing them through a textual, human-readable bottleneck. The plan is designed for an 8 x H100 GPU environment, emphasizing scalability and efficiency.

## Project Structure

The project will be organized as follows:

```
consistency-lens/
├── README.md                  # one-pager: what, why, how to run
├── env/
│   ├── Dockerfile             # FROM nvcr.io/nvidia/pytorch:24.04-py3 + deps
│   └── requirements.txt       # transformers==0.20, deepspeed, flash-attn, …
├── config/                    # everything declarative lives here
│   ├── ds_stage2.yaml         # DeepSpeed & FSDP tuning knobs
│   ├── lens.yaml              # α, τ schedule, layer L, token-pos P, etc.
│   └── wandb.yaml             # project/name tags
├── data/
│   ├── tokenizer/             # cached HF tokenizer files
│   ├── corpus/                # mmap shards of raw token IDs
│   └── activations/           # *.pt files with (A, A′, input_ids) triples
├── scripts/
│   ├── 00_dump_activations.py # single-pass frozen-model extractor
│   ├── 01_train.py            # main DeepSpeed launcher
│   ├── 02_eval.py             # runs qualitative + metrics suite on checkpoints
│   └── slurm.sh               # sbatch wrapper
├── lens/                      # Python package
│   ├── __init__.py
│   ├── models/
│   │   ├── orig.py            # thin HuggingFace wrapper w/ “replace-activation” hook
│   │   ├── decoder.py         # D: prompt-prep + Gumbel-Softmax + STE
│   │   ├── encoder.py         # E: final-token slice + projector
│   │   └── utils.py           # share dropout masks, weight init helpers
│   ├── data/
│   │   ├── dataset.py         # PyTorch Dataset that streams triples from mmap
│   │   └── collate.py         # pads & moves to GPU
│   ├── training/
│   │   ├── loop.py            # step() with both loss terms, tau & α schedulers
│   │   ├── optim.py           # fused AdamW + parameter groups
│   │   └── distributed.py     # all-reduce hygiene, set_seed for each rank
│   ├── evaluation/
│   │   ├── metrics.py         # KL, cosine, ppl
│   │   └── dump_text.py       # sample fixed activation set ↦ text, push to W&B
│   └── utils/
│       ├── logging.py         # wandb & rich progress bar
│       └── checkpoint.py      # save/load, EMA, resume safety belt
└── tests/
    ├── test_dataset.py
    ├── test_swap_hook.py
    └── smoke_train.py         # 1-step run on cpu to catch import bugs
```

## Getting Started

Create a fresh environment with [uv](https://github.com/astral-sh/uv) and install
the package in editable mode:

```bash
cd consistency-lens
uv venv
uv pip install -e .
```

Extra dependencies for the Docker image can be installed with:

```bash
uv pip install -r env/requirements.txt
```

## Leaning on the Ecosystem

A core principle is to leverage existing, well-tested libraries for heavy lifting, allowing our codebase to focus on the novel aspects of Consistency Lenses.

| Section                       | Library                               | How we lean on it                                                                                                             |
|-------------------------------|---------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| Tokenizer & raw text I/O      | HuggingFace `tokenizers` + `datasets` | Pre-tokenise once (`datasets.map(batch_encode_plus)`), memory-map to skip bespoke dataloader code.                            |
| Frozen model                  | `transformers.AutoModelForCausalLM`   | Out-of-the-box forward; we only patch a single hook to overwrite one hidden slice (same trick as prompt-tuning).              |
| Parameter off-loading & sharding | DeepSpeed ZeRO-1 + FSDP (torch 2.3)  | No hand-rolled DistributedDataParallel; we declare shards in `config/ds_stage2.yaml`.                                       |
| 8-bit weights                 | `bitsandbytes`                        | `load_in_8bit=True` on the frozen model gets us 4× memory cut for free.                                                      |
| Attention speedup             | `flash-attn` 3                         | `model._apply_flash_attention_2()` after load—zero code rewrite. (Note: `flash-attn` 3 uses `_apply_flash_attention_2`)     |
| Gumbel-Softmax + STE          | `torch.nn.functional.gumbel_softmax`  | Provided primitive already supports temperature; STE is two lines (`soft.detach() + hard - hard.detach()`).                   |
| Training loop plumbing        | `torch.compile` (nvFuser)             | Wrap decoder/encoder forward; eliminates manual kernel fusion.                                                               |
| Optimiser                     | DeepSpeed fused AdamW                 | One config flag; avoids building custom fused kernels.                                                                       |
| Logging & dashboards          | Weights & Biases (`wandb`)            | Push metrics + the 128-sample text grid via `wandb.Table`, no HTML hacking. Configured via `config/wandb.yaml`.              |
| Checkpoint format             | DeepSpeed checkpointing               | Handles partition re-stitching; `lens.utils.checkpoint` decorates with EMA tensors and resume logic.                          |
| Scheduling (Slurm)            | Existing cluster                      | Five-line `scripts/slurm.sh`; no custom orchestration.                                                                        |
| Tests                         | `pytest`                              | Catch regressions; rely on HF’s mocked models to keep CI light (`tests/smoke_train.py`, `tests/test_swap_hook.py`, etc.). |

**Bottom line:** Every heavy lift—tokenisation, flash attention, distributed memory ops, fused optimisers, checkpoints—is already solved in the ecosystem. Our codebase, primarily within the `lens/` package, is the slim glue that:
1.  Inserts the projection layers (`lens/models/decoder.py`, `lens/models/encoder.py`).
2.  Runs the Gumbel-Softmax loop (`lens/models/decoder.py`, `lens/training/loop.py`).
3.  Wires up the composite loss (`lens/training/loop.py`).
4.  Streams cached activations (`lens/data/dataset.py`).

Nothing else gets bespoke unless empirical pain forces it.

---

## 0. Environment

The choice of environment components is critical for performance and reproducibility. This will be managed via `env/Dockerfile` and `env/requirements.txt`.

| Piece        | Choice                                       | Reason                                                                 |
|--------------|----------------------------------------------|------------------------------------------------------------------------|
| Framework    | PyTorch 2.3+ (`torch.compile`)               | Most mature hooks for customization, integrates well with FSDP, benefits from nvFuser on H100. |
| Distributed  | DeepSpeed + FSDP                             | FSDP for efficient sharding of two trainable LLaMA-sized models (Decoder & Encoder); DeepSpeed's ZeRO-1 for CPU-offloading the frozen original LLM. Configured in `config/ds_stage2.yaml`. |
| Precision    | BF16 (Brain Floating Point)                  | Native support on Hopper architecture (H100) for optimal performance; simplifies training by avoiding mixed-precision complexities. |
| Extras       | `bitsandbytes` 0.42, `flash-attn` 3, `transformers` 0.20+, `wandb` | `bitsandbytes` for 8-bit quantization of the frozen model, `flash-attn` for optimized attention, `transformers` for base models (tokenizer in `data/tokenizer/`), `wandb` for logging (config in `config/wandb.yaml`). All specified in `env/requirements.txt`. |
| Container    | `nvcr.io/nvidia/pytorch:24.04-py3`           | Provides a clean, tested, and reproducible environment (base for `env/Dockerfile`). `apt-get clang-17` may be needed for specific C++ extensions if any. |

---


## 1. Data & Activation Cache
Decoder D: As a base model, it's a strong general-purpose text generator. The prompt engineering (fixed prefix + projected activation) will guide it to generate the textual explanation.
Encoder E: As a base model, it's a strong general-purpose text encoder, suitable for taking the generated text and trying to reconstruct the original activation's semantic content.
Recommendation:

Use a base/foundation model for LLM_orig (e.g., meta-llama/Llama-2-7b-hf rather than meta-llama/Llama-2-7b-chat-hf). This will provide activations that are more representative of general language understanding.
Consequently, D and E will also be initialized from this base model.
Efficient data handling and pre-computation of activations are key. Managed by `scripts/00_dump_activations.py` and consumed by `lens.data.dataset`.

1.  **Corpus Selection:**
    *   Choose a substantial corpus, approximately 10 billion tokens (e.g., The Pile or a suitable internal dataset).

2.  **Pre-tokenization (using Hugging Face `tokenizers` and `datasets`):**
    *   Tokenize the entire corpus using the tokenizer corresponding to the `LLM_orig`. Tokenizer files cached in `data/tokenizer/`.
    *   Save the tokenized data as memory-mapped (mmap) Hugging Face `Dataset` shards in `data/corpus/`.

3.  **Activation Dump Job (`scripts/00_dump_activations.py`):**
    *   One-off pre-processing step.
    *   **Process:**
        *   Load `LLM_orig` (from `transformers`) using `bitsandbytes` 8-bit quantization (`load_in_8bit=True`).
        *   Perform forward pass with `torch.no_grad()` and `output_hidden_states=True`.
        *   Extract target activation `A` from layer `L` and token position `P` (specified in `config/lens.yaml`). Store as `float16`.
        *   Store minimal `input_ids` required to recompute logits from `LLM_orig`.
        *   Randomly select `A_prime` (and its `input_ids`).
    *   **Output:** Dataset of `(input_ids_for_A, A, input_ids_for_A_prime, A_prime)` tuples as `*.pt` files in `data/activations/`. Consumed by `lens.data.dataset.ActivationDataset`.

Token Position P: For each training sequence, randomly sample P from a valid range (e.g., 10 <= P < sequence_length - 1). This ensures diverse contextual states.
The input_ids_for_A will be the tokens 0...P from that sequence.
The logits for loss_KL will be computed for predicting token P+1.
Layer Depth L: Start with a single, fixed layer in the mid-to-late-mid range of LLM_orig (e.g., for a 32-layer model, try L15, L20, or L24). Make this easily configurable.
Why this combination?
This approach aims to train a general-purpose lens. By sampling activations from random (valid) positions across a diverse corpus and from a semantically rich layer, the system is encouraged to learn how to explain a wide array of model states. The hypothesis is that the model learns general principles of how activations map to textual explanations, rather than specializing in particular sentence structures or conceptual points.

Regarding A_prime and input_ids_for_A_prime:
When sampling A_prime, you also sample its corresponding input_ids_for_A_prime. The crucial part for the KL loss calculation is that logits_M_A and logits_M_A_target are both conditioned on batch.input_ids_A. This means the delta_residual (derived from A_prime's reconstruction) is used to perturb A_reconstructed within the original context of A. This setup tests if the information lost/gained during the autoencoding of A_prime is meaningfully transferable to improve the functional behavior of A in its original context.
---

## 2. Models

Core components reside in `lens.models/`.

| Component  | Source & Implementation Details                                                                                                                                                                                                                                                                                                                                                           |
|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `LLM_orig` | `transformers.AutoModelForCausalLM`, wrapped in `lens.models.orig.OriginalModelWrapper`. Loaded in 8-bit and frozen. Custom `forward_with_replacement` hook allows swapping hidden state.                                                                                                                                                                                                |
| Decoder `D`| Clone of `LLM_orig`, implemented in `lens.models.decoder.DecoderModel`. Prepends prompt tokens, incorporates `Proj_A_to_D_emb`. Output layer uses `torch.nn.functional.gumbel_softmax` + STE for differentiable token sampling. Trainable.                                                                                                                                               |
| Encoder `E`| Another clone of `LLM_orig`, implemented in `lens.models.encoder.EncoderModel`. Processes discrete text from `D`. Extracts final token's hidden state, passes through `Proj_E_hidden_to_A` to reconstruct `A_reconstructed`. Trainable.                                                                                                                                                   |

**Weight Sharing & Initialization:**
*   Initial weights for `D` and `E` copied from `LLM_orig`. Helper functions in `lens.models.utils`.
*   Projection layers initialized separately. `D` and `E` weights updated independently.

**Memory Considerations:** Remain as previously outlined, feasible with DeepSpeed/FSDP.

---

## 3. Distributed Layout

Managed by DeepSpeed as configured in `config/ds_stage2.yaml`. `lens.training.distributed` handles rank-specific setup.

*   **Process Topology:** 8 ranks (one per GPU) on a single node.
*   **Parameter Placement & Sharding:**
    *   **Frozen `LLM_orig`:** DeepSpeed ZeRO-1 with CPU offload.
    *   **Trainable `D` & `E`:** Combined FSDP group (via DeepSpeed integration) sharding parameters, gradients, optimizer states.
    *   **Gradient Checkpointing:** Applied to `D` and `E` transformer blocks.

*   **Batching Strategy:**
    *   Micro-batch size: 1 (per GPU). Collation handled by `lens.data.collate.CollateFn`.
    *   Gradient Accumulation Steps: e.g., 16 (configured in `config/ds_stage2.yaml` or script args).
    *   Effective Batch Size: `1 * 8 * 16 = 128` sequences.

---

## 4. Training Loop Skeleton

The core logic is in `lens.training.loop.training_step`. `scripts/01_train.py` is the entry point.

```python
# Pseudocode reflecting components from lens/ package
# (Simplified, actual implementation in lens.training.loop)

# D_model: lens.models.decoder.DecoderModel
# E_model: lens.models.encoder.EncoderModel
# LLM_orig_model: lens.models.orig.OriginalModelWrapper

for step, batch in enumerate(loader): # loader uses lens.data.dataset.ActivationDataset
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        # --- Primary Path ---
        # generate_soft logic within D_model
        generated_output_D_A = D_model.generate_soft(
            activation_input=batch.A,
            max_length=config.T_text,
            gumbel_tau=gumbel_temperature_schedule(step, config.lens_yaml) # from config/lens.yaml
        )
        A_reconstructed = E_model(generated_output_D_A.generated_text_embeddings)

        # --- Auxiliary Path ---
        generated_output_D_A_prime = D_model.generate_soft(
            activation_input=batch.A_prime, # ...
        )
        A_prime_reconstructed = E_model(generated_output_D_A_prime.generated_text_embeddings)

        with torch.no_grad():
            delta_residual = batch.A_prime - A_prime_reconstructed

        A_target_for_KL = A_reconstructed + delta_residual

        # --- Logit Calculation using LLM_orig_model ---
        logits_M_A = LLM_orig_model.forward_with_replacement(
            input_ids=batch.input_ids_A,
            new_activation=batch.A, # ...
        ).logits[:, config.TARGET_TOKEN_POS, :]

        logits_M_A_target = LLM_orig_model.forward_with_replacement(
            input_ids=batch.input_ids_A,
            new_activation=A_target_for_KL, # ...
        ).logits[:, config.TARGET_TOKEN_POS, :]

        # --- Loss Calculation ---
        loss_lm = cross_entropy_loss_fn(...) # uses generated_output_D_A
        loss_kl = kl_divergence_loss_fn(
            torch.log_softmax(logits_M_A_target, dim=-1),
            torch.softmax(logits_M_A, dim=-1)
        )

        current_alpha = alpha_schedule(step, config.lens_yaml) # from config/lens.yaml
        loss = loss_lm + current_alpha * loss_kl

    # Optimization (DeepSpeed engine handles backward/step)
    # optimizer setup in lens.training.optim (AdamW, param groups)
    engine.backward(loss)
    engine.step()

    # Schedules for LR, tau, alpha from config/lens.yaml via lens.training.loop
    # Logging via lens.utils.logging (to wandb)
```

**Key Details:**
*   **`D.generate_soft`:** Implemented in `lens.models.decoder.DecoderModel`.
*   **Schedules (`gumbel_temperature_schedule`, `alpha_schedule`):** Defined in `config/lens.yaml`, applied in `lens.training.loop`.
*   **Optimizer (`lens.training.optim`):** Fused AdamW from DeepSpeed, differential LRs for projections vs. transformer blocks.
*   `torch.compile` wraps the forward methods of `D` and `E` for performance.

---

## 5. Evaluation

Handled by `scripts/02_eval.py` and `lens.evaluation/`.

*   **Quantitative Metrics (`lens.evaluation.metrics`):**
    *   `loss_lm`, `loss_kl`, `total_loss`, `cosine_similarity(A, A_reconstructed)`, perplexity. Logged to W&B via `lens.utils.logging`.
*   **Qualitative Spot-Checks (`lens.evaluation.dump_text`):**
    *   Periodically run `D` on fixed activations, log generated text to W&B table.
*   **Early Stopping Criteria:** Monitored based on W&B metrics.

---

## 6. Check-pointing & Resilience

Managed by `lens.utils.checkpoint` leveraging DeepSpeed's capabilities.

*   **Checkpointing Strategy:** Full DeepSpeed checkpoints.
*   **EMA for Evaluation:** Maintained by `lens.utils.checkpoint.EMA` for `D` and `E` weights.
*   **State Resumption:** `lens.utils.checkpoint` ensures all necessary states (step, schedules, optimizer) are saved/loaded.

---

## 7. Job Launcher (Slurm Example)

Using `scripts/slurm.sh`.

```bash
#!/bin/bash
#SBATCH --job-name=consistency_lens
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --constraint=h100
#SBATCH --output=logs/consistency_lens_%j.out # Consider putting logs in a dedicated dir
#SBATCH --error=logs/consistency_lens_%j.err

# Activate environment (conda/venv/module load)
# cd /path/to/consistency-lens # Ensure PWD is project root

# Path to your training script and config
TRAIN_SCRIPT="scripts/01_train.py"
DEEPSPEED_CONFIG="config/ds_stage2.yaml" # Updated to YAML
LENS_CONFIG="config/lens.yaml"
WANDB_CONFIG="config/wandb.yaml"

deepspeed --num_gpus 8 ${TRAIN_SCRIPT} \
          --deepspeed_config ${DEEPSPEED_CONFIG} \
          --lens_config ${LENS_CONFIG} \
          --wandb_config ${WANDB_CONFIG} \
          --model_name_or_path "meta-llama/Llama-2-7b-hf" \
          --output_dir "outputs/run_XYZ" \
          # activation_cache_path now implicitly data/activations/
          --num_train_epochs 3 \
          # per_device_train_batch_size, grad_accum likely in ds_stage2.yaml or train.py defaults
          # learning_rates likely in lens.yaml or train.py defaults
```

**`config/ds_stage2.yaml` (Example DeepSpeed Configuration - structure):**
(Content similar to the JSON, but in YAML format, defining optimizer, scheduler, bf16, zero_optimization stage 2, etc. `train_batch_size`, `train_micro_batch_size_per_gpu`, `gradient_accumulation_steps` would be set here or inferred.)

---

## 8. Post-MVP Enhancements & Future Directions

(This section remains largely the same as the original, focusing on future research avenues rather than specific code paths.)

*   **Optimize `LLM_orig`:** Transition frozen `LLM_orig` to FP8.
*   **Parameter-Efficient Fine-Tuning (PEFT):** Investigate LoRA for `D` and `E`.
*   **Multi-Layer and Multi-Position Interpretation.**
*   **Advanced Text Generation Strategies.**
*   **Dictionary Learning on Bottleneck.**

This revised plan aligns the architecture with the proposed practical directory structure and explicitly states the reliance on existing libraries, making the unique contributions of the `lens/` package very clear.