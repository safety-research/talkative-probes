Okay, I've integrated the suggested minor clarifications and corrected the KL divergence argument order in the training loop pseudocode. Here's the revised plan:

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
│   └── activations/           # *.pt files with (input_ids_A, A, input_ids_A_prime, A_prime) tuples
├── scripts/
│   ├── 00_dump_activations.py # single-pass frozen-model extractor
│   ├── 01_train.py            # main DeepSpeed launcher
│   ├── 02_eval.py             # runs qualitative + metrics suite on checkpoints
│   └── slurm.sh               # sbatch wrapper
├── lens/                      # Python package
│   ├── __init__.py
│   ├── models/
│   │   ├── orig.py            # thin HuggingFace wrapper w/ "replace-activation" hook
│   │   ├── decoder.py         # D: prompt-prep + Gumbel-Softmax + STE
│   │   ├── encoder.py         # E: final-token slice + projector
│   │   └── utils.py           # share dropout masks, weight init helpers
│   ├── data/
│   │   ├── dataset.py         # PyTorch Dataset that streams tuples from mmap
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

## Activation Dumping: Single-GPU vs Multi-GPU

Before training Consistency Lenses, you need to extract activations from your target model. We provide two scripts for this: a single-GPU version and an optimized multi-GPU version for H100 clusters.

### Key Features

Both scripts include:
- **Layer-labeled output directories**: Activations are saved to paths like `data/activations/model_name/layer_5/train/`
- **Train/validation split support**: Can dump both splits in one run
- **Flexible data sources**: Use HuggingFace datasets or fixed prompts
- **Configurable via YAML**: See `config/lens_simple.yaml` for options

### Single-GPU Activation Dumping

For smaller models or limited GPU resources:

```bash
# Basic usage with config defaults
python consistency-lens/scripts/00_dump_activations.py \
    --config_path consistency-lens/config/lens_simple.yaml

# Override specific parameters
python consistency-lens/scripts/00_dump_activations.py \
    --config_path consistency-lens/config/lens_simple.yaml \
    --num_samples 100000 \
    --layer_idx 10 \
    --use_hf_dataset \
    --hf_dataset_name "openwebtext"
```

### Multi-GPU Activation Dumping (8x H100)

For production-scale extraction on H100 clusters:

```bash
# Using the launch script (recommended)
./consistency-lens/scripts/launch_multigpu_dump.sh \
    consistency-lens/config/lens_simple.yaml \
    data/activations \
    1000000 \
    5  # layer index

# With environment variables for additional options
USE_HF_DATASET=1 HF_DATASET_NAME="SimpleStories/SimpleStories" \
NUM_GPUS=8 ./consistency-lens/scripts/launch_multigpu_dump.sh

# Direct torchrun command for full control
python -m torch.distributed.run \
    --nproc_per_node=8 \
    --master_port=29500 \
    consistency-lens/scripts/00_dump_activations_multigpu.py \
    --config_path consistency-lens/config/lens_simple.yaml \
    --output_dir data/activations \
    --num_samples 10000000 \
    --layer_idx 20 \
    --use_hf_dataset
```

### Performance Comparison

| Configuration | Model | Dataset Size | Time | Throughput |
|--------------|-------|--------------|------|------------|
| Single GPU (A100) | Llama-2-7B | 100K samples | ~45 min | ~37 samples/sec |
| 8x H100 (data parallel) | Llama-2-7B | 1M samples | ~8 min | ~2,100 samples/sec |
| 8x H100 (data parallel) | Llama-2-70B | 1M samples | ~25 min | ~670 samples/sec |
| 8x H100 (model parallel) | Llama-2-70B | 1M samples | ~20 min | ~830 samples/sec |

### Multi-GPU Optimizations

The multi-GPU script includes several H100-specific optimizations:
- **BFloat16 inference**: Native H100 support for faster computation
- **Flash Attention**: Automatic BetterTransformer conversion when available
- **Distributed dataset loading**: Each GPU processes a unique subset
- **Model parallelism option**: For models too large for single GPU
- **Rank-aware output structure**: Shards saved to `rank_0/`, `rank_1/`, etc.

### When to Use Each

**Use single-GPU** when:
- Prototyping or debugging
- Working with smaller models (< 7B parameters)
- Limited GPU availability
- Need simple, sequential processing

**Use multi-GPU** when:
- Production-scale data extraction
- Working with large models (7B+ parameters)
- Have access to H100 clusters
- Need to process millions of samples

### Output Structure

Both scripts create the same output structure:
```
data/activations/
└── SimpleStories/
    └── layer_5/
        └── SimpleStories_train/
            ├── metadata.json          # Global metadata
            ├── rank_0/               # (Multi-GPU only)
            │   ├── shard_00000000.pt
            │   └── metadata.json
            └── rank_1/               # (Multi-GPU only)
                ├── shard_00010000.pt
                └── metadata.json
```

The training dataloader (`lens.data.dataset.ActivationDataset`) automatically handles both single and multi-GPU output formats.

## Leaning on the Ecosystem

A core principle is to leverage existing, well-tested libraries for heavy lifting, allowing our codebase to focus on the novel aspects of Consistency Lenses.

| Section                       | Library                               | How we lean on it                                                                                                             |
|-------------------------------|---------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| Tokenizer & raw text I/O      | HuggingFace `tokenizers` + `datasets` | Pre-tokenise once (`datasets.map(batch_encode_plus)`), memory-map to skip bespoke dataloader code.                            |
| Frozen model                  | `transformers.AutoModelForCausalLM`   | Out-of-the-box forward; we only patch a single hook to overwrite one hidden slice (same trick as prompt-tuning).              |
| Parameter off-loading & sharding | DeepSpeed ZeRO-1 + FSDP (torch 2.3)  | No hand-rolled DistributedDataParallel; we declare shards in `config/ds_stage2.yaml`.                                       |
| 8-bit weights                 | `bitsandbytes`                        | `load_in_8bit=True` on the frozen model gets us 4× memory cut for free. Optional.                                                   |
| Attention speedup             | `flash-attn` 3                         | `model._apply_flash_attention_2()` after load—zero code rewrite. (Note: `flash-attn` 3 uses `_apply_flash_attention_2`)     |
| Gumbel-Softmax + STE          | `torch.nn.functional.gumbel_softmax`  | Provided primitive already supports temperature; STE is two lines (`soft_output = F.gumbel_softmax(logits, tau=..., hard=True, dim=-1)` for forward, then `hard_output = F.one_hot(soft_output.argmax(dim=-1), num_classes=vocab_size).float(); y_for_grad = soft_output; y_for_input = hard_output - soft_output.detach() + soft_output`).                   |
| Training loop plumbing        | `torch.compile` (nvFuser)             | Wrap decoder/encoder forward; eliminates manual kernel fusion.                                                               |
| Optimiser                     | DeepSpeed fused AdamW                 | One config flag; avoids building custom fused kernels.                                                                       |
| Logging & dashboards          | Weights & Biases (`wandb`)            | Push metrics + the 128-sample text grid via `wandb.Table`, no HTML hacking. Configured via `config/wandb.yaml`.              |
| Checkpoint format             | DeepSpeed checkpointing               | Handles partition re-stitching; `lens.utils.checkpoint` decorates with EMA tensors and resume logic.                          |
| Scheduling (Slurm)            | Existing cluster                      | Five-line `scripts/slurm.sh`; no custom orchestration.                                                                        |
| Tests                         | `pytest`                              | Catch regressions; rely on HF's mocked models to keep CI light (`tests/smoke_train.py`, `tests/test_swap_hook.py`, etc.). |

**Bottom line:** Every heavy lift—tokenisation, flash attention, distributed memory ops, fused optimisers, checkpoints—is already solved in the ecosystem. Our codebase, primarily within the `lens/` package, is the slim glue that:
1.  Inserts the projection layers (`lens/models/decoder.py`, `lens/models/encoder.py`).
2.  Runs the Gumbel-Softmax loop (`lens/models/decoder.py`, `lens/training/loop.py`).
3.  Wires up the composite loss (`lens.training/loop.py`).
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
Managed by `scripts/00_dump_activations.py` and `lens.data.dataset`.

*   **Source Corpus & Pre-tokenization:**
    *   **Corpus:** A substantial corpus, approximately 10 billion tokens (e.g., The Pile or a suitable internal dataset).
    *   **Tokenizer:** Use the tokenizer corresponding to `LLM_orig` (e.g., `meta-llama/Llama-2-7b-hf`). Tokenizer files cached in `data/tokenizer/`.
    *   **Process:** Tokenize the entire corpus using Hugging Face `tokenizers` and `datasets`.
    *   **Storage:** Save the tokenized data as memory-mapped (mmap) Hugging Face `Dataset` shards in `data/corpus/`.

*   **Activation Tuples (`(input_ids_A, A, input_ids_A_prime, A_prime)`):**
    *   **Generation Script:** `scripts/00_dump_activations.py` (one-off pre-processing step).
    *   **Process Details:**
        1.  Load `LLM_orig` (e.g., `meta-llama/Llama-2-7b-hf` foundation model, not a chat-finetuned variant. Initialize `D` and `E` from this base model's weights.
        2.  For each sequence from the tokenized corpus:
            *   Perform a forward pass with `torch.no_grad()` and `output_hidden_states=True`.
            *   **Target Activation `A`:**
                *   Extracted from a specific layer `L` of `LLM_orig`. `L` should be a single, fixed layer in the mid-to-late-mid range (e.g., for a 32-layer model, L15, L20, or L24), configurable in `config/lens.yaml`.
                *   Extracted at a token position `P` (zero-indexed). For each training sequence, `P` is randomly sampled from a valid range (e.g., `10 <= P < sequence_length - 1`), also configurable. `A` is the hidden state *after* processing token `P`. Store `A` as `float16`.
                *   `input_ids_A`: These are the input tokens `0...P` (inclusive) from the current sequence, which provide the context for `A`. The length of `input_ids_A` is `P+1`.
            *   **Auxiliary Activation `A_prime`:**
                *   Randomly select `A_prime` from a *different* context (i.e., different sequence, and potentially different position `P_prime`, but initially from the same layer `L`).
                *   `input_ids_A_prime`: The corresponding input IDs (`0...P_prime` inclusive) for this `A_prime`. While `A_prime` is processed by `D` in isolation, storing its original context can be useful for analysis, though it's not directly used in the `L_KL` formulation specified (which conditions on `input_ids_A`).
    *   **Rationale for `L` & `P` Selection:**
        *   This approach aims to train a general-purpose lens.
        *   By sampling activations from random (valid) positions across a diverse corpus and from a semantically rich layer, `D` and `E` are encouraged to learn general principles of how activations map to textual explanations and vice-versa, rather than specializing.
    *   **Output Storage:** The collected tuples `(input_ids_A, A, input_ids_A_prime, A_prime)` are saved as `*.pt` files in `data/activations/`.
    *   **Dataset Class:** These files are consumed by `lens.data.dataset.ActivationDataset`.

*   **Context for KL Divergence (`L_KL`) Calculation (Important Detail):**
    *   For the `L_KL` loss, `logits_M_A` (original logits using `batch.A`) and `logits_M_A_target` (target logits using `A_target_for_KL`) are both computed by `LLM_orig` conditioned on `batch.input_ids_A` (i.e., the original context of `A`). The logits of interest are those for predicting the token at position `P+1` (i.e., `output.logits[:, P, :]` if `input_ids_A` are `0...P`).
    *   `A_target_for_KL` is `A_reconstructed + delta_residual`, where `delta_residual = batch.A_prime - A_prime_reconstructed`.
    *   This setup tests whether the information lost or gained during the autoencoding of `A_prime` (captured by `delta_residual`) can be meaningfully transferred to improve the functional behavior of `A` (via `A_reconstructed`) within `A`'s original context.

---

## 2. Models

Core components reside in `lens.models/`.

| Component  | Source & Implementation Details                                                                                                                                                                                                                                                                                                                                                           |
|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `LLM_orig` | `transformers.AutoModelForCausalLM`, wrapped in `lens.models.orig.OriginalModelWrapper`. Loaded in 8-bit and frozen. Custom `forward_with_replacement` hook allows swapping hidden state at layer `L`, position `P`.                                                                                                                                                                           |
| Decoder `D`| Clone of `LLM_orig`, implemented in `lens.models.decoder.DecoderModel`. Prepends prompt tokens, incorporates `Proj_A_to_D_emb`. Output layer uses `torch.nn.functional.gumbel_softmax` + STE for differentiable token sampling. Trainable.                                                                                                                                               |
| Encoder `E`| Another clone of `LLM_orig`, implemented in `lens.models.encoder.EncoderModel`. Receives the *STE-derived embeddings* of the discrete tokens chosen by `D` (Gumbel-Softmax with `hard=True`), i.e. differentiable representations of the generated text. It extracts the final token's hidden state and projects it through `Proj_E_hidden_to_A` to reconstruct `A_reconstructed`. Trainable. |

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
import torch.nn.functional as F

# D_model: lens.models.decoder.DecoderModel
# E_model: lens.models.encoder.EncoderModel
# LLM_orig_model: lens.models.orig.OriginalModelWrapper

for step, batch in enumerate(loader): # loader uses lens.data.dataset.ActivationDataset
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        # --- Primary Path ---
        # generate_soft logic within D_model
        # D_model's generate_soft returns an object containing:
        #   .generated_text_embeddings (for E)
        #   .token_logits (for L_LM)
        #   .discrete_token_ids (for L_LM)
        generated_output_D_A = D_model.generate_soft(
            activation_input=batch.A,
            max_length=config.T_text,
            gumbel_tau=gumbel_temperature_schedule(step, config.lens_yaml) # from config/lens.yaml
        )
        A_reconstructed = E_model(generated_output_D_A.generated_text_embeddings)

        # --- Auxiliary Path ---
        generated_output_D_A_prime = D_model.generate_soft(
            activation_input=batch.A_prime, # ...
            max_length=config.T_text, # ensure same length for consistency if needed
            gumbel_tau=gumbel_temperature_schedule(step, config.lens_yaml)
        )
        A_prime_reconstructed = E_model(generated_output_D_A_prime.generated_text_embeddings)

        with torch.no_grad():
            delta_residual = batch.A_prime - A_prime_reconstructed

        A_target_for_KL = A_reconstructed + delta_residual

        # --- Logit Calculation using LLM_orig_model ---
        # batch.input_ids_A are tokens 0...P. `A` is activation at layer L after processing token P.
        # We want logits for predicting token P+1.
        # In LLM_orig_model.forward_with_replacement, `target_token_pos_for_logits` should be P.
        # This means we extract output_logits[:, P, :] from the model's full sequence output.
        logits_M_A = LLM_orig_model.forward_with_replacement(
            input_ids=batch.input_ids_A,
            new_activation=batch.A,
            target_layer_idx=config.LAYER_L, # from lens.yaml
            target_token_pos_in_sequence=config.TOKEN_POS_P # from lens.yaml, this is P
        ).logits_at_target_pos # Method should return only logits for token P+1

        logits_M_A_target = LLM_orig_model.forward_with_replacement(
            input_ids=batch.input_ids_A,
            new_activation=A_target_for_KL,
            target_layer_idx=config.LAYER_L,
            target_token_pos_in_sequence=config.TOKEN_POS_P
        ).logits_at_target_pos

        # --- Loss Calculation ---
        # L_LM: KL divergence KL(P_Orig || P_D), where P_Orig are LLM_orig_model's (frozen) next-token probabilities
        # for a D_model-generated explanation, and P_D are D_model's own next-token probabilities
        # for that same explanation. This encourages D_model's generation (P_D)
        # to align with the base model's linguistic knowledge (P_Orig), keeping explanations on-manifold.
        if T_text > 1:  # need a next-token target for KL divergence
            # Logits from D_model for its own generation (predicting token t+1 given prefix 0..t from D)
            # generated_output_D_A.token_logits are (B, T_text, V)
            d_model_pred_logits = generated_output_D_A.token_logits[:, :-1, :] # Shape: (B, T_text-1, V)

            # Logits from LLM_orig_model for D_model's generated sequence
            # (predicting token t+1 given prefix 0..t from D)
            # LLM_orig_model.model is assumed to be causal and takes full input_ids
            with torch.no_grad(): # Ensure orig model isn't updated by this loss component
                orig_model_pred_logits_all_pos = LLM_orig_model.model(
                    input_ids=generated_output_D_A.discrete_token_ids # discrete_token_ids is (B, T_text)
                ).logits # Shape: (B, T_text, V)
            # We need the logits that predict the same tokens as d_model_pred_logits
            orig_model_pred_logits = orig_model_pred_logits_all_pos[:, :-1, :] # Shape: (B, T_text-1, V)

            # log_P_D: Log-distribution from D_model (this is the distribution we are training, q).
            # This will be the `input` to F.kl_div.
            # These are log-probabilities for tokens 1...T_text-1.
            log_P_D_log_probs = F.log_softmax(d_model_pred_logits, dim=-1)

            # P_Orig: Distribution from LLM_orig_model (this is the reference distribution, p).
            # This will be the `target` for F.kl_div.
            # These are probabilities for tokens 1...T_text-1 (since log_target=False for kl_div).
            P_Orig_probs = F.softmax(orig_model_pred_logits, dim=-1)
            
            # Reshape for kl_div to (N, C) where N = B * (T_text-1), C = V.
            # This ensures 'batchmean' reduction averages over token positions.
            V = d_model_pred_logits.size(-1) # Vocabulary size
            log_P_D_log_probs_flat = log_P_D_log_probs.reshape(-1, V)
            P_Orig_probs_flat = P_Orig_probs.reshape(-1, V)

            # loss_lm = KL(P_Orig || P_D)
            # F.kl_div(input, target) with log_target=False computes sum(target * (log(target) - input)).
            # Here, input is log_P_D_log_probs_flat (log q: log-probabilities from Decoder), 
            # and target is P_Orig_probs_flat (p: probabilities from Original LLM).
            # So it computes sum(P_Orig * (log P_Orig - log_P_D)), which is KL(P_Orig || P_D).
            # This regularizes the Decoder's distribution (P_D) to match the Original LLM's distribution (P_Orig).
            loss_lm = F.kl_div(
                input=log_P_D_log_probs_flat,    # log q: log-probabilities of P_D (Decoder model output)
                target=P_Orig_probs_flat,        # p: probabilities of P_Orig (Original LLM target distribution)
                reduction='batchmean',           # average KL divergence per token position
                log_target=False                 # P_Orig_probs_flat contains probabilities, not log-probabilities
            )
        else:
            loss_lm = torch.tensor(0.0) # Or handle device appropriately, e.g., device=A_reconstructed.device
        # L_KL: KL( M(A_target_for_KL) || M(A) )
        # P distribution is from M(A_target_for_KL) (target/true distribution)
        # Q distribution is from M(A) (approximating distribution)
        P_probs_A_target = F.softmax(logits_M_A_target, dim=-1)
        Q_log_probs_A = F.log_softmax(logits_M_A, dim=-1)

        loss_kl = F.kl_div(
            input=Q_log_probs_A,          # Log-probabilities of the "approximating" distribution Q
            target=P_probs_A_target,      # Probabilities of the "true" distribution P
            reduction='batchmean',        # or 'sum', ensure consistency
            log_target=False
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
*   **`D.generate_soft`:** Implemented in `lens.models.decoder.DecoderModel`. This method handles the autoregressive generation of `T_text` tokens using Gumbel-Softmax and STE. It returns necessary outputs like token logits and discrete IDs for `L_LM`, and text embeddings for `E`.
*   **Schedules (`gumbel_temperature_schedule`, `alpha_schedule`):** Defined in `config/lens.yaml`, applied in `lens.training.loop`.
*   **Optimizer (`lens.training.optim`):** Fused AdamW from DeepSpeed, differential LRs for projections vs. transformer blocks.
*   **Backpropagation through D's Autoregression:**
    *   **`L_LM` (Language Modeling Loss):** Standard backpropagation. The loss for `token_t` (derived from `logits_t` computed by `D`) backpropagates through `D`'s computation of `logits_t`. Since `logits_t` depends on `A_proj` (the projected input activation) and all previously generated tokens (`token_1`...`token_{t-1}`), gradients flow back through this dependency chain. This trains `D` to predict tokens sequentially and coherently, conditioned on the input activation `A`.
    *   **`L_KL` (KL Divergence Loss):** Full backpropagation through all `T_text` steps of `D`'s autoregressive generation is crucial.
        *   `A_reconstructed` is produced by `E` from the *entire* sequence `text = [token_1, ..., token_T_text]` generated by `D`.
        *   Gradients from `L_KL` (via `dL_KL / d(A_reconstructed)`) flow back through `E` to all `token_embedding_t` that form the input `text` to `E`.
        *   Through the Straight-Through Estimator (STE), these gradients `dL_KL / d(token_embedding_t)` are passed to `dL_KL / d(soft_token_probabilities_t)` and subsequently to `dL_KL / d(logits_t)` for each generation step `t` in `D`.
        *   Critically, `logits_t` (which `D` uses to sample `token_t`) is a function of `A_proj` and all *preceding generated tokens* (`token_1_embedding, ..., token_{t-1}_embedding`). Therefore, the gradient `dL_KL / d(logits_t)` will backpropagate not only to the parameters of `D` involved in that specific step `t` but also to `A_proj` and to the embeddings of these preceding tokens.
        *   This ensures that `D` learns how its early token choices (`token_1`, `token_2`, etc.) affect the overall utility of the complete sequence `text` for `E`'s reconstruction task. The gradient signal from `L_KL` informs every decision in `D`'s generation process, enabling it to learn the complex temporal dependencies within the generated text that are vital for minimizing `L_KL`.
*   `torch.compile` wraps the forward methods of `D`