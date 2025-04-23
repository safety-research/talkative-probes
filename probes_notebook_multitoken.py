# %% [markdown]
# # Probing LLM Activations using Activation Patching
#
# **Goal:** Investigate whether specific ground-truth information is encoded within LLM activations by patching those activations onto a neutral context and asking questions.
#
# **Methodology:**
# 1.  **Setting Definition:** Choose a domain with ground-truth information (e.g., Chess PGN, NLP Statements).
# 8.  **Activation Extraction:** Run the LLM on original contexts from the chosen domain and extract relevant hidden state activations (`x`).
# 3.  **Patching Input:** Create inputs of the form `[filler token] [question]`.
# 4.  **Activation Patching:** Run the LLM on the patching input, but replace the activation at the `[filler token]` position with the extracted activation `x`.
# 5.  **Evaluation (Patched):** Evaluate the LLM's 0-shot accuracy in answering the `[question]` based *only* on the patched activation `x`.
# 6.  **Evaluation (Context Baseline):** Evaluate the LLM's 0-shot accuracy on `[original context] [question]` as a baseline.
# 7.  **Analysis:** Compare patched accuracy vs. baseline accuracy.
import os
# Explicitly set which GPUs to use (based on your allocation)
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
# %% [markdown]
# ## 1. Setup and Configuration
import sys
import gc
def free_unused_cuda_memory():
    """Free unused cuda memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        raise RuntimeError("not using cuda")
    gc.collect()
print("hi")
print(sys.executable)
print(sys.version)
import os
# Get the SLURM environment variables
slurm_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '1')
print(slurm_gpus)
#os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in range(int(slurm_gpus)))
#os.environ["CUDA_VISIBLE_DEVICES"] = '5,6'
print(f"Restricted to {slurm_gpus} GPU(s)")
# %%
# Core libraries
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
free_unused_cuda_memory()
import nnsight
from nnsight import LanguageModel
from nnsight import CONFIG, util
from nnsight.tracing.graph import Proxy # For type hinting if needed
from IPython.display import clear_output
import os
import random
from tqdm.auto import tqdm # Progress bars
import einops # If manipulating activations like attention heads

# Plotting
import plotly.express as px
import plotly.io as pio

# Specific setting libraries (add as needed)
from patch_and_run import *
import dataset_loading
# %%
# --- Configuration ---

# Model Selection
# Start small (e.g., gpt2, pythia-70m), then scale up
#MODEL_ID = "openai-community/gpt2"
#ODEL_ID = "openai-community/gpt2-xl"
#MODEL_ID = "meta-llama/Llama-2-13b-hf"
MODEL_ID = "google/gemma-3-4b-pt"
#MODEL_ID = "EleutherAI/pythia-70m"
#num_layers = 48# vs 12
# MODEL_ID = "meta-llama/Meta-Llama-3.1-8B" # Requires HF token + access

# Model Type Configuration
IS_INSTRUCTION_MODEL = False  # Set to True for instruction-tuned models like Llama

# Execution Environment
# Set to True to use NDIF for large models
USE_REMOTE_EXECUTION = False
# NDIF_API_KEY = "YOUR_NDIF_API_KEY" # Load from secrets or env variable
# HF_TOKEN = "YOUR_HF_TOKEN" # Required for gated models like Llama

# Activation Targeting
# Example: Last layer's residual stream
#TARGET_LAYER = 11# Use negative indexing for end layers
# Use nnsight's module access syntax. Print model structure to find paths.
# Examples:
#GPT2: f"model.transformer.h.{TARGET_LAYER}.output.0" #(Residual stream output of block)
# GPT2: model.transformer.h.{TARGET_LAYER}.mlp.output[0] (MLP output)
# Llama3: model.layers.{TARGET_LAYER}.output[0] (Residual stream)
# Llama3: model.layers.{TARGET_LAYER}.mlp.output (MLP output)
# Llama3: model.layers[TARGET_LAYER].self_attn.o_proj.input (Attention output value - see tutorial)
#TARGET_MODULE_PATH = f"transformer.h[{TARGET_LAYER}].output[0]" # Adjust based on model arch
#TARGET_MODULE_PATH = f"transformer.h.{TARGET_LAYER}.mlp" #(Residual stream output of block)
# TARGET_MODULE_PATH = f"model.layers[{TARGET_LAYER}].output[0]" # For Llama

# Activation Position
# Often the last token of the original context
TARGET_TOKEN_IDX =-1

# Patching Configuration
FILLER_TOKEN = "..." # Choose a token unlikely to strongly affect things
# FILLER_TOKEN = "..." # Or something else simple

# Generation Configuration
MAX_NEW_TOKENS = 3  # Set > 1 to sample multiple tokens
TEMPERATURE = 1.0   # Higher = more random, lower = more deterministic
TOP_P = 0.9         # Control diversity of sampling
# Layer selection
LAYERS_TO_PATCH = None  # Set to None to patch all layers, or specify list like [0, 5, 10]

# Experiment Setting ('chess' or 'nlp_sentiment', 'nlp_truth', etc.)
SETTING = 'chess' # Or 'nlp_sentiment', etc.

BATCH_SIZE = 5 # For batching extraction/patching if desired
# Evaluation
if SETTING == 'chess':
    NUM_DATA = {"NUM_GAMES": 10, "NUM_SAMPLES_PER_GAME": 1, "NUM_MOVES": 10}
else:
    NUM_DATA = {"NUM_SAMPLES": 1000}
#print(TARGET_MODULE_PATH)
# %%
# --- Environment Setup ---

try:
    import google.colab
    IS_COLAB = True
    print("Running in Google Colab")
    # Setup Colab secrets if needed
    # from google.colab import userdata
    # NDIF_API_KEY = userdata.get('NDIF_API_KEY')
    # HF_TOKEN = userdata.get('HF_TOKEN')
except ImportError:
    IS_COLAB = False
    # Load from environment variables or local config if not in Colab
    NDIF_API_KEY = os.environ.get("NDIF_API_KEY", None)
    HF_TOKEN = os.environ.get("HF_TOKEN", None)

if USE_REMOTE_EXECUTION:
    print("Configuring for Remote Execution (NDIF)")
    if not NDIF_API_KEY:
        raise ValueError("NDIF_API_KEY is required for remote execution.")
    CONFIG.set_default_api_key(NDIF_API_KEY)
    print("NDIF API Key configured.")
    if "llama" in MODEL_ID.lower(): # Example check for gated models
        if not HF_TOKEN:
             raise ValueError("Hugging Face token (HF_TOKEN) is required for gated models like Llama.")
        print("HF Token available (needed for gated models).")
        # Login if needed (might be handled by environment setup elsewhere)
        # os.system(f"huggingface-cli login --token {HF_TOKEN}")
else:
    print("Configuring for Local Execution")

# Plotly renderer setup
pio.renderers.default = "colab" if IS_COLAB else "plotly_mimetype+notebook_connected+colab+notebook"


# %%
# --- Model Loading ---

print(f"Loading model: {MODEL_ID}")
# device_map='auto' works well locally, especially for multi-GPU.
# For remote execution, device_map is ignored.
# dispatch=True loads immediately, otherwise loading happens on first trace.
model = LanguageModel(
    MODEL_ID,
    device_map="auto" if not USE_REMOTE_EXECUTION else None,
    # dispatch=True # Optional: load weights immediately
)
clear_output() # Suppress verbose loading output
print(f"Model {MODEL_ID} loaded.")

# Print model structure to verify module paths
# print(model)
# Tokenizer helper
tokenizer = model.tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
filler_token_id = tokenizer.encode(FILLER_TOKEN)[-1]


# %%
#print(model.transformer.h[11])
#print(model)
# %% [markdown]
# ## 2. Setting Definition and Data Preparation
#
# Implement data loading and ground-truth extraction for the chosen `SETTING`.

# %%
# --- Data Loading and Preparation ---
dataset = dataset_loading.load_dataset(SETTING, tokenizer, NUM_DATA, IS_INSTRUCTION_MODEL, MAX_NEW_TOKENS)

# Ensure all lists have the same length
assert len(dataset['original_contexts']) == len(dataset['questions']) == len(dataset['ground_truths']) == len(dataset['ground_truth_token_ids']), \
    "Data preparation resulted in mismatched list lengths!"
print(f"Final dataset size: {len(dataset['original_contexts'])}")

# %%
print(model)
# %% [markdown]
# ## 3. Activation Extraction

# --- Activation Extraction ---
#print(model.transformer.h[-1].output.shape)


# # ## 4. Patching Experiment
# print(f"Sample activation: {list(extracted_activations.keys())[0]}, shape: {extracted_activations[list(extracted_activations.keys())[0]][0].shape}")
# Run the model on `[filler token] [question]`, patching activations from all layers onto the filler token. Also run the baseline `[original context] [question]`.
# %%
patching_inputs = [f"{FILLER_TOKEN}{q}" for q in dataset['questions']]
baseline_inputs = [f"We are playing chess. The PGN is: {ctx}\n{q}" for ctx, q in zip(dataset['original_contexts'], dataset['questions'])]
unpatched_inputs = patching_inputs.copy()

bos_or_filler_token_idx = tokenizer(patching_inputs[0], return_tensors="pt")['input_ids'][0]
print("bos_or_filler_token_idx", bos_or_filler_token_idx, tokenizer.bos_token_id)
filler_token_idx_in_patched_input = 1 if (tokenizer.bos_token_id == bos_or_filler_token_idx[0]) else 0
print(f"Assuming filler token is at index: {filler_token_idx_in_patched_input}, it translates to: {tokenizer.decode([bos_or_filler_token_idx[filler_token_idx_in_patched_input]])}")
print(dataset['original_contexts'])
print(tokenizer.encode(dataset['original_contexts'][0]))
print("\n--- Sample Example ---")
print("baseline input:", "'" + baseline_inputs[0] + "'")
print('patched input:', "'" + patching_inputs[0] + "'")
print("Ground truth:", dataset['ground_truths'][0])

#%%
# --- Patching Experiment ---

if model.config.model_type == "gemma3":
    model_lmhead = model.language_model.lm_head
else:
    model_lmhead = model.lm_head

patched_results = []
baseline_results = []
unpatched_results = []

batch_size = BATCH_SIZE
num_samples = len(dataset['original_contexts'])

# Process in batches for memory efficiency
for start in tqdm(range(0, num_samples, batch_size), desc="Extract + Patch"):
    end = min(start + batch_size, num_samples)
    batch_contexts = dataset['original_contexts'][start:end]
    patching_inputs_batch = patching_inputs[start:end]

    # Extract activations from original contexts
    activations_batch = extract_activations_batch(
        batch_contexts, 
        model, 
        TARGET_TOKEN_IDX,
        layers_to_patch=LAYERS_TO_PATCH
    )

    # Apply multi-token patching using the new function
    if MAX_NEW_TOKENS > 1:
        for i, (patching_input, activations) in enumerate(zip(patching_inputs_batch, activations_batch)):
            tokens, preds, topk_idxs, topk_logits, topk_logprobs = run_multi_token_patching(
                patching_input,
                activations,
                model, 
                filler_token_idx_in_patched_input,
                max_new_tokens=MAX_NEW_TOKENS,
                topkn=100,
                layers_to_patch=LAYERS_TO_PATCH
            )
            # Only keep the first token prediction for now
            patched_results.append((preds[0], topk_idxs[0], topk_logits[0], topk_logprobs[0]))
    else:
        # Single token prediction - use existing batch function
        patched_preds, patched_topk_idxs, patched_topk_logitss, patched_topk_logprobss = run_patching_batch(
            patching_inputs_batch, 
            activations_batch, 
            model, 
            filler_token_idx_in_patched_input,
            layers_to_patch=LAYERS_TO_PATCH
        )
        for p, idxs, logits, logprobs in zip(patched_preds, patched_topk_idxs, patched_topk_logitss, patched_topk_logprobss):
            patched_results.append((p, idxs, logits, logprobs))

# Second loop: run baseline and unpatched in batches
for start in tqdm(range(0, num_samples, batch_size), desc="Baseline + Unpatched"):
    end = min(start + batch_size, num_samples)

    baseline_inputs_batch = baseline_inputs[start:end]
    unpatched_inputs_batch = unpatched_inputs[start:end]

    # Run multi-token baseline
    if MAX_NEW_TOKENS > 1:
        for baseline_input in baseline_inputs_batch:
            tokens, preds, topk_idxs, topk_logits, topk_logprobs = run_multi_token_baseline(
                baseline_input,
                model,
                max_new_tokens=MAX_NEW_TOKENS,
                topkn=100
            )
            # Only keep the first token prediction for now
            baseline_results.append((preds[0], topk_idxs[0], topk_logits[0], topk_logprobs[0]))
        
        for unpatched_input in unpatched_inputs_batch:
            tokens, preds, topk_idxs, topk_logits, topk_logprobs = run_multi_token_baseline(
                unpatched_input,
                model,
                max_new_tokens=MAX_NEW_TOKENS,
                topkn=100
            )
            # Only keep the first token prediction for now
            unpatched_results.append((preds[0], topk_idxs[0], topk_logits[0], topk_logprobs[0]))
    else:
        # Single token prediction - use existing batch function
        baseline_preds, baseline_topk_idxs, baseline_topk_logitss, baseline_topk_logprobss = run_baseline_batch(
            baseline_inputs_batch, model
        )
        for p, idxs, logits, logprobs in zip(baseline_preds, baseline_topk_idxs, baseline_topk_logitss, baseline_topk_logprobss):
            baseline_results.append((p, idxs, logits, logprobs))

        # Run unpatched batch on batch slice only
        unpatched_preds, unpatched_topk_idxs, unpatched_topk_logitss, unpatched_topk_logprobss = run_baseline_batch(
            unpatched_inputs_batch, model
        )
        for p, idxs, logits, logprobs in zip(unpatched_preds, unpatched_topk_idxs, unpatched_topk_logitss, unpatched_topk_logprobss):
            unpatched_results.append((p, idxs, logits, logprobs))

    free_unused_cuda_memory()

print(f"Done with {len(patched_results)} examples, using {MAX_NEW_TOKENS} tokens per generation.")

# %%
# Display first few tokens of the generated sequences (if multi-token)
if MAX_NEW_TOKENS > 1:
    print("\n--- Multi-Token Generation Examples ---")
    MAX_NEW_TOKENS = 30
    
    # Get the full token predictions for sample i=0
    tokens, preds, topk_idxs, topk_logits, topk_logprobs = run_multi_token_patching(
        patching_inputs[0],
        extract_activations_for_example(dataset['original_contexts'][0], model, TARGET_TOKEN_IDX, layers_to_patch=LAYERS_TO_PATCH),
        model,
        filler_token_idx_in_patched_input,
        max_new_tokens=MAX_NEW_TOKENS,
        topkn=10
    )
    
    generated_text = tokenizer.decode(tokens[0])
    print('ground truth', dataset['ground_truths'][0])
    print(f"Sample patched input: '{patching_inputs[0]}'")
    print(f"Full generated text: '{generated_text}'")
    print(f"Tokens predicted: {[tokenizer.decode([pred.item()]) for pred in preds]}")
    # Also show baseline
    tokens, preds, topk_idxs, topk_logits, topk_logprobs = run_multi_token_baseline(
        baseline_inputs[0],
        model,
        max_new_tokens=MAX_NEW_TOKENS,
        topkn=10
    )
    
    generated_text = tokenizer.decode(tokens[0])
    print(f"\nSample baseline input: '{baseline_inputs[0]}'")
    print(f"Full generated text: '{generated_text}'")
    print(f"Tokens predicted: {[tokenizer.decode([pred.item()]) for pred in preds]}")
    
    # Also show unpatched
    tokens, preds, topk_idxs, topk_logits, topk_logprobs = run_multi_token_baseline(
        unpatched_inputs[0],
        model,
        max_new_tokens=MAX_NEW_TOKENS,
        topkn=10
    )
    
    generated_text = tokenizer.decode(tokens[0])
    print(f"\nSample unpatched input: '{unpatched_inputs[0]}'")
    print(f"Full generated text: '{generated_text}'")
    print(f"Tokens predicted: {[tokenizer.decode([pred.item()]) for pred in preds]}")

# %%
print("\nTop 10 tokens for example 0, where the ground truth is '", dataset['ground_truths'][0], "'")
print("\nbaseline input:", "'" + baseline_inputs[0] + "'\n")
print('\npatched input:', "'" + patching_inputs[0] + "'\n")
header = f"{'Rank':<5} {'Baseline':<20} {'Patched':<20} {'Unpatched':<20}"
print(header)
print("-" * len(header))

if len(baseline_results) > 0:
    baseline_pred, baseline_topk_indices, baseline_topk_logits, baseline_topk_logprobs = baseline_results[0]
    patched_pred, patched_topk_indices, patched_topk_logits, patched_topk_logprobs = patched_results[0]
    unpatched_pred, unpatched_topk_indices, unpatched_topk_logits, unpatched_topk_logprobs = unpatched_results[0]

    baseline_topk = (baseline_topk_logits, baseline_topk_indices)
    patched_topk = (patched_topk_logits, patched_topk_indices)
    unpatched_topk = (unpatched_topk_logits, unpatched_topk_indices)

    for rank in range(10):
        b_id = baseline_topk[1][rank].item()
        p_id = patched_topk[1][rank].item()
        u_id = unpatched_topk[1][rank].item()
        b_tok = tokenizer.decode([b_id]).replace("\n", "\\n")
        p_tok = tokenizer.decode([p_id]).replace("\n", "\\n")
        u_tok = tokenizer.decode([u_id]).replace("\n", "\\n")
        print(f"{rank+1:<5} {b_tok:<20} {p_tok:<20} {u_tok:<20}")

# %% [markdown]
# ## 5. Evaluation and Visualization (Plotly)

import plotly.graph_objects as go
import numpy as np

print("\n--- Sample Example ---")
print("Patched output:", "'" + tokenizer.decode(patched_results[0][0]) + "'")
print("Ground truth #1:", dataset['ground_truths'][0])
print("baseline input:", "'" + baseline_inputs[0] + "'")
print('patched input:', "'" + patching_inputs[0] + "'")

def compute_rank_and_surprisal(topk_indices, topk_logits, gt_token_ids):
    """
    Compute the rank, surprisal, and probability of the ground truth token(s).
    Accepts a list/set of possible ground truth token ids (e.g., lowercase and capitalized variants).
    Returns the best (lowest) rank among the options, and corresponding surprisal and prob.
    """
    probs = torch.softmax(topk_logits, dim=-1)
    best_rank = None
    best_prob = 0.0
    best_surprisal = float('inf')

    for gt_token_id in gt_token_ids:
        matches = (topk_indices == gt_token_id).nonzero()
        if len(matches) > 0:
            rank = matches.item() + 1
            prob_gt = probs[rank - 1].item()
            surprisal = -np.log2(prob_gt + 1e-20)
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_prob = prob_gt
                best_surprisal = surprisal
        else:
            # If not in topk, treat as rank None, prob 0, surprisal large
            pass

    if best_rank is None:
        best_prob = 0.0
        best_surprisal = -np.log2(1e-20)

    return best_rank, best_surprisal, best_prob

def compare_example(i, gt_token_ids, gt_text, baseline_result, patched_result, unpatched_result, extra_str=''):
    """
    Given index, ground truth token ids, text, and results for baseline, patched, unpatched,
    compute ranks, surprisals, and print formatted comparison line.
    Returns dicts of rank and surprisal for each condition.
    """
    _, baseline_topk_indices, baseline_topk_logits, _ = baseline_result
    _, patched_topk_indices, patched_topk_logits, _ = patched_result
    _, unpatched_topk_indices, unpatched_topk_logits, _ = unpatched_result

    rank_b, surprisal_b, _ = compute_rank_and_surprisal(baseline_topk_indices, baseline_topk_logits, gt_token_ids)
    rank_p, surprisal_p, _ = compute_rank_and_surprisal(patched_topk_indices, patched_topk_logits, gt_token_ids)
    rank_u, surprisal_u, _ = compute_rank_and_surprisal(unpatched_topk_indices, unpatched_topk_logits, gt_token_ids)

    def fmt(rank, surprisal):
        if rank is None:
            return f"{'N/A':<14}{'N/A':<10}"
        else:
            return f"{rank:<14}{surprisal:<10.2f}"

    def delta_fmt(rank_other, rank_base):
        if rank_other is None or rank_base is None:
            return ""
        delta = rank_other - rank_base
        sign = "+" if delta > 0 else ""
        return f" ({sign}{delta})"

    patched_delta = delta_fmt(rank_p, rank_b)
    unpatched_delta = delta_fmt(rank_u, rank_b)

    # Compose patched rank with delta, aligned in 18 spaces
    if rank_p is None:
        patched_rank_str = f"{'N/A':<18}"
    else:
        patched_rank_str = f"{str(rank_p) + patched_delta:<18}"

    # Compose unpatched rank with delta, aligned in 18 spaces
    if rank_u is None:
        unpatched_rank_str = f"{'N/A':<18}"
    else:
        unpatched_rank_str = f"{str(rank_u) + unpatched_delta:<18}"

    print(f"{i:<4}{str(list(gt_token_ids)):<20}{gt_text:<15} | "
          f"{fmt(rank_b, surprisal_b)} | "
          f"{patched_rank_str}{surprisal_p:<10.2f} | "
          f"{unpatched_rank_str}{surprisal_u:<10.2f} | "
          f"{extra_str}")

    return (
        {'baseline': rank_b, 'patched': rank_p, 'unpatched': rank_u},
        {'baseline': surprisal_b, 'patched': surprisal_p, 'unpatched': surprisal_u}
    )

gt_ranks = {'baseline': [], 'patched': [], 'unpatched': []}
gt_surprisals = {'baseline': [], 'patched': [], 'unpatched': []}

print("\n--- Ground Truth Token Ranks and Surprisals ---")
print(f"{'Idx':<4}{'GT_IDs':<20}{'Truth':<15} | "
      f"{'BL rank':<14}{'Surp':<10} | "
      f"{'P rank':<18}{'Surp':<10} | "
      f"{'UP rank':<18}{'Surp':<10} | "
      f"{'Info'}")

for i in range(len(dataset['ground_truths'])):
    if i >= len(baseline_results) or i >= len(patched_results) or i >= len(unpatched_results):
        break

    gt_text = dataset['ground_truths'][i]
    gt_token_ids = tokenizer.encode(gt_text, add_special_tokens=False)

    # Also add capitalized version if different
    gt_text_cap = gt_text.capitalize()
    gt_token_ids_cap = tokenizer.encode(gt_text_cap, add_special_tokens=False)

    # Only support single-token ground truths (and their capitalized variant)
    if len(gt_token_ids) == 1:
        ids_set = {gt_token_ids[0]}
    else:
        print(f"{i:<4} Multi-token GT not supported")
        continue

    if len(gt_token_ids_cap) == 1:
        ids_set.add(gt_token_ids_cap[0])

    extra_info = dataset.get('extra_info', [''] * len(dataset['ground_truths']))
    extra_str = extra_info[i] if i < len(extra_info) else ''

    ranks, surprisals = compare_example(
        i,
        ids_set,
        gt_text,
        baseline_results[i],
        patched_results[i],
        unpatched_results[i],
        extra_str=extra_str
    )

    for key in ['baseline', 'patched', 'unpatched']:
        gt_ranks[key].append(ranks[key])
        gt_surprisals[key].append(surprisals[key])
        #gt_probs[key].append(probs[key])

# Plotly boxplot for surprisals
fig_surp = go.Figure()

x_labels = ['Baseline', 'Patched', 'Unpatched']
colors = ['blue', 'red', 'green']
for key, color, label in zip(['baseline', 'patched', 'unpatched'], colors, x_labels):
    fig_surp.add_trace(go.Box(
        y=gt_surprisals[key],
        x=[label] * len(gt_surprisals[key]),
        name=label,
        boxmean=True,
        marker_color=color,
        boxpoints='outliers'
    ))

# Add faint lines connecting each instance's baseline->patched->unpatched
n = len(gt_surprisals['baseline'])
for i in range(n):
    y_vals = [
        gt_surprisals['baseline'][i],
        gt_surprisals['patched'][i],
        gt_surprisals['unpatched'][i]
    ]
    fig_surp.add_trace(go.Scatter(
        x=x_labels,
        y=y_vals,
        mode='lines',
        line=dict(color='rgba(0,0,0,0.03)', width=1),
        hoverinfo='skip',
        showlegend=False
    ))

fig_surp.update_layout(
    title=f"ground truth token surprisal (↓ better), chess, {NUM_DATA['NUM_MOVES']} moves, {NUM_DATA['NUM_GAMES']*NUM_DATA['NUM_SAMPLES_PER_GAME']} examples = {NUM_DATA['NUM_SAMPLES_PER_GAME']}q x {NUM_DATA['NUM_GAMES']}g, model: {MODEL_ID}",
    yaxis_title="surprisal (bits)",
    template="plotly_white"
)
fig_surp.show()

def summarize_metric(metric_dict, name):
    print(f"\n--- {name} Summary ---")
    for key in metric_dict:
        arr = np.array(metric_dict[key])
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            continue
        print(f"{key.capitalize():<10}: Mean={np.mean(arr):.2f}, Median={np.median(arr):.2f}, Min={np.min(arr):.2f}, Max={np.max(arr):.2f}")

summarize_metric(gt_ranks, "GT Token Rank")
summarize_metric(gt_surprisals, "GT Token Surprisal")
#summarize_metric(gt_probs, "GT Token Probability")

print("\n--- Examples with Lowest Patched Surprisal ---")
sorted_idx = np.argsort(gt_surprisals['patched'])
for idx in sorted_idx[:5]:
    extra_info = dataset.get('extra_info', [''] * len(dataset['ground_truths']))
    extra_str = extra_info[idx] if idx < len(extra_info) else ''
    print(f"Example {idx}: Surprisal={gt_surprisals['patched'][idx]:.2f}, Rank={gt_ranks['patched'][idx]}, GT='{dataset['ground_truths'][idx]}', Q='{dataset['questions'][idx]}', Extra='{extra_str}', UP_sup={gt_surprisals['unpatched'][idx]:.2f}, UP_R={gt_ranks['unpatched'][idx]}, B_sup={gt_surprisals['baseline'][idx]:.2f}, B_R={gt_ranks['baseline'][idx]}")

print("\n--- Examples with Highest Patched Surprisal ---")
sorted_idx = np.argsort(gt_surprisals['patched'])[::-1]
for idx in sorted_idx[:5]:
    extra_info = dataset.get('extra_info', [''] * len(dataset['ground_truths']))
    extra_str = extra_info[idx] if idx < len(extra_info) else ''
    print(f"Example {idx}: Surprisal={gt_surprisals['patched'][idx]:.2f}, Rank={gt_ranks['patched'][idx]}, GT='{dataset['ground_truths'][idx]}', Q='{dataset['questions'][idx]}', Extra='{extra_str}', UP_sup={gt_surprisals['unpatched'][idx]:.2f}, UP_R={gt_ranks['unpatched'][idx]}, B_sup={gt_surprisals['baseline'][idx]:.2f}, B_R={gt_ranks['baseline'][idx]}")

print("\n--- Examples with Highest Baseline Surprisal ---")
sorted_idx = np.argsort(gt_surprisals['baseline'])[::-1]
for idx in sorted_idx[:5]:
    extra_info = dataset.get('extra_info', [''] * len(dataset['ground_truths']))
    extra_str = extra_info[idx] if idx < len(extra_info) else ''
    print(f"Example {idx}: Surprisal={gt_surprisals['baseline'][idx]:.2f}, Rank={gt_ranks['baseline'][idx]}, GT='{dataset['ground_truths'][idx]}', Q='{dataset['questions'][idx]}', Extra='{extra_str}'")

# %%
idx = min(0, len(dataset['original_contexts']) - 1)
print(dataset['original_contexts'][idx])
print(dataset['questions'][idx])
print(dataset['ground_truths'][idx])
if idx < len(baseline_results):
    baseline_pred, baseline_topk_indices, baseline_topk_logits, baseline_topk_logprobs = baseline_results[idx]
    patched_pred, patched_topk_indices, patched_topk_logits, patched_topk_logprobs = patched_results[idx]
    unpatched_pred, unpatched_topk_indices, unpatched_topk_logits, unpatched_topk_logprobs = unpatched_results[idx]

    baseline_topk = (baseline_topk_logits, baseline_topk_indices)
    patched_topk = (patched_topk_logits, patched_topk_indices)
    unpatched_topk = (unpatched_topk_logits, unpatched_topk_indices)

    print(f"\nTop 12 tokens for example {idx}:")
    header = f"{'Rank':<5} {'Baseline':<20} {'Patched':<20} {'Unpatched':<20}"
    print(header)
    print("-" * len(header))

    for rank in range(12):
        b_id = baseline_topk[1][rank].item()
        p_id = patched_topk[1][rank].item()
        u_id = unpatched_topk[1][rank].item()
        b_tok = tokenizer.decode([b_id]).replace("\n", "\\n")
        p_tok = tokenizer.decode([p_id]).replace("\n", "\\n")
        u_tok = tokenizer.decode([u_id]).replace("\n", "\\n")
        print(f"{rank+1:<5} {b_tok:<20} {p_tok:<20} {u_tok:<20}")
else:
    print(f"Example {idx} is out of range.")

# # %%
# print("\n--- nnsight generate baseline continuation (first example) ---")
# prompt = baseline_inputs[0]
# n_new_tokens = 5
# with model.generate(prompt, max_new_tokens=n_new_tokens, temperature=TEMPERATURE, top_p=TOP_P) as tracer:
#     out = model.generator.output.save()

# decoded_prompt = model.tokenizer.decode(out[0][0:-n_new_tokens].cpu())
# decoded_answer = model.tokenizer.decode(out[0][-n_new_tokens:].cpu())

# print("Prompt:", decoded_prompt)
# print("Generated continuation:", decoded_answer)

# # %%
# print(len(patched_results[0]))
# %%
# Display first few tokens of the generated sequences (if multi-token)

    # Also show baseline
    # tokens, preds, topk_idxs, topk_logits, topk_logprobs = run_multi_token_baseline(
    #     baseline_inputs[0],
    #     model,
    #     max_new_tokens=MAX_NEW_TOKENS,
    #     topkn=10
    # )
    
    # generated_text = tokenizer.decode(tokens[0])
    # print(f"\nSample baseline input: '{baseline_inputs[0]}'")
    # print(f"Full generated text: '{generated_text}'")
    # print(f"Tokens predicted: {[tokenizer.decode([pred.item()]) for pred in preds]}")
    
    # # Also show unpatched
    # tokens, preds, topk_idxs, topk_logits, topk_logprobs = run_multi_token_baseline(
    #     unpatched_inputs[0],
    #     model,
    #     max_new_tokens=MAX_NEW_TOKENS,
    #     topkn=10
    # )
    
    # generated_text = tokenizer.decode(tokens[0])
    # print(f"\nSample unpatched input: '{unpatched_inputs[0]}'")
    # print(f"Full generated text: '{generated_text}'")
    # print(f"Tokens predicted: {[tokenizer.decode([pred.item()]) for pred in preds]}")
# %%
example_idx = 0

# Extract activations for a single example
toextract = "bananas are my favourite food"
activations = extract_activations_for_example(toextract, model, TARGET_TOKEN_IDX, layers_to_patch=None)
print("target token: ", tokenizer.decode(tokenizer.encode(toextract)[TARGET_TOKEN_IDX]))
# Run patching on a single example
test_single_patch = "ChessWhat colour are they? They are"
print(test_single_patch)
patchedactivations ={k:v for k,v in activations.items()}
patched_pred, patched_topk_indices, patched_topk_logits, patched_topk_logprobs = run_patching(
    test_single_patch, patchedactivations, model, filler_token_idx_in_patched_input
)
print(tokenizer.decode(patched_pred))
print([tokenizer.decode(p) for p in patched_topk_indices])

# %%

if MAX_NEW_TOKENS > 1:
    print("\n--- Multi-Token Generation Examples ---")
    MAX_NEW_TOKENS = 30
    #test_patch_input = "QueensWhat colour are they? They are"
    toextract = "The tallest building in New York"
    #test_patch_input = "QueensWhat colour are they? They are"
    #test_patch_input = "...It is perhaps"
    test_patch_input = """Syria: Country in the Middle East, Leonardo DiCaprio: American actor, Samsung: South
Korean multinational major appliance and consumer electronics corporation, King: """
    # Get the full token predictions for sample i=0
    tokenextractidx = -1
    acts = extract_activations_for_example(toextract, model, tokenextractidx, layers_to_patch=[0,5,10,20,30])
    print("extracted token:", tokenizer.decode(tokenizer.encode(toextract)[tokenextractidx]))
    acts = {k:v for k,v in acts.items()}
    print(acts.keys())
    filler_token_here = -3
    print("replacing token:", tokenizer.decode(tokenizer.encode(test_patch_input)[filler_token_here]))
    tokens, preds, topk_idxs, topk_logits, topk_logprobs = run_multi_token_patching(
        test_patch_input,
        acts,
        model,
        filler_token_here,#filler_token_idx_in_patched_input
        max_new_tokens=MAX_NEW_TOKENS,
        topkn=5,
        temperature=None#0.0000000001
    )

    print([tokenizer.decode(k) for k in topk_idxs]) 
    
    generated_text = tokenizer.decode(tokens[0])
    print('ground truth', dataset['ground_truths'][0])
    print(f"Sample patched input: '{test_patch_input}'")
    print(f"Full generated text: '{generated_text}'")
    print(f"Tokens predicted: {[tokenizer.decode([pred.item()]) for pred in preds]}")


# # %%
# tokens, preds, topk_idxs, topk_logits, topk_logprobs = run_multi_token_baseline(test_patch_input, model, max_new_tokens=MAX_NEW_TOKENS, topkn=10, temperature=0.0000000001)

# print([tokenizer.decode(k) for k in topk_idxs]) 
# generated_text = tokenizer.decode(tokens[0])
# print('ground truth', dataset['ground_truths'][0])
# print(f"Sample patched input: '{test_patch_input}'")
# print(f"Full generated text: '{generated_text}'")
# print(f"Tokens predicted: {[tokenizer.decode([pred.item()]) for pred in preds]}")


# %%
topk_logits[0]

# _, model_lmhead = get_model_info(model)

    
with model.generate(test_patch_input, max_new_tokens=MAX_NEW_TOKENS, remote=USE_REMOTE_EXECUTION, temperature=0.0000000001) as tracer:
        
    # Create empty lists to store results
    all_preds = nnsight.list().save()
    all_topk_idx = nnsight.list().save()
    all_topk_logits = nnsight.list().save()
    all_topk_logprobs = nnsight.list().save()
    # Get generated tokens
    all_tokens = model.generator.output.save()
print([tokenizer.decode(k) for k in all_tokens])
# %%

def run_multi_token_patching(patching_input, activations, model, filler_token_idx, max_new_tokens=1, topkn=100, layers_to_patch=None, temperature = None):
    """
    Run patching with multi-token prediction using model.generate
    
    Args:
        patching_input: Input text for patching
        activations: Dictionary of activations to patch
        model: Model to patch
        filler_token_idx: Index of token to replace with activation
        max_new_tokens: Number of new tokens to generate
        topkn: Number of top predictions to return
        layers_to_patch: Optional list of layer indices to patch
        
    Returns:
        Generated tokens, predictions, and logprobs for each new token
    """
    num_layers, model_lmhead = get_model_info(model)
    
    # Create a dictionary to map layer path to module
    layer_modules = {}
    for layer_idx in range(num_layers):
        if layers_to_patch is not None and layer_idx not in layers_to_patch:
            continue
        layer_path = get_layer_output_path(layer_idx, model)
        layer_modules[layer_path] = util.fetch_attr(model, layer_path)
    

    
    with model.generate(patching_input, max_new_tokens=max_new_tokens, remote=USE_REMOTE_EXECUTION, temperature=temperature) as tracer:
        # Create empty lists to store results
        all_tokens = []
        all_preds = nnsight.list().save()
        all_topk_idx = nnsight.list().save()
        all_topk_logits = nnsight.list().save()
        all_topk_logprobs = nnsight.list().save()
        # First apply patching at the initial position
        for layer_path, act in activations.items():
            if layers_to_patch is not None:
                try:
                    layer_idx = int(layer_path.split('.')[-1])
                except ValueError:
                    continue
                if layer_idx not in layers_to_patch:
                    continue
            module = layer_modules.get(layer_path)
            #if module:
            #print(layer_path)
            module.output[0][0, filler_token_idx, :] = act
        
        # Get generated tokens
        all_tokens = model.generator.output.save()
        
        # Use .all() to apply to all new token predictions
        with model_lmhead.all():
            logits = model_lmhead.output[0, -1, :]
            pred = torch.argmax(logits, dim=-1)
            sorted_idx = torch.argsort(logits, descending=True)
            topk_idx = sorted_idx[:topkn]
            topk_logits = logits[topk_idx]
            topk_logprobs = torch.log_softmax(logits, dim=-1)[topk_idx]
            
            all_preds.append(pred)
            all_topk_idx.append(topk_idx)
            all_topk_logits.append(topk_logits)
            all_topk_logprobs.append(topk_logprobs)
    
    # Convert to CPU tensors
    all_tokens = all_tokens.detach().cpu()
    all_preds = [p.detach().cpu() for p in all_preds]
    all_topk_idx = [idx.detach().cpu() for idx in all_topk_idx]
    all_topk_logits = [logit.detach().cpu() for logit in all_topk_logits]
    all_topk_logprobs = [logprob.detach().cpu() for logprob in all_topk_logprobs]
    
    return all_tokens, all_preds, all_topk_idx, all_topk_logits, all_topk_logprobs# %%

# %%
def apply_logit_lens(model, prompt, topkn=5):
    """
    Apply logit lens to model layers to visualize intermediate predictions
    """
    num_layers, model_lmhead = get_model_info(model)
    #num_layers = 3
    probs_layers = []
    topk_tokens_layers = []
    topk_probs_layers = []
    
    with model.trace() as tracer:
        with tracer.invoke(prompt) as invoker:
            for layer_idx in range(num_layers):
                layer_path = get_layer_output_path(layer_idx, model)
                module = util.fetch_attr(model, layer_path)
                
                # Process layer output through the model's head and normalization
                if model.config.model_type == "gemma3":
                    layer_output = -model.language_model.lm_head(model.language_model.model.norm(module.output[0]))
                elif model.config.model_type == "gpt2":
                    layer_output = model.lm_head(model.transformer.ln_f(module.output[0]))
                elif model.config.model_type == "llama":
                    layer_output = model.lm_head(model.norm(module.output[0]))
                else:
                    # Generic fallback
                    layer_output = model_lmhead(module.output[0])
                
                # Apply softmax to obtain probabilities and save the result
                probs = torch.nn.functional.softmax(layer_output, dim=-1).save()
                probs_layers.append(probs)
                
                # Get top-k tokens by sorting logits
                sorted_idx = torch.argsort(layer_output, descending=True, dim=-1)
                topk_tokens = sorted_idx[..., :topkn]
                topk_probs = probs.gather(-1, topk_tokens)
                topk_tokens_layers.append(topk_tokens.save())
                topk_probs_layers.append(topk_probs.save())
    
    # Convert to CPU tensors
    probs = torch.cat([probs.value.detach().cpu() for probs in probs_layers])
    topk_tokens = torch.cat([tokens.value.detach().cpu() for tokens in topk_tokens_layers])
    topk_probs = torch.cat([probs.value.detach().cpu() for probs in topk_probs_layers])
    
    # Find the maximum probability and corresponding tokens for each position
    print("SHAPE:",probs.shape)
    if model.config.model_type == "gemma3":
        print('taking max over -2 dimension')
        max_probs, tokens = probs.max(dim=-1)
    else:
        print('taking max over -1 dimension')
        max_probs, tokens = probs.max(dim=-1)


    print('max probs shape:', max_probs.shape)
    print('tokens shape:', tokens.shape)
    # print(tokens[0:10,0:10])

    # return None
    
    # Decode token IDs to words for each layer
    words = [[model.tokenizer.decode(t).encode("unicode_escape").decode() for t in layer_tokens]
        for layer_tokens in tokens]
    
    # Decode top-k tokens for each layer and position
    topk_words = [[[model.tokenizer.decode(t).encode("unicode_escape").decode() 
                    for t in pos_tokens] 
                   for pos_tokens in layer_tokens]
                  for layer_tokens in topk_tokens]
    
    # Get input words
    input_words = [model.tokenizer.decode(t) for t in invoker.inputs[0][0]["input_ids"][0]]
    
    return {
        'max_probs': max_probs,
        'tokens': tokens,
        'words': words,
        'input_words': input_words,
        'probs': probs,
        'topk_tokens': topk_tokens,
        'topk_probs': topk_probs,
        'topk_words': topk_words
    }

def visualize_logit_lens(logit_lens_results):
    """
    Visualize logit lens results using plotly
    """
    import plotly.express as px
    import plotly.io as pio
    
    max_probs = logit_lens_results['max_probs']
    words = logit_lens_results['words']
    input_words = logit_lens_results['input_words']
    
    # Determine if running in Colab
    try:
        import google.colab
        is_colab = True
    except:
        is_colab = False
    
    if is_colab:
        pio.renderers.default = "colab"
    else:
        pio.renderers.default = "plotly_mimetype+notebook_connected+notebook"
    
    fig = px.imshow(
        max_probs,
        x=input_words,
        y=list(range(len(words))),
        color_continuous_scale=px.colors.diverging.RdYlBu_r,
        color_continuous_midpoint=0.50,
        text_auto=True,
        labels=dict(x="Input Tokens", y="Layers", color="Probability")
    )
    
    fig.update_layout(
        title='Logit Lens Visualization',
        xaxis_tickangle=0
    )
    
    fig.update_traces(text=words, texttemplate="%{text}")
    return fig

# Example usage
example_prompt = "The tallest building in New York is the"
logit_lens_results = apply_logit_lens(model, example_prompt)
fig = visualize_logit_lens(logit_lens_results)
fig.show()

# Print top predictions for the last token at different layers
print("\nTop predictions at different layers for the last token:")
last_token_idx = -1
for layer_idx in range(0, len(logit_lens_results['topk_words']), 4):  # Sample every 4th layer
    top_words = logit_lens_results['topk_words'][layer_idx][last_token_idx][:3]  # Top 3 predictions
    top_probs = logit_lens_results['topk_probs'][layer_idx][last_token_idx][:3]  # Top 3 probabilities
    print(f"Layer {layer_idx}: {list(zip(top_words, [f'{p:.4f}' for p in top_probs]))}")

# Example usage
# out = apply_logit_lens(model, "The tallest building in New York")
# visualize_logit_lens(out)
# %%
print(model)
# # %%
# (language_model): Gemma3ForCausalLM(
#     (model): Gemma3TextModel(
#       (embed_tokens): Gemma3TextScaledWordEmbedding(262208, 2560, padding_idx=0)
#       (layers): ModuleList(
#         (0-33): 34 x Gemma3DecoderLayer(
#           (self_attn): Gemma3Attention(
#             (q_proj): Linear(in_features=2560, out_features=2048, bias=False)
#             (k_proj): Linear(in_features=2560, out_features=1024, bias=False)
#             (v_proj): Linear(in_features=2560, out_features=1024, bias=False)
#             (o_proj): Linear(in_features=2048, out_features=2560, bias=False)
#             (q_norm): Gemma3RMSNorm((256,), eps=1e-06)
#             (k_norm): Gemma3RMSNorm((256,), eps=1e-06)
#           )
#           (mlp): Gemma3MLP(
#             (gate_proj): Linear(in_features=2560, out_features=10240, bias=False)
#             (up_proj): Linear(in_features=2560, out_features=10240, bias=False)
#             (down_proj): Linear(in_features=10240, out_features=2560, bias=False)
#             (act_fn): PytorchGELUTanh()
#           )
#           (input_layernorm): Gemma3RMSNorm((2560,), eps=1e-06)
#           (post_attention_layernorm): Gemma3RMSNorm((2560,), eps=1e-06)
#           (pre_feedforward_layernorm): Gemma3RMSNorm((2560,), eps=1e-06)
#           (post_feedforward_layernorm): Gemma3RMSNorm((2560,), eps=1e-06)
#         )
#       )
#       (norm): Gemma3RMSNorm((2560,), eps=1e-06)
#       (rotary_emb): Gemma3RotaryEmbedding()
#       (rotary_emb_local): Gemma3RotaryEmbedding()
#     )
#     (lm_head): Linear(in_features=2560, out_features=262208, bias=False)
#   )