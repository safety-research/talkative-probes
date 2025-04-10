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
try:
    import chess.pgn
    import chess
    print("Chess library loaded.")
except ImportError:
    print("Chess library not found. Install with 'pip install python-chess'")

try:
    from datasets import load_dataset
    print("Hugging Face datasets library loaded.")
except ImportError:
    print("datasets library not found. Install with 'pip install datasets'")

from patch_and_run import *


# %%
# --- Configuration ---

# Model Selection
# Start small (e.g., gpt2, pythia-70m), then scale up
#MODEL_ID = "openai-community/gpt2"
#MODEL_ID = "openai-community/gpt2-xl"
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
MAX_NEW_TOKENS = 1  # Set > 1 to sample multiple tokens
TEMPERATURE = 1.0   # Higher = more random, lower = more deterministic
TOP_P = 0.9         # Control diversity of sampling

# Experiment Setting ('chess' or 'nlp_sentiment', 'nlp_truth', etc.)
SETTING = 'chess' # Or 'nlp_sentiment', etc.

BATCH_SIZE = 5 # For batching extraction/patching if desired
# Evaluation
if SETTING == 'chess':
    NUM_GAMES = 500
    NUM_SAMPLES_PER_GAME = 1#50 # Number of data points to process
    NUM_MOVES = 10
else:
    NUM_SAMPLES = 1000 #50 # Number of data points to process
    BATCH_SIZE = 10 # For batching extraction/patching if desired
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

# Use a dictionary to store dataset items
dataset = {
    'original_contexts': [],
    'questions': [],
    'ground_truths': [], # Store the expected correct answer (e.g., token ID, string label)
    'ground_truth_token_ids': [], # Store the token ID of the correct answer word/phrase
    'extra_info': [] # Store the square name of the correct answer
}

# Helper function to format questions based on model type
def format_question(question_text, is_instruction_model=IS_INSTRUCTION_MODEL):
    """Format questions differently for base vs instruction models."""
    if is_instruction_model:
        # For instruction models, use direct questioning
        return question_text
    else:
        # For base models, prime with a completion pattern
        # Add text that encourages the model to begin answering
        if SETTING == 'chess':
            return f"{question_text} It is a"
        else:
            return f"{question_text} The answer is"
if SETTING == 'chess':
    print("Preparing Chess PGN data...")
    try:
        import io

        # Define multiple PGN strings
        # pgn_strings = [
        #     """
        #     1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *
        #     """,
        #     """
        #     1. d4 d5 2. c4 e6 3. Nc3 Nf6 *
        #     """,
        #     """
        #     1. e4 c5 2. Nf3 d6 3. d4 cxd4 *
        #     """,
        #     """
        #     1. e4 e6 2. d4 d5 3. Nc3 Nf6 *
        #     """,
        #     """
        #     1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 *
        #     """,
        #     """
        #     1. e4 c6 2. d4 d5 3. Nc3 dxe4 *
        #     """,
        #     """
        #     1. d4 d5 2. Nf3 Nf6 3. e3 e6 *
        #     """,
        #     """
        #     1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 *
        #     """,
        #     """
        #     1. d4 Nf6 2. c4 e6 3. Nf3 d5 *
        #     """,
        #     """
        #     1. e4 c5 2. Nf3 Nc6 3. d4 cxd4 *
        #     """
        # ]
        import pandas as pd

        df = pd.read_csv('/workspace/kitf/data/kaggle_chessgames.csv')
        if 'Moves' not in df.columns:
            raise ValueError("CSV does not contain a 'moves' column")

        pgn_strings = df['Moves'].dropna().tolist()


        num_pgns = min(NUM_GAMES, len(pgn_strings))
        samples_per_pgn = NUM_SAMPLES_PER_GAME 
        #

        total_loaded = 0
        for idx, pgn_string in enumerate(pgn_strings[:num_pgns]):
            n_samples = samples_per_pgn
            print(f"Processing PGN {idx+1} of {num_pgns}, {n_samples} samples")
            pgn_file = io.StringIO(pgn_string)
            game = chess.pgn.read_game(pgn_file)
            board = game.board()

            moved_squares = set()
            node = game
            move_count = 0
            # Play up to NUM_MOVES moves, track moved piece squares simultaneously
            while node.variations and move_count < NUM_MOVES:
                next_node = node.variations[0]
                move = next_node.move
                board.push(move)
                moved_squares.add(move.to_square)
                node = next_node
                move_count += 1

            # Reconstruct PGN string up to capped move count
            capped_pgn_moves = []
            node = game
            move_counter = 0
            while node.variations and move_counter < move_count:
                next_node = node.variations[0]
                move_san = node.board().san(next_node.move)
                capped_pgn_moves.append(move_san)
                node = next_node
                move_counter += 1

            # Compose PGN string fragment
            capped_pgn_str = ""
            for idx_move, move_san in enumerate(capped_pgn_moves):
                if idx_move % 2 == 0:
                    capped_pgn_str += f"{(idx_move // 2) +1}. {move_san} "
                else:
                    capped_pgn_str += f"{move_san} "
            #capped_pgn_str = capped_pgn_str.strip() + " *"
            capped_pgn_str = capped_pgn_str.strip() + f" {idx_move//2+2}."

            count = 0
            while count < n_samples:
                # Only consider occupied squares where the piece has moved during the game
                candidate_squares = [sq for sq in moved_squares if board.piece_at(sq)]
                if not candidate_squares:
                    break

                target_square = random.choice(candidate_squares)
                square_name = chess.square_name(target_square)
                question_text = f"What chess piece is on {square_name}?"

                piece = board.piece_at(target_square)
                piece_name = chess.piece_name(piece.piece_type).capitalize()
                correct_answer_str = f" {piece_name}".lower()

                # Use the PGN string up to capped move count as context
                dataset['original_contexts'].append(capped_pgn_str.strip())
                dataset['questions'].append(format_question(question_text))
                dataset['ground_truths'].append(correct_answer_str)
                dataset['extra_info'].append(square_name)

                gt_token_id = tokenizer.encode(correct_answer_str, add_special_tokens=False)
                if len(gt_token_id) == 1:
                    dataset['ground_truth_token_ids'].append(gt_token_id[0])
                else:
                    dataset['ground_truth_token_ids'].append(gt_token_id)
                    #if MAX_NEW_TOKENS == 1:
                    #    dataset['original_contexts'].pop()
                    #    dataset['questions'].pop()
                    #    dataset['ground_truths'].pop()
                    #    dataset['ground_truth_token_ids'].pop()
                    #    continue

                count += 1
                total_loaded += 1

            pgn_file.close()

        print(f"Loaded {total_loaded} chess positions and questions equally from {num_pgns} PGNs, up to a maximum of {NUM_MOVES} moves per game.")

    except Exception as e:
        print(f"Error processing PGN data: {e}")

elif SETTING == 'nlp_sentiment':
    print("Preparing NLP Sentiment data (e.g., SST-2)...")
    try:
        hf_dataset = load_dataset("sst2", split='validation') # Use validation set
        hf_dataset = hf_dataset.shuffle(seed=42).select(range(min(NUM_SAMPLES, len(hf_dataset))))

        sentiment_map = {0: " Negative", 1: " Positive"} # Add space for tokenization

        for example in hf_dataset:
            dataset['original_contexts'].append(example['sentence'])
            # Apply formatting based on model type
            dataset['questions'].append(format_question("What is the sentiment?"))
            gt_label = example['label']
            gt_text = sentiment_map[gt_label]
            dataset['ground_truths'].append(gt_text)
            # Get token IDs for the ground truth
            gt_token_id = tokenizer.encode(gt_text, add_special_tokens=False)
            if len(gt_token_id) == 1 or MAX_NEW_TOKENS > 1:
                dataset['ground_truth_token_ids'].append(gt_token_id)
            else:
                print(f"Warning: Sentiment label '{gt_text}' is multi-token. Check tokenization.")
                # Decide how to handle: skip, use first token, etc. Skipping for now.
                dataset['original_contexts'].pop()
                dataset['questions'].pop()
                dataset['ground_truths'].pop()
                continue

        print(f"Loaded {len(dataset['original_contexts'])} sentiment analysis samples.")

    except Exception as e:
        print(f"Error loading/processing dataset: {e}")

elif SETTING == 'nlp_truth':
    print("Preparing NLP Truthfulness data (e.g., BoolQ)...")
    # Example using BoolQ - requires careful question formulation
    try:
        hf_dataset = load_dataset("boolq", split='validation')
        hf_dataset = hf_dataset.shuffle(seed=42).select(range(min(NUM_SAMPLES, len(hf_dataset))))

        truth_map = {True: " Yes", False: " No"} # Add space

        for example in hf_dataset:
            # BoolQ context is usually long (passage). Use question as context?
            # Or use passage and ask 'Is the following statement true: [question]'?
            # Let's use the provided question as the primary 'context' for simplicity
            dataset['original_contexts'].append(example['question'])
            # Apply formatting based on model type
            dataset['questions'].append(format_question("Is this statement true or false?"))
            gt_label = example['answer']
            gt_text = truth_map[gt_label]
            dataset['ground_truths'].append(gt_text)

            gt_token_id = tokenizer.encode(gt_text, add_special_tokens=False)
            if len(gt_token_id) == 1 or MAX_NEW_TOKENS > 1:
                dataset['ground_truth_token_ids'].append(gt_token_id)
            else:
                print(f"Warning: Truth label '{gt_text}' is multi-token. Check tokenization.")
                dataset['original_contexts'].pop()
                dataset['questions'].pop()
                dataset['ground_truths'].pop()
                continue

        print(f"Loaded {len(dataset['original_contexts'])} truthfulness samples.")
    except Exception as e:
        print(f"Error loading/processing dataset: {e}")
elif SETTING == 'secret':
    print("Preparing secret information probing data...")
    # Create examples where a specific piece of information is embedded in context
    # Each secret should be a single token for clear representation
    
    # Check which tokens are single tokens in our tokenizer
    test_tokens = [" dog", " triangle", " dragon", " chair", " square"]
    single_tokens = []
    for token in test_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) == 1:
            single_tokens.append((token, ids[0]))
    
    if not single_tokens:
        raise ValueError("No single-token secrets found. Check tokenization.")
    
    # Create contexts with the verified single tokens
    for token, token_id in single_tokens:
        # Strip the space for the context sentence
        word = token.strip()
        context = f"The secret object is a {word}. What is the secret object?"
        question = ": the secret object is a"
        
        dataset['original_contexts'].append(context)
        dataset['questions'].append(question)
        dataset['ground_truths'].append(token)  # Keep the space for proper tokenization
        dataset['ground_truth_token_ids'].append(token_id)
    
    print(f"Created {len(dataset['original_contexts'])} secret information examples with single-token answers.")
else:
    raise ValueError(f"Unknown SETTING: {SETTING}")

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

# Batched extraction and patching
for start in tqdm(range(0, num_samples, batch_size), desc="Extract + Patch"):
    end = min(start + batch_size, num_samples)
    batch_contexts = dataset['original_contexts'][start:end]
    patching_inputs_batch = patching_inputs[start:end]

    activations_batch = extract_activations_batch(batch_contexts, model, TARGET_TOKEN_IDX)

    patched_preds, patched_topk_idxs, patched_topk_logitss, patched_topk_logprobss = run_patching_batch(
        patching_inputs_batch, activations_batch, model, filler_token_idx_in_patched_input
    )
    for p, idxs, logits, logprobs in zip(patched_preds, patched_topk_idxs, patched_topk_logitss, patched_topk_logprobss):
        patched_results.append((p, idxs, logits, logprobs))

# Second loop: run baseline and unpatched in batches
for start in tqdm(range(0, num_samples, batch_size), desc="Baseline + Unpatched"):
    end = min(start + batch_size, num_samples)

    baseline_inputs_batch = baseline_inputs[start:end]
    unpatched_inputs_batch = unpatched_inputs[start:end]

    # Run baseline batch on batch slice only
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
    # free_unused_cuda_memory()

print("Done with all examples.")

#%%
print(board)
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
    title=f"ground truth token surprisal (↓ better), chess, {NUM_MOVES} moves, {NUM_GAMES*NUM_SAMPLES_PER_GAME} examples = {NUM_SAMPLES_PER_GAME}q x {NUM_GAMES}g, model: {MODEL_ID}",
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
idx = 243
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

# %%
print("\n--- nnsight generate baseline continuation (first example) ---")
prompt = baseline_inputs[0]
n_new_tokens = 5
with model.generate(prompt, max_new_tokens=n_new_tokens, temperature=TEMPERATURE, top_p=TOP_P) as tracer:
    out = model.generator.output.save()

decoded_prompt = model.tokenizer.decode(out[0][0:-n_new_tokens].cpu())
decoded_answer = model.tokenizer.decode(out[0][-n_new_tokens:].cpu())

print("Prompt:", decoded_prompt)
print("Generated continuation:", decoded_answer)

# %%
print(len(patched_results[0]))
# %%
