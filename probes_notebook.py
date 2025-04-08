# %% [markdown]
# # Probing LLM Activations using Activation Patching
#
# **Goal:** Investigate whether specific ground-truth information is encoded within LLM activations by patching those activations onto a neutral context and asking questions.
#
# **Methodology:**
# 1.  **Setting Definition:** Choose a domain with ground-truth information (e.g., Chess PGN, NLP Statements).
# 2.  **Activation Extraction:** Run the LLM on original contexts from the chosen domain and extract relevant hidden state activations (`x`).
# 3.  **Patching Input:** Create inputs of the form `[filler token] [question]`.
# 4.  **Activation Patching:** Run the LLM on the patching input, but replace the activation at the `[filler token]` position with the extracted activation `x`.
# 5.  **Evaluation (Patched):** Evaluate the LLM's 0-shot accuracy in answering the `[question]` based *only* on the patched activation `x`.
# 6.  **Evaluation (Context Baseline):** Evaluate the LLM's 0-shot accuracy on `[original context] [question]` as a baseline.
# 7.  **Analysis:** Compare patched accuracy vs. baseline accuracy.

# %% [markdown]
# ## 1. Setup and Configuration

# %%
# Core libraries
import torch
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


# %%
# --- Configuration ---

# Model Selection
# Start small (e.g., gpt2, pythia-70m), then scale up
MODEL_ID = "openai-community/gpt2"
#MODEL_ID = "EleutherAI/pythia-70m"
# MODEL_ID = "meta-llama/Meta-Llama-3.1-8B" # Requires HF token + access

# Execution Environment
# Set to True to use NDIF for large models
USE_REMOTE_EXECUTION = False
# NDIF_API_KEY = "YOUR_NDIF_API_KEY" # Load from secrets or env variable
# HF_TOKEN = "YOUR_HF_TOKEN" # Required for gated models like Llama

# Activation Targeting
# Example: Last layer's residual stream
TARGET_LAYER = 11# Use negative indexing for end layers
# Use nnsight's module access syntax. Print model structure to find paths.
# Examples:
#GPT2: f"model.transformer.h.{TARGET_LAYER}.output.0" #(Residual stream output of block)
# GPT2: model.transformer.h.{TARGET_LAYER}.mlp.output[0] (MLP output)
# Llama3: model.layers.{TARGET_LAYER}.output[0] (Residual stream)
# Llama3: model.layers.{TARGET_LAYER}.mlp.output (MLP output)
# Llama3: model.layers[TARGET_LAYER].self_attn.o_proj.input (Attention output value - see tutorial)
#TARGET_MODULE_PATH = f"transformer.h[{TARGET_LAYER}].output[0]" # Adjust based on model arch
TARGET_MODULE_PATH = f"transformer.h.{TARGET_LAYER}.mlp" #(Residual stream output of block)
# TARGET_MODULE_PATH = f"model.layers[{TARGET_LAYER}].output[0]" # For Llama

# Activation Position
# Often the last token of the original context
TARGET_TOKEN_IDX = -1

# Patching Configuration
FILLER_TOKEN = "<|endoftext|>" # Choose a token unlikely to strongly affect things
# FILLER_TOKEN = "..." # Or something else simple

# Experiment Setting ('chess' or 'nlp_sentiment', 'nlp_truth', etc.)
SETTING = 'chess' # Or 'nlp_sentiment', etc.

# Evaluation
NUM_SAMPLES = 50 # Number of data points to process
BATCH_SIZE = 10 # For batching extraction/patching if desired
print(TARGET_MODULE_PATH)
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
filler_token_id = tokenizer.encode(FILLER_TOKEN)[0]
print(f"Using filler token: '{FILLER_TOKEN}' (ID: {filler_token_id})")

# Resolve the target module dynamically based on the path string
# This uses nnsight's ability to access modules via attribute strings
print(f"Attempting to target module: {TARGET_MODULE_PATH}")
target_module = util.fetch_attr(model, TARGET_MODULE_PATH)
print(f"Targeting module: {TARGET_MODULE_PATH}")

# %%
print(model.transformer.h[11])
# %% [markdown]
# ## 2. Setting Definition and Data Preparation
#
# Implement data loading and ground-truth extraction for the chosen `SETTING`.

# %%
# --- Data Loading and Preparation ---

original_contexts = []
questions = []
ground_truths = [] # Store the expected correct answer (e.g., token ID, string label)
ground_truth_token_ids = [] # Store the token ID of the correct answer word/phrase

if SETTING == 'chess':
    print("Preparing Chess PGN data...")
    # Example: Load a PGN file and extract board states + formulate questions
    # This is a simplified example. A real implementation would need more robust
    # PGN parsing and question generation.
    try:
        # Replace with your PGN data source
        # pgn_file = open("path/to/your/games.pgn")
        # For demonstration, using a simple PGN string:
        # pgn_string = """
        # [Event "Example Game"]
        # [Site "?"]
        # [Date "????.??.??"]
        # [Round "?"]
        # [White "?"]
        # [Black "?"]
        # [Result "*"]
        pgn_string = """
        1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *
        """
        import io
        pgn_file = io.StringIO(pgn_string)

        game = chess.pgn.read_game(pgn_file)
        board = game.board()
        print(board)
        node = game

        count = 0
        print(dir(game))
        while node.variations and count < NUM_SAMPLES:
            next_node = node.variations[0]
            move = next_node.move
            board.push(move)
            node = next_node

            # Example context: PGN string up to current move
            exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
            pgn_context = game.accept(exporter)
            original_contexts.append(pgn_context.strip())

            # Example question: What piece is on square X?
            # Choose a random occupied square
            occupied_squares = [sq for sq in chess.SQUARES if board.piece_at(sq)]
            if not occupied_squares: continue
            target_square = random.choice(occupied_squares)
            square_name = chess.square_name(target_square)
            question_text = f"What piece is on {square_name}?"
            questions.append(question_text)

            # Ground truth
            piece = board.piece_at(target_square)
            piece_name = chess.piece_name(piece.piece_type).capitalize()
            # Need to map this name to the expected token(s)
            # This mapping can be tricky! Start simple. Assume single token answer.
            # Add space for common tokenization practice
            correct_answer_str = f" {piece_name}"
            ground_truths.append(correct_answer_str)
            # Get the token ID - handle potential multi-token answers if needed
            gt_token_id = tokenizer.encode(correct_answer_str, add_special_tokens=False)
            if len(gt_token_id) == 1:
                 ground_truth_token_ids.append(gt_token_id[0])
            else:
                 # Handle multi-token answers - skip for simplicity now
                 original_contexts.pop()
                 questions.pop()
                 ground_truths.pop()
                 print(f"Skipping multi-token answer: {correct_answer_str}")
                 continue # Skip this sample

            count += 1

        pgn_file.close()
        print(f"Loaded {len(original_contexts)} chess positions and questions.")

    except FileNotFoundError:
        print("Error: PGN data file not found.")
    except Exception as e:
        print(f"Error processing PGN data: {e}")


elif SETTING == 'nlp_sentiment':
    print("Preparing NLP Sentiment data (e.g., SST-2)...")
    try:
        dataset = load_dataset("sst2", split='validation') # Use validation set
        dataset = dataset.shuffle(seed=42).select(range(min(NUM_SAMPLES, len(dataset))))

        sentiment_map = {0: " Negative", 1: " Positive"} # Add space for tokenization

        for example in dataset:
            original_contexts.append(example['sentence'])
            questions.append("What is the sentiment?") # Fixed question
            gt_label = example['label']
            gt_text = sentiment_map[gt_label]
            ground_truths.append(gt_text)
            # Assuming single token answers " Positive" / " Negative"
            gt_token_id = tokenizer.encode(gt_text, add_special_tokens=False)
            if len(gt_token_id) >0:
                #ground_truth_token_ids.append(gt_token_id[0])
                ground_truth_token_ids.append(gt_token_id)
            else:
                print(f"Warning: Sentiment label '{gt_text}' is multi-token. Check tokenization.")
                # Decide how to handle: skip, use first token, etc. Skipping for now.
                original_contexts.pop()
                questions.pop()
                ground_truths.pop()
                continue

        print(f"Loaded {len(original_contexts)} sentiment analysis samples.")

    except Exception as e:
        print(f"Error loading/processing dataset: {e}")

elif SETTING == 'nlp_truth':
    print("Preparing NLP Truthfulness data (e.g., BoolQ)...")
    # Example using BoolQ - requires careful question formulation
    try:
        dataset = load_dataset("boolq", split='validation')
        dataset = dataset.shuffle(seed=42).select(range(min(NUM_SAMPLES, len(dataset))))

        truth_map = {True: " Yes", False: " No"} # Add space

        for example in dataset:
            # BoolQ context is usually long (passage). Use question as context?
            # Or use passage and ask 'Is the following statement true: [question]'?
            # Let's use the provided question as the primary 'context' for simplicity
            original_contexts.append(example['question'])
            questions.append("Is this statement true or false?") # Or simply "True or False?"
            gt_label = example['answer']
            gt_text = truth_map[gt_label]
            ground_truths.append(gt_text)

            gt_token_id = tokenizer.encode(gt_text, add_special_tokens=False)
            if len(gt_token_id) == 1:
                ground_truth_token_ids.append(gt_token_id[0])
            else:
                print(f"Warning: Truth label '{gt_text}' is multi-token. Check tokenization.")
                original_contexts.pop()
                questions.pop()
                ground_truths.pop()
                continue

        print(f"Loaded {len(original_contexts)} truthfulness samples.")
    except Exception as e:
        print(f"Error loading/processing dataset: {e}")

else:
    raise ValueError(f"Unknown SETTING: {SETTING}")

# Ensure all lists have the same length
assert len(original_contexts) == len(questions) == len(ground_truths) == len(ground_truth_token_ids), \
    "Data preparation resulted in mismatched list lengths!"
print(f"Final dataset size: {len(original_contexts)}")

# %%
print(util.fetch_attr(model, "transformer.h.11.mlp"))
# %% [markdown]
# ## 3. Activation Extraction
#
# Run the model on the `original_contexts` and save the target activations.

# %%
# --- Activation Extraction ---

extracted_activations = {}  # Dictionary to store activations from all layers
print(f"Extracting activations from all layers at token index {TARGET_TOKEN_IDX}...")

# Process each context individually without batching
for i in tqdm(range(len(original_contexts)), desc="Extracting Activations"):
    context = original_contexts[i]
    
    # Use nnsight's trace context
    with model.trace(context, remote=USE_REMOTE_EXECUTION, scan=False, validate=False) as tracer:
        # Extract activations from all layers
        # Initialize dictionary for this example if first iteration
        if i == 0:
            # Get all module paths we want to extract from
            num_layers = len(model.transformer.h)
            for layer_idx in range(num_layers):
                # Track MLP outputs
                extracted_activations[f"transformer.h.{layer_idx}.mlp"] = []
                # Track attention outputs
                extracted_activations[f"transformer.h.{layer_idx}.attn"] = []
        print(extracted_activations)
        
        # Extract activations from each layer
        for layer_path in extracted_activations.keys():
            # Safely fetch the module's output
            module = util.fetch_attr(model, layer_path)
            activation_proxy = module.output
            
            # Select the specific token index (typically the last token)
            if isinstance(activation_proxy, Proxy):
                # For a single example, batch dimension is 1
                target_activation = activation_proxy[0, TARGET_TOKEN_IDX, :].save()
            else:
                target_activation = activation_proxy[0, TARGET_TOKEN_IDX, :].save()
            
            # Append the activation for this example to the appropriate list
            extracted_activations[layer_path].append(target_activation)

print(f"Extracted activations from {len(extracted_activations)} modules.")
if extracted_activations:
    sample_key = list(extracted_activations.keys())[0]
    print(f"Shape of first activation vector from {sample_key}: {extracted_activations[sample_key][0].shape}")

# %% [markdown]
# ## 4. Patching Experiment
print(f"Sample activation: {list(extracted_activations.keys())[0]}, shape: {extracted_activations[list(extracted_activations.keys())[0]][0].shape}")
#
# Run the model on `[filler token] [question]`, patching activations from all layers onto the filler token. Also run the baseline `[original context] [question]`.

# %%
# --- Patching Experiment ---

patched_results = []  # Store model's prediction after patching
baseline_results = []  # Store model's prediction from original context

# Combine filler token and question for patching input
patching_inputs = [f"{FILLER_TOKEN}{q}" for q in questions]

# Combine original context and question for baseline input
baseline_inputs = [f"{ctx}{q}" for ctx, q in zip(original_contexts, questions)]

# Find the index of the filler token in the tokenized patching inputs
tokenized_filler_check = tokenizer(patching_inputs[0], return_tensors="pt")['input_ids'][0]
# Simple check: assume it's the first non-BOS token if BOS exists
filler_token_idx_in_patched_input = 1 if tokenizer.bos_token_id == tokenized_filler_check[0] else 0
print(f"Assuming filler token is at index: {filler_token_idx_in_patched_input}")

# Process each example individually without batching
for i in tqdm(range(len(original_contexts)), desc="Running Patching/Baseline"):
    # Get single example data
    patching_input = patching_inputs[i]
    baseline_input = baseline_inputs[i]
    
    # Process patched run
    with model.trace(remote=USE_REMOTE_EXECUTION, scan=False, validate=False) as tracer:
        # --- Patched Run ---
        with tracer.invoke(patching_input) as invoker_patched:
            # Patch activations from all layers
            for layer_path, activations in extracted_activations.items():
                # Access the target module's output where patching should occur
                module = util.fetch_attr(model, layer_path)
                module_to_patch = module.output
                
                # Perform the patch: Set the activation at the filler token position
                module_to_patch[0, filler_token_idx_in_patched_input, :] = activations[i]
            
            # Get the final logits from the patched run
            patched_logits = model.lm_head.output[0, -1, :]  # Logits for the last token
            # Get the predicted token ID (argmax)
            patched_prediction = torch.argmax(patched_logits, dim=-1).save()
        
        # --- Baseline Run ---
        with tracer.invoke(baseline_input) as invoker_baseline:
            # No patching needed here, just get the output
            baseline_logits = model.lm_head.output[0, -1, :]  # Logits for the last token
            baseline_prediction = torch.argmax(baseline_logits, dim=-1).save()
    
    # Store results for this example
    patched_results.append(patched_prediction.item())
    baseline_results.append(baseline_prediction.item())

print("Patching and baseline runs complete.")
print(f"Collected {len(patched_results)} patched results and {len(baseline_results)} baseline results.")

# %% [markdown]
# ## 5. Evaluation
#
# Compare the model's predictions (patched and baseline) against the ground truth.
print(patched_results)
print(baseline_results)
for i in range(len(ground_truth_token_ids)):
    print(tokenizer.decode([ground_truth_token_ids[i]]))    
for i in range(len(ground_truth_token_ids)):
    print(tokenizer.decode([patched_results[i]]))
for i in range(len(ground_truth_token_ids)):
    print(tokenizer.decode([baseline_results[i]]))
# %%
# --- Evaluation ---

patched_correct = 0
baseline_correct = 0
num_evaluated = len(ground_truth_token_ids) # Use the count after filtering/preparation

if num_evaluated != len(patched_results) or num_evaluated != len(baseline_results):
     print(f"Warning: Mismatch in number of results and ground truths. Evaluated: {num_evaluated}, Patched: {len(patched_results)}, Baseline: {len(baseline_results)}")
     # Adjust evaluation range if necessary
     num_evaluated = min(num_evaluated, len(patched_results), len(baseline_results))
     print(f"Evaluating based on {num_evaluated} samples.")


print("\n--- Sample Results ---")
for i in range(min(num_evaluated, 10)): # Print first 10 samples
    gt_token_id = ground_truth_token_ids[i]
    pt_pred_token_id = patched_results[i]
    bl_pred_token_id = baseline_results[i]

    gt_token_str = tokenizer.decode([gt_token_id])
    pt_pred_token_str = tokenizer.decode([pt_pred_token_id])
    bl_pred_token_str = tokenizer.decode([bl_pred_token_id])

    patched_match = (pt_pred_token_id == gt_token_id)
    baseline_match = (bl_pred_token_id == gt_token_id)

    if patched_match:
        patched_correct += 1
    if baseline_match:
        baseline_correct += 1

    print(f"\nSample {i+1}:")
    print(f"  Context: {original_contexts[i][:80]}...") # Show start of context
    print(f"  Question: {questions[i]}")
    print(f"  Ground Truth: '{gt_token_str}' (ID: {gt_token_id})")
    print(f"  Patched Pred: '{pt_pred_token_str}' (ID: {pt_pred_token_id}) --> {'Correct' if patched_match else 'Incorrect'}")
    print(f"  Baseline Pred: '{bl_pred_token_str}' (ID: {bl_pred_token_id}) --> {'Correct' if baseline_match else 'Incorrect'}")


# Calculate overall accuracy
patched_accuracy = (patched_correct / num_evaluated) * 100 if num_evaluated > 0 else 0
baseline_accuracy = (baseline_correct / num_evaluated) * 100 if num_evaluated > 0 else 0

print("\n--- Overall Accuracy ---")
print(f"Total Samples Evaluated: {num_evaluated}")
print(f"Patched Accuracy (Answer from Activation): {patched_accuracy:.2f}% ({patched_correct}/{num_evaluated})")
print(f"Baseline Accuracy (Answer from Context): {baseline_accuracy:.2f}% ({baseline_correct}/{num_evaluated})")


# %% [markdown]
# ## 6. Analysis and Discussion
#
# - How does the patched accuracy compare to the baseline?
# - Does patched accuracy significantly exceed random chance (1 / vocab_size or 1 / num_possible_answers)?
# - What does a high patched accuracy imply about the information encoded in the specific activation?
# - How does accuracy vary across layers or module types (if experimented)?
# - What are the limitations (e.g., single token answers, complexity of questions, specific model/setting)?

# %% [markdown]
# --- Analysis ---
#
# The comparison between the **Patched Accuracy** and the **Baseline Accuracy** is key.
#
# *   **If Patched Accuracy is high (significantly above chance) and close to Baseline Accuracy:** This suggests the targeted activation `x` strongly encodes the information needed to answer the question, almost as well as having the full original context. The model can effectively "read" the information directly from the hidden state.
#
# *   **If Patched Accuracy is high but lower than Baseline Accuracy:** The activation contains relevant information, but the model still benefits from attending to the full original context (perhaps for nuance, confirmation, or retrieving related facts not solely in `x`).
#
# *   **If Patched Accuracy is low (near chance):** The specific activation `x` (at that layer/token position) does *not* seem to encode the necessary information in a way the model can readily use when presented in this patched format. The information might be encoded elsewhere, distributed across multiple activations, or require the structure of the original context to be interpreted.
#
# *   **Interesting Case: Patched Accuracy > Baseline Accuracy:** This would be unexpected but could potentially happen if the original context is somehow misleading or confusing for the specific question, while the isolated activation provides a clearer signal. This warrants further investigation.
#
# Consider the specific `SETTING`:
# *   For Chess: Does the activation encode the full board state, or just local information? Can it answer "Is White winning?" or only "What piece is on e4?"?
# *   For NLP Sentiment/Truth: Is the signal purely the classification label, or does it retain nuances of the original statement?
#
# Further experiments could involve:
# *   Varying `TARGET_LAYER` and `TARGET_MODULE_PATH`.
# *   Trying different `FILLER_TOKEN`s.
# *   Using more complex questions.
# *   Probing attention head outputs instead of residual stream/MLP outputs.
# *   Comparing different model sizes or families.
# *   Using techniques like attribution patching (as mentioned in the tutorial) for potentially faster analysis across many components, although it's an approximation.

# %% [markdown]
# --- End of Notebook ---