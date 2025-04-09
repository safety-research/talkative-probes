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
import gc
def free_unused_cuda_memory():
    """Free unused cuda memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        raise RuntimeError("not using cuda")
    gc.collect()

# %% [markdown]
# ## 1. Setup and Configuration
import sys
print(sys.executable)
print(sys.version)
import os
# Get the SLURM environment variables
slurm_gpus = os.environ.get('SLURM_GPUS_ON_NODE', '1')
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


# %%
# --- Configuration ---

# Model Selection
# Start small (e.g., gpt2, pythia-70m), then scale up
#MODEL_ID = "openai-community/gpt2"
#MODEL_ID = "openai-community/gpt2-xl"
#MODEL_ID = "meta-llama/Llama-2-13b-hf"
MODEL_ID = "google/gemma-3-27b-pt"
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

# Evaluation
NUM_SAMPLES = 200#50 # Number of data points to process
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

# Resolve the target module dynamically based on the path string
# # This uses nnsight's ability to access modules via attribute strings
# print(f"Attempting to target module: {TARGET_MODULE_PATH}")
# target_module = util.fetch_attr(model, TARGET_MODULE_PATH)
# print(f"Targeting module: {TARGET_MODULE_PATH}")
if model.config.model_type == "gpt2":
    num_layers = len(model.transformer.h)
elif model.config.model_type == "llama":
    num_layers = len(model.layers)
elif model.config.model_type == "gemma3":
    num_layers = model.config.text_config.num_hidden_layers
print(f"{model.config.model_type} has {num_layers} layers")


def get_layer_output_path(layer_idx):
    if model.config.model_type == "gpt2":
        return f"transformer.h.{layer_idx}"
    elif model.config.model_type == "llama":
        return f"layers.{layer_idx}"
    elif model.config.model_type == "gemma3":
        return f"language_model.model.layers.{layer_idx}"
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
        # For demonstration, using a simple PGN string
        pgn_string = """
        1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *
        """
        import io
        pgn_file = io.StringIO(pgn_string)

        game = chess.pgn.read_game(pgn_file)
        board = game.board()
        
        # Process all moves to reach the end state
        node = game
        while node.variations:
            next_node = node.variations[0]
            move = next_node.move
            board.push(move)
            node = next_node
        
        # Now we have the end state board
        count = 0
        while count < NUM_SAMPLES:
            # Generate questions only about the final board position
            occupied_squares = [sq for sq in chess.SQUARES if board.piece_at(sq)]
            if not occupied_squares:
                break
                
            target_square = random.choice(occupied_squares)
            square_name = chess.square_name(target_square)
            question_text = f"What chess piece is on {square_name}?"
            
            # Get the ground truth answer
            piece = board.piece_at(target_square)
            piece_name = chess.piece_name(piece.piece_type).capitalize()
            correct_answer_str = f" {piece_name}".lower()
            
            # Create context: PGN string of the full game
            exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
            pgn_context = game.accept(exporter)
            
            # Add to our dataset
            dataset['original_contexts'].append(pgn_context.strip())
            dataset['questions'].append(format_question(question_text))
            dataset['ground_truths'].append(correct_answer_str)
            dataset['extra_info'].append(square_name)
            
            # Convert to token IDs
            gt_token_id = tokenizer.encode(correct_answer_str, add_special_tokens=False)
            if len(gt_token_id) == 1:
                dataset['ground_truth_token_ids'].append(gt_token_id[0])
            else:
                # For multi-token answers
                dataset['ground_truth_token_ids'].append(gt_token_id)
                # Skip if we need single token answers
                if MAX_NEW_TOKENS == 1:
                    dataset['original_contexts'].pop()
                    dataset['questions'].pop()
                    dataset['ground_truths'].pop()
                    dataset['ground_truth_token_ids'].pop()
                    continue
            
            count += 1
            
        pgn_file.close()
        print(f"Loaded {len(dataset['original_contexts'])} chess positions and questions about the end state.")

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
print(model)

# # %%
# print(util.fetch_attr(model, "transformer.h.11.mlp"))
# # Run the model on the `original_contexts` and save the target activations.
# from nnsight import LanguageModel

# model = LanguageModel('openai-community/gpt2', device_map='auto')

# with model.trace('The Eiffel Tower is in the city of') as tracer:

#       hidden_states = model.transformer.h[-1].output[0].save()
#       print((model.transformer.h[-1].output))
      

#       output = model.output[0].save()
# %% [markdown]
# ## 3. Activation Extraction
print(dataset['original_contexts'])
print(tokenizer.encode(dataset['original_contexts'][0]))
# --- Activation Extraction ---
#print(model.transformer.h[-1].output.shape)

# --- Modularized Activation Extraction ---
def extract_activations_for_example(context, model, target_token_idx):
    activations = {}
    with model.trace(context, remote=USE_REMOTE_EXECUTION, scan=False, validate=False) as tracer:
        for layer_idx in range(num_layers):
            layer_path = get_layer_output_path(layer_idx)
            module = util.fetch_attr(model, layer_path)
            proxy = module.output
            if isinstance(proxy, Proxy):
                act = proxy[0][0, target_token_idx, :].save()
            else:
                act = proxy[0][0, target_token_idx, :].save()
            activations[layer_path] = act

    return activations

# --- Modularized Patching ---
def run_patching(patching_input, activations, model, filler_token_idx):
    topkn = 100
    with model.trace(patching_input, remote=USE_REMOTE_EXECUTION) as patched_tracer:
        for layer_path, act in activations.items():
            module = util.fetch_attr(model, layer_path)
            module.output[0][0, filler_token_idx, :] = act
        logits = model_lmhead.output[0, -1, :]
        patched_pred = torch.argmax(logits, dim=-1).save()
        sorted_idx = torch.argsort(logits, descending=True)
        topk_idx = sorted_idx[:topkn]
        topk_logits = logits[topk_idx]
        topk_logprobs = torch.log_softmax(logits, dim=-1)[topk_idx]
        patched_result = (patched_pred, topk_idx.save(), topk_logits.save(), topk_logprobs.save())
    patched_result = tuple(p.detach() for p in patched_result)
    return patched_result

# --- Modularized Baseline ---
def run_baseline(baseline_input, model):
    topkn = 100
    with model.trace(baseline_input, remote=USE_REMOTE_EXECUTION) as baseline_tracer:
        logits = model_lmhead.output[0, -1, :]
        baseline_pred = torch.argmax(logits, dim=-1).save()
        sorted_idx = torch.argsort(logits, descending=True)
        topk_idx = sorted_idx[:topkn]
        topk_logits = logits[topk_idx]
        topk_logprobs = torch.log_softmax(logits, dim=-1)[topk_idx]
        baseline_result = (baseline_pred, topk_idx.save(), topk_logits.save(), topk_logprobs.save())
    baseline_result = tuple(b.detach() for b in baseline_result)
    return baseline_result

# --- Main Loop ---

# %% [markdown]
# # ## 4. Patching Experiment
# print(f"Sample activation: {list(extracted_activations.keys())[0]}, shape: {extracted_activations[list(extracted_activations.keys())[0]][0].shape}")
# #
# # Run the model on `[filler token] [question]`, patching activations from all layers onto the filler token. Also run the baseline `[original context] [question]`.
# %%
print(os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'))
torch.cuda.empty_cache()
#%%
# --- Patching Experiment ---

# patched_outputs = []
# patched_logits_all = []
# baseline_outputs = []
# baseline_logits_all = []
# unpatched_outputs = []
# unpatched_logits_all = []

patching_inputs = [f"{FILLER_TOKEN}{q}" for q in dataset['questions']]
baseline_inputs = [f"We are playing chess. The PGN is: {ctx}\n{q}" for ctx, q in zip(dataset['original_contexts'], dataset['questions'])]
unpatched_inputs = patching_inputs.copy()

bos_or_filler_token_idx = tokenizer(patching_inputs[0], return_tensors="pt")['input_ids'][0]
print("bos_or_filler_token_idx", bos_or_filler_token_idx, tokenizer.bos_token_id)
filler_token_idx_in_patched_input = 1 if (tokenizer.bos_token_id == bos_or_filler_token_idx[0]) else 0
print(f"Assuming filler token is at index: {filler_token_idx_in_patched_input}, it translates to: {tokenizer.decode([bos_or_filler_token_idx[filler_token_idx_in_patched_input]])}")

if model.config.model_type == "gemma3":
    model_lmhead = model.language_model.lm_head
else:
    model_lmhead = model.lm_head

patched_results = []
baseline_results = []
unpatched_results = []

# First loop: extract activations and run patching
for i in tqdm(range(len(dataset['original_contexts'])), desc="Extract + Patch"):
    free_unused_cuda_memory()

    context = dataset['original_contexts'][i]
    question = dataset['questions'][i]

    patching_input = f"{FILLER_TOKEN}{question}"
    baseline_input = f"We are playing chess. The PGN is: {context}\n{question}"

    # Extract activations for this example
    activations = extract_activations_for_example(context, model, TARGET_TOKEN_IDX)
    patched_res = run_patching(
        patching_input, activations, model, filler_token_idx_in_patched_input
    )
    baseline_res = run_baseline(
        baseline_input, model
    )

    patched_results.append(patched_res)
    baseline_results.append(baseline_res)

    #del activations
# %%
# Second loop: run unpatched control
for i in tqdm(range(len(dataset['original_contexts'])), desc="Unpatched control"):
    free_unused_cuda_memory()

    unpatched_input = f"{FILLER_TOKEN}{dataset['questions'][i]}"

    with model.trace(unpatched_input, remote=USE_REMOTE_EXECUTION) as unpatched_tracer:
        logits = model_lmhead.output[0, -1, :]
        unpatched_pred = torch.argmax(logits, dim=-1).save()
        sorted_idx = torch.argsort(logits, descending=True)
        topk_idx = sorted_idx[:100]
        topk_logits = logits[topk_idx]
        topk_logprobs = torch.log_softmax(logits, dim=-1)[topk_idx]
        unpatched_results.append((unpatched_pred, topk_idx.save(), topk_logits.save(), topk_logprobs.save()))
    unpatched_results[-1]=tuple(u.detach() for u in unpatched_results[-1])
        
    # free_unused_cuda_memory()

print("Done with all examples.")

# # %%
# print(unpatched_results[0])
# # %%
# for i in tqdm(range(len(dataset['original_contexts'])), desc="Running Patching/Baseline"):
#     free_unused_cuda_memory()
#     # Get single example data
#     patching_input = patching_inputs[i]
#     baseline_input = baseline_inputs[i]
#     if i == 0:
#         print("patching_input: '", patching_input, "'")
#         print("baseline_input: '", baseline_input, "'")
#         print("check patching: ", tokenizer.decode(tokenizer.encode(patching_input)))
#         print("check baseline: ", tokenizer.decode(tokenizer.encode(baseline_input)))
#         print(f"donor token: ' {tokenizer.decode(tokenizer.encode(dataset['original_contexts'][i])[TARGET_TOKEN_IDX])}' at position {TARGET_TOKEN_IDX} in the original context")
#         print(f"recipient token: '{tokenizer.decode(filler_token_id)}' at position {filler_token_idx_in_patched_input} in the patching input")
    
#     #--- Patched Run ---
#     # Use separate trace calls for patched and baseline to avoid interleaving issues
#     with model.trace(patching_input, remote=USE_REMOTE_EXECUTION) as patched_tracer:
#         # Patch activations from all layers
#         for layer_path, activations in extracted_activations.items():
#             # Access the target module's output where patching should occur
#             module = util.fetch_attr(model, layer_path)
#             module_to_patch = module.output
            
#             # Perform the patch: Set the activation at the filler token position
#             module_to_patch[0][0, filler_token_idx_in_patched_input, :] = 1*activations[i]#*(-1000000)
        
#         # For single token generation (MAX_NEW_TOKENS = 1)
#         if MAX_NEW_TOKENS == 1:
#             # Get the final logits from the patched run
#             patched_logits = model_lmhead.output[0, -1, :]  # Logits for the last token
#             # Get the predicted token ID (argmax)
#             patched_prediction = torch.argmax(patched_logits, dim=-1).save()
#             patched_outputs.append([patched_prediction])
#             # Save indices, logits, and logprobs of the top 100 tokens
#             sorted_indices = torch.argsort(patched_logits, descending=True)
#             topk_indices = sorted_indices[:100]
#             topk_logits = patched_logits[topk_indices]
#             topk_logprobs = torch.log_softmax(patched_logits, dim=-1)[topk_indices]
#             patched_logits_all.append((topk_indices.save(), topk_logits.save(), topk_logprobs.save()))
#         else:
#             # For multi-token generation, we need to save the initial logits
#             # then continue generating additional tokens
            
#             # Get logits for last position
#             initial_logits = model_lmhead.output[0, -1, :].detach()
            
#             # Sample first token (using temperature and top_p)
#             if TEMPERATURE > 0:
#                 # Apply temperature
#                 initial_logits = initial_logits / TEMPERATURE
#                 # Apply top_p sampling
#                 sorted_logits, sorted_indices = torch.sort(initial_logits, descending=True)
#                 cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=0), dim=0)
#                 sorted_indices_to_remove = cumulative_probs > TOP_P
#                 sorted_indices_to_remove[0] = False  # Keep at least the highest prob token
#                 indices_to_remove = sorted_indices_to_remove.scatter(
#                     0, sorted_indices, sorted_indices_to_remove
#                 )
#                 initial_logits[indices_to_remove] = -float('inf')
#                 # Sample from the filtered distribution
#                 probs = torch.softmax(initial_logits, dim=0)
#                 first_token = torch.multinomial(probs, 1).item()
#             else:
#                 # Greedy sampling
#                 first_token = torch.argmax(initial_logits, dim=0).item()
            
#             # Initialize the sequence with the first token
#             generated_tokens = [first_token]
            
#             # Continue generating tokens one by one (autoregressive)
#             current_input = tokenizer.decode(tokenizer.encode(patching_input) + [first_token], skip_special_tokens=False)
            
#             # Generate additional tokens (MAX_NEW_TOKENS - 1)
#             for _ in range(MAX_NEW_TOKENS - 1):
#                 # We need to run a separate trace for each additional token
#                 with model.trace(current_input, remote=USE_REMOTE_EXECUTION) as next_token_tracer:
#                     # Get logits for the last token position
#                     next_logits = model_lmhead.output[0, -1, :]
                
#                 # Apply temperature and top_p sampling
#                 next_logits = next_logits.detach()
#                 if TEMPERATURE > 0:
#                     next_logits = next_logits / TEMPERATURE
#                     # Apply top_p sampling (same as above)
#                     sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
#                     cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=0), dim=0)
#                     sorted_indices_to_remove = cumulative_probs > TOP_P
#                     sorted_indices_to_remove[0] = False
#                     indices_to_remove = sorted_indices_to_remove.scatter(
#                         0, sorted_indices, sorted_indices_to_remove
#                     )
#                     next_logits[indices_to_remove] = -float('inf')
#                     probs = torch.softmax(next_logits, dim=0)
#                     next_token = torch.multinomial(probs, 1).item()
#                 else:
#                     next_token = torch.argmax(next_logits, dim=0).item()
                
#                 # Add the new token to the sequence
#                 generated_tokens.append(next_token)
                
#                 # Update input for next iteration
#                 current_input = tokenizer.decode(
#                     tokenizer.encode(patching_input) + generated_tokens, 
#                     skip_special_tokens=False
#                 )
            
#             patched_outputs.append(generated_tokens.save())
#         # --- Baseline Run ---
#     with model.trace(baseline_input, remote=USE_REMOTE_EXECUTION) as baseline_tracer:
#         # For single token generation
#         if MAX_NEW_TOKENS == 1:
#             # Get the final logits from the baseline run
#             baseline_logits = model_lmhead.output[0, -1, :]  # Logits for the last token
#             baseline_prediction = torch.argmax(baseline_logits, dim=-1).save()
#             baseline_outputs.append([baseline_prediction])
#             # Save indices, logits, and logprobs of the top 100 tokens
#             sorted_indices = torch.argsort(baseline_logits, descending=True)
#             topk_indices = sorted_indices[:100]
#             topk_logits = baseline_logits[topk_indices]
#             topk_logprobs = torch.log_softmax(baseline_logits, dim=-1)[topk_indices]
#             baseline_logits_all.append((topk_indices.save(), topk_logits.save(), topk_logprobs.save()))
#         else:
#             # For multi-token generation
#             # Using similar approach as with patched input
            
#             # Get logits for last position 
#             initial_logits = model_lmhead.output[0, -1, :].detach()
            
#             # Sample first token with temperature and top_p
#             if TEMPERATURE > 0:
#                 initial_logits = initial_logits / TEMPERATURE
#                 # Apply top_p sampling
#                 sorted_logits, sorted_indices = torch.sort(initial_logits, descending=True)
#                 cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=0), dim=0)
#                 sorted_indices_to_remove = cumulative_probs > TOP_P
#                 sorted_indices_to_remove[0] = False
#                 indices_to_remove = sorted_indices_to_remove.scatter(
#                     0, sorted_indices, sorted_indices_to_remove
#                 )
#                 initial_logits[indices_to_remove] = -float('inf')
#                 probs = torch.softmax(initial_logits, dim=0)
#                 first_token = torch.multinomial(probs, 1)
#             else:
#                 first_token = torch.argmax(initial_logits, dim=0)
            
#             # Initialize sequence with first token
#             generated_tokens = [first_token]
            
#             # Continue generating tokens
#             current_input = tokenizer.decode(tokenizer.encode(baseline_input) + [first_token], skip_special_tokens=False)
            
#             # Generate additional tokens
#             for _ in range(MAX_NEW_TOKENS - 1):
#                 with model.trace(current_input, remote=USE_REMOTE_EXECUTION) as next_token_tracer:
#                     next_logits = model_lmhead.output[0, -1, :]
                
#                 # Apply temperature and top_p sampling
#                 next_logits = next_logits.detach()
#                 if TEMPERATURE > 0:
#                     next_logits = next_logits / TEMPERATURE
#                     # Apply top_p sampling
#                     sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
#                     cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=0), dim=0)
#                     sorted_indices_to_remove = cumulative_probs > TOP_P
#                     sorted_indices_to_remove[0] = False
#                     indices_to_remove = sorted_indices_to_remove.scatter(
#                         0, sorted_indices, sorted_indices_to_remove
#                     )
#                     next_logits[indices_to_remove] = -float('inf')
#                     probs = torch.softmax(next_logits, dim=0)
#                     next_token = torch.multinomial(probs, 1)
#                 else:
#                     next_token = torch.argmax(next_logits, dim=0)
                
#                 # Add new token to sequence
#                 generated_tokens.append(next_token)
                
#                 # Update input for next iteration
#                 current_input = tokenizer.decode(
#                     tokenizer.encode(baseline_input) + generated_tokens, 
#                     skip_special_tokens=False
#                 )
            
#             baseline_outputs.append(generated_tokens)
            
#     # --- Unpatched Run (same input as patching but without patching) ---
#     with model.trace(unpatched_inputs[i], remote=USE_REMOTE_EXECUTION) as unpatched_tracer:
#         if MAX_NEW_TOKENS == 1:
#             unpatched_logits = model_lmhead.output[0, -1, :]
#             unpatched_prediction = torch.argmax(unpatched_logits, dim=-1)
#             unpatched_outputs.append([unpatched_prediction])
#             # Save indices, logits, and logprobs of the top 100 tokens
#             sorted_indices = torch.argsort(unpatched_logits, descending=True)
#             topk_indices = sorted_indices[:100]
#             topk_logits = unpatched_logits[topk_indices]
#             topk_logprobs = torch.log_softmax(unpatched_logits, dim=-1)[topk_indices]
#             unpatched_logits_all.append((topk_indices.save(), topk_logits.save(), topk_logprobs.save()))

# print("Patching and baseline runs complete.")
# print(f"Collected {len(patched_outputs)} patched outputs and {len(baseline_outputs)} baseline outputs.")

# %%
# len(patched_results)
# Start of Selection
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

def compute_rank_and_surprisal(topk_indices, topk_logits, gt_token_id):
    matches = (topk_indices == gt_token_id).nonzero()
    rank = matches.item() + 1 if len(matches) > 0 else None
    probs = torch.softmax(topk_logits, dim=-1)
    if rank is not None:
        prob_gt = probs[rank - 1].item()
    else:
        prob_gt = 0.0
    surprisal = -np.log2(prob_gt + 1e-20)
    return rank, surprisal, prob_gt

gt_ranks = {'baseline': [], 'patched': [], 'unpatched': []}
gt_surprisals = {'baseline': [], 'patched': [], 'unpatched': []}
gt_probs = {'baseline': [], 'patched': [], 'unpatched': []}

print("\n--- Ground Truth Token Ranks and Surprisals ---")
print(f"{'Idx':<4} {'GT_ID':<8} {'Truth':<15} | {'BL rank':<12} {'Surp':<10} | {'P rank':<16} {'Surp':<10} | {'UP rank':<12} {'Surp':<10} | {'Info'}")

for i in range(len(dataset['ground_truths'])):
    if i >= len(baseline_results) or i >= len(patched_results) or i >= len(unpatched_results):
        break

    gt_text = dataset['ground_truths'][i]
    gt_token_ids = tokenizer.encode(gt_text, add_special_tokens=False)
    if len(gt_token_ids) != 1:
        print(f"{i:<4} Multi-token GT not supported")
        continue
    gt_token_id = gt_token_ids[0]

    _, baseline_topk_indices, baseline_topk_logits, _ = baseline_results[i]
    _, patched_topk_indices, patched_topk_logits, _ = patched_results[i]
    _, unpatched_topk_indices, unpatched_topk_logits, _ = unpatched_results[i]

    rank_b, surprisal_b, prob_b = compute_rank_and_surprisal(baseline_topk_indices, baseline_topk_logits, gt_token_id)
    rank_p, surprisal_p, prob_p = compute_rank_and_surprisal(patched_topk_indices, patched_topk_logits, gt_token_id)
    rank_u, surprisal_u, prob_u = compute_rank_and_surprisal(unpatched_topk_indices, unpatched_topk_logits, gt_token_id)

    gt_ranks['baseline'].append(rank_b)
    gt_ranks['patched'].append(rank_p)
    gt_ranks['unpatched'].append(rank_u)

    gt_surprisals['baseline'].append(surprisal_b)
    gt_surprisals['patched'].append(surprisal_p)
    gt_surprisals['unpatched'].append(surprisal_u)

    gt_probs['baseline'].append(prob_b)
    gt_probs['patched'].append(prob_p)
    gt_probs['unpatched'].append(prob_u)

    def fmt(rank, surprisal):
        if rank is None:
            return f"{'N/A':<12} {'N/A':<10}"
        else:
            return f"{rank:<12} {surprisal:<10.2f}"

    extra_info = dataset.get('extra_info', [''] * len(dataset['ground_truths']))
    extra_str = extra_info[i] if i < len(extra_info) else ''

    def delta_fmt(rank_other, rank_base):
        if rank_other is None or rank_base is None:
            return ""
        delta = rank_other - rank_base
        sign = "+" if delta > 0 else ""
        return f" ({sign}{delta})"

    patched_delta = delta_fmt(rank_p, rank_b)
    unpatched_delta = delta_fmt(rank_u, rank_b)

    # Compose patched rank with delta inside brackets, aligned in 16 spaces
    if rank_p is None:
        patched_rank_str = f"{'N/A':<16}"
    else:
        patched_rank_str = f"{str(rank_p) + patched_delta:<16}"

    # Compose unpatched rank with delta inside brackets, aligned in 12 spaces
    if rank_u is None:
        unpatched_rank_str = f"{'N/A':<12}"
    else:
        unpatched_rank_str = f"{str(rank_u) + unpatched_delta:<12}"

    print(f"{i:<4} {gt_token_id:<8} {gt_text:<15} | "
          f"{fmt(rank_b, surprisal_b)} | "
          f"{patched_rank_str} {surprisal_p:<10.2f} | "
          f"{unpatched_rank_str} {surprisal_u:<10.2f} | "
          f"{extra_str}")


# Plotly boxplot for surprisals
fig_surp = go.Figure()
for key, color in zip(['baseline', 'patched', 'unpatched'], ['blue', 'red', 'green']):
    fig_surp.add_trace(go.Box(
        y=gt_surprisals[key],
        name=key.capitalize(),
        boxmean=True,
        marker_color=color,
        boxpoints='outliers'
    ))
fig_surp.update_layout(
    title="Ground Truth Token Surprisal (Lower is Better)",
    yaxis_title="Surprisal (bits)",
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
summarize_metric(gt_probs, "GT Token Probability")

print("\n--- Examples with Lowest Patched Surprisal ---")
sorted_idx = np.argsort(gt_surprisals['patched'])
for idx in sorted_idx[:5]:
    extra_info = dataset.get('extra_info', [''] * len(dataset['ground_truths']))
    extra_str = extra_info[idx] if idx < len(extra_info) else ''
    print(f"Example {idx}: Surprisal={gt_surprisals['patched'][idx]:.2f}, Rank={gt_ranks['patched'][idx]}, GT='{dataset['ground_truths'][idx]}', Q='{dataset['questions'][idx]}', Extra='{extra_str}'")

print("\n--- Examples with Highest Patched Surprisal ---")
sorted_idx = np.argsort(gt_surprisals['patched'])[::-1]
for idx in sorted_idx[:5]:
    extra_info = dataset.get('extra_info', [''] * len(dataset['ground_truths']))
    extra_str = extra_info[idx] if idx < len(extra_info) else ''
    print(f"Example {idx}: Surprisal={gt_surprisals['patched'][idx]:.2f}, Rank={gt_ranks['patched'][idx]}, GT='{dataset['ground_truths'][idx]}', Q='{dataset['questions'][idx]}', Extra='{extra_str}'")

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

# %

# %%
