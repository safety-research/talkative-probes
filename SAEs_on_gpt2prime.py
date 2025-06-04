# %%
# Dictionary Learning and Sparse Autoencoders on Llama

# This interactive script demonstrates how to use a Sparse Autoencoder (SAE) to extract and visualize monosemantic features from a Llama model's activations.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from dotenv import load_dotenv
print("loaded successfully?",load_dotenv())


import sys
import gc
def free_unused_cuda_memory():
    """Free unused cuda memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        raise RuntimeError("not using cuda")
    gc.collect()
print(sys.executable)
print(sys.version)
# Get the SLURM environment variables
slurm_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', None)
print(slurm_gpus)
#os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in range(int(slurm_gpus)))
#os.environ["CUDA_VISIBLE_DEVICES"] = '5,6'
print(f"Restricted to {slurm_gpus} GPU(s)")
!pwd
# %%
# Core libraries
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
free_unused_cuda_memory()
# %%
# --- Setup: Install and Import Dependencies ---

from IPython.display import clear_output, display
# try:
#     import google.colab
#     is_colab = True
# except ImportError:
#     is_colab = False

# if is_colab:
#     # Install required packages in Colab
#     # Note: Uncomment the following lines if running in Colab
#     # !uv pip install -U nnsight
#     # !git clone https://github.com/saprmarks/dictionary_learning
#     # %cd dictionary_learning
#     # !uv pip install -r requirements.txt
#     pass
clear_output()

# %%
from nnsight import LanguageModel
from dictionary_learning.dictionary import AutoEncoder
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from safetensors.torch import load_file
import requests
import os
import json
from tqdm.auto import tqdm
from circuitsvis.tokens import colored_tokens_multi
from huggingface_hub import hf_hub_download
from typing import Optional, Callable
import torch
import nnsight
from nnsight import LanguageModel
from nnsight import CONFIG, util
#from nnsight.intervention import InterventionProxy
#from nnsight.tracing.graph import Proxy # For 
from SAEclasses import *
import goodfire


# %%
# --- Helper Functions ---


# %%
# --- SAE Class Definition ---


# %%
# --- Configuration ---

# Base Model Configuration
MODEL_ID = "openai-community/gpt2" # Base model ID
#MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct" # Base model ID

# SAE Configuration

SAE_REPO_ID = "jbloom/GPT2-Small-SAEs" # SAE repository on Hugging Face
SAE_LAYER = 6 # The layer the SAE was trained on (usually indicated in repo name)

# Common paths: Residual stream: model.layers[N].output[0], MLP out: model.layers[N].mlp.output
# Verify based on SAE training details if unsure.
TARGET_MODULE_PATH = f"transformer.h.{SAE_LAYER}"  # For GPT-2 residual stream
# Expansion factor needed for Goodfire SAEs (determines d_hidden = d_model * expansion_factor)
# Check the SAE repo card or config if unsure
# Goodfire Llama-3.1-8B uses 16, Llama-3.1-70B uses 8
EXPANSION_FACTOR = 32# 16 if "8B" in SAE_REPO_ID else 8 

# --- Optional: Alternative Configurations ---
# Example for a different model/SAE:
# MODEL_ID = "meta-llama/Meta-Llama-3.1-70B-Instruct"
# SAE_REPO_ID = "Goodfire/Llama-3.3-70B-Instruct-SAE-l50"
# SAE_LAYER = 50
# TARGET_MODULE_PATH = f"model.layers[{SAE_LAYER}].output[0]"
# EXPANSION_FACTOR = 8

# Example for a standard safetensors SAE:
# MODEL_ID = "EleutherAI/pythia-70m-deduped"
# SAE_REPO_ID = "ArthurConmy/pythia-70m-deduped-layer-0-av-sparsity-target-10" # Example standard SAE
# SAE_LAYER = 0
# TARGET_MODULE_PATH = f"gpt_neox.layers[{SAE_LAYER}].mlp.output" # Example path for Pythia MLP
# EXPANSION_FACTOR = None # Not needed if cfg.json exists



print("--- Configuration Loaded ---")
print(f"Model ID: {MODEL_ID}")
print(f"SAE Repo ID: {SAE_REPO_ID}")
print(f"Target Layer: {SAE_LAYER}")
print(f"Target Module Path: {TARGET_MODULE_PATH}")
print(f"Expansion Factor (for Goodfire): {EXPANSION_FACTOR}")
#print(f"Tokens to Generate: {N_GENERATE_TOKENS}")

# %%
# --- Load Model and Tokenizer ---

print(f"Loading base model using ObservableLanguageModel: {MODEL_ID}...")
# Note: Ensure you have necessary HF credentials/access if required for the model
# device='auto' should work, but specify GPU if needed e.g., 'cuda:0'
# dtype=torch.bfloat16 is used by default, matching the demo.
model = ObservableLanguageModel(MODEL_ID, device_map="auto",device="cuda", dtype=torch.bfloat16)
tokenizer = model.tokenizer 

# Get model properties needed for SAE loading
model_d_model = model.d_model # Get d_model inferred by ObservableLanguageModel
model_device = model.model_device # Get device from the wrapped nnsight model
model_dtype = model.model_dtype # Get dtype from the wrapped nnsight model

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer pad_token set to eos_token")

print(f"ObservableLanguageModel loaded. D_model={model_d_model}, Device={model_device}, Dtype={model_dtype}")

# %%
# --- Load Sparse Autoencoder (SAE) ---

print(f"Loading SAE from {SAE_REPO_ID}...")

sae = load_sae_from_repo(
    repo_id=SAE_REPO_ID,
    model_device=model_device,
    model_dtype=model_dtype,
    d_model=model_d_model,
    layer=SAE_LAYER,
    hook_point="hook_resid_pre",
    expansion_factor=EXPANSION_FACTOR  # This will be ignored for jbloom but kept for compatibility
)

print(f"SAE loaded: input dimension (d_in): {sae.d_in}, feature dimension (d_hidden): {sae.d_hidden}")

# --- Extract Activations from Base Model ---

#print(f"Extracting activations from module: {TARGET_MODULE_PATH}")

# Prepare inputs for the model's forward method
# Needs token IDs as a PyTorch tensor
# Use add_generation_prompt=True for instruct models if appropriate
# based on how the model expects input.
#input_tokens = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
# For instruct/chat models, consider using apply_chat_template:

def prompt_model_baseformat(model, tokenizer, model_device, PROMPT, N_GENERATE_TOKENS):
    """Generate text using a base model and return full sequence for analysis"""
    # For base models, just tokenize the raw prompt
    input_tokens = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(model_device)
    initial_prompt_len = input_tokens.shape[1]
    
    print(f"Generating {N_GENERATE_TOKENS} tokens...")
    print(f"Initial prompt length: {initial_prompt_len} tokens")
    
    # Generate continuation using nnsight
    with model._model.generate(input_tokens, max_new_tokens=N_GENERATE_TOKENS) as tracer:
        generated_output = model._model.generator.output.save()
    
    # Extract the full generated sequence
    full_tokens = generated_output[0]  # Shape: [sequence_length]
    print(f"Full generated length: {full_tokens.shape[0]} tokens")
    
    # For display purposes, separate prompt and continuation
    start_idx_response = initial_prompt_len
    decoded_prompt = tokenizer.decode(full_tokens[:start_idx_response].cpu())
    decoded_continuation = tokenizer.decode(full_tokens[start_idx_response:].cpu())
    
    print("\n\n******Full Text for Analysis:")
    print(decoded_prompt)
    print("--- GENERATED CONTINUATION STARTS HERE ---")
    print(decoded_continuation)
    print("\n\n")
    
    return full_tokens, start_idx_response, decoded_continuation

def generate_model_output(model, chat_input, n_generate_tokens):
    """Generate continuation from the model."""
    print(f"Generating {n_generate_tokens} tokens...")
    with model._model.generate(chat_input, max_newtokens=n_generate_tokens) as tracer:
        generated_output = model._model.generator.output.save()
    return generated_output

def process_conversation(tokenizer, generated_output):
    """Process the generated conversation to extract tokens and response position."""
    full_conversation_tokens = generated_output[0]  # Shape: [sequence_length]
    print(f"Full conversation length: {full_conversation_tokens.shape[0]} tokens")

    # Find the position where the model's response starts
    end_header_positions = [i for i, t in enumerate(full_conversation_tokens) if tokenizer.decode([t]) == "<|end_header_id|>"]
    start_idx_response = end_header_positions[-1] + 1 if end_header_positions else 0

    # Decode and print the prompt and generated answer
    decoded_prompt = tokenizer.decode(full_conversation_tokens[:start_idx_response].cpu())
    decoded_answer = tokenizer.decode(full_conversation_tokens[start_idx_response:].cpu())
    print("\n\n******Prompt: ", decoded_prompt)
    print("\n\n******Generated Answer: ", decoded_answer, "\n\n")
    
    return full_conversation_tokens, start_idx_response

def extract_activations(model, full_conversation_tokens, target_module_path):
    """Extract activations from the model for the full conversation."""
    print(f"Extracting activations for the full conversation from module: {target_module_path}")
    with torch.no_grad():
        # Use the ObservableLanguageModel's forward method on the full sequence
        # Add batch dimension back for forward pass
        last_token_logits, activation_cache = model.forward(
            inputs=full_conversation_tokens.unsqueeze(0),
            cache_activations_at=[target_module_path]
        )

    # Get the specific activation tensor from the cache dictionary
    if target_module_path not in activation_cache:
        raise KeyError(f"Target module path '{target_module_path}' not found in activation cache. Cached keys: {list(activation_cache.keys())}")
        
    activations_val = activation_cache[target_module_path]

    # If activations have batch, sequence and hidden_dim (e.g., [1, seq_len, hidden_dim])
    # often we need [seq_len, hidden_dim] for the SAE
    if activations_val.ndim == 3 and activations_val.shape[0] == 1:
        activations_val = activations_val.squeeze(0)

    print(f"Activations extracted. Shape: {activations_val.shape}, Dtype: {activations_val.dtype}")
    return activations_val

def apply_sae_to_activations(sae, activations_val):
    """Apply SAE to the extracted activations."""
    print("Applying SAE to activations...")
    with torch.no_grad():
        # Pass activations to the SAE's encode method
        # The encode method handles moving the input to the correct device/dtype
        sae_features = sae.encode(activations_val)
        # features will have shape [seq_len, d_hidden]

    #print(f"SAE features obtained. Shape: {sae_features.shape}, Dtype: {sae_features.dtype}")
    #print(torch.sum(sae_features,axis=-1)[0:10].detach().float().cpu().numpy(), "...")
    return sae_features

def prepare_tokens_for_visualization(tokenizer, full_tokens):
    """Prepare token strings for visualization"""
    return [tokenizer.decode([t]) for t in full_tokens]

def get_top_features(masked_sae_features, num_features=100):
    """Find the top activating features based on max activation value."""
    max_feature_activations, _ = masked_sae_features.abs().max(dim=0)
    top_feature_indices = max_feature_activations.topk(num_features).indices
    return top_feature_indices, max_feature_activations

def visualize_sae_features(masked_sae_features, display_tokens, start_idx_response, 
                          model_id, num_features=8192, goodfire_api_key=None):
    """Visualize SAE features using CircuitsVis."""
    print("Visualizing SAE features using CircuitsVis...")
    client = goodfire.Client(goodfire_api_key)
    variant = goodfire.Variant(model_id)
    
    # Get the top features to visualize (limit to specified number)
    num_features_to_visualize = min(num_features, masked_sae_features.shape[1])
    feature_importance = masked_sae_features.abs().mean(dim=0)
    top_feature_indices = feature_importance.topk(num_features_to_visualize).indices.cpu().numpy()

    # Prepare data for visualization
    tokens_for_vis = display_tokens
    features_for_vis = masked_sae_features[:, top_feature_indices].cpu().numpy()
    
    # Get feature labels from Goodfire API
    feature_labels = get_feature_labels(client, top_feature_indices, model_id)

    # Use CircuitsVis SAE visualization
    from circuitsvis import sae as sae_vis

    # Create the visualization
    sae_vis_output = sae_vis.sae_vis(
        tokens=tokens_for_vis,
        feature_activations=features_for_vis,
        feature_labels=feature_labels,
        feature_ids=top_feature_indices.tolist(),
        initial_ranking_metric="l1",        # Rank by mean absolute activation initially
        num_top_features_overall=15,        # Show top 15 features in the list
        num_top_features_per_token=3,       # Show top 3 features in token tooltips
    )

    # Display the visualization
    from IPython.display import display
    display(sae_vis_output)
    
    # Print some info about where the continuation starts
    print(f"Tokens 0-{start_idx_response-1}: Original prompt")
    print(f"Tokens {start_idx_response}-{len(tokens_for_vis)-1}: Generated continuation")
    print("--- SAE Feature Visualization Complete ---")
    
    return sae_vis_output, top_feature_indices

def get_feature_labels(client, top_feature_indices, model_id):
    """Get feature labels from Goodfire API."""
    feature_labels = []
    try:
        # Query Goodfire API for feature labels
        feature_dict = client.features.lookup(
            top_feature_indices.tolist(), 
            model=model_id
        )
        sorted_top_feature_indices = {i: feature_dict.get(i, f"Feature {i}") for i in top_feature_indices}
        feature_labels = [sorted_top_feature_indices[i].label if isinstance(sorted_top_feature_indices[i], goodfire.Feature) 
                         else sorted_top_feature_indices[i] for i in top_feature_indices]
        print(f"Retrieved {len(feature_labels)} feature labels from Goodfire API")
    except Exception as e:
        print(f"Error retrieving feature labels: {e}")
        # Fallback to generic labels if API fails
        feature_labels = [f"Feature {i}" for i in top_feature_indices]
    
    return feature_labels

# %%
# --- Analyze and Visualize SAE Features ---
# Visualization Configuration
NUM_TOP_FEATURES = 100 # Number of top activating features to visualize
N_GENERATE_TOKENS = 50 # Number of tokens to generate after the prompt

# Input Prompt (Modify as needed)
PROMPT = """
Call me Ishmael. Some years ago--never mind how long precisely--having little or no money in my purse, and nothing particular to interest me on shore,
"""
PROMPT = """
Veracruz was in a heap of trouble before week 15 of the Liga MX Clausura 2017. Most people had deemed Veracruz as the franchise that was doomed to abandon the Liga MX and relegate to the Ascenso MX. Many people also wanted to see Los Tiburones suffer after the incidents from the fans and the ownership that had occurred all year long. In week 15 the ‘jarochos’ got an enormous 2-0 victory over"""
# # Load prompt from file
# with open("talkative_probes_prompts/steg_attempt_firstletter.txt", "r") as f:
#     PROMPT = f.read()

full_conversation_tokens, start_idx_response, decoded_continuation = prompt_model_baseformat(model, tokenizer, model_device, PROMPT, N_GENERATE_TOKENS)

# --- Extract Activations for Full Sequence ---
print(f"Extracting activations for the full sequence from module: {TARGET_MODULE_PATH}")
with torch.no_grad():
    # Use the ObservableLanguageModel's forward method on the full sequence
    last_token_logits, activation_cache = model.forward(
        inputs=full_conversation_tokens.unsqueeze(0),
        cache_activations_at=[TARGET_MODULE_PATH]
    )

# Get activations and apply SAE
if TARGET_MODULE_PATH not in activation_cache:
    raise KeyError(f"Target module path '{TARGET_MODULE_PATH}' not found in activation cache.")
    
activations_val = activation_cache[TARGET_MODULE_PATH]

if activations_val.ndim == 3 and activations_val.shape[0] == 1:
    activations_val = activations_val.squeeze(0)

print(f"Activations extracted. Shape: {activations_val.shape}, Dtype: {activations_val.dtype}")

# Apply SAE to get features
sae_features = apply_sae_to_activations(sae, activations_val)
sae_features_cpu = sae_features.detach().float().cpu()

print(f"SAE features obtained. Shape: {sae_features_cpu.shape}, Dtype: {sae_features_cpu.dtype}")

# Prepare tokens for visualization (no separators)
str_tokens = prepare_tokens_for_visualization(tokenizer, full_conversation_tokens)

# Print the separation info
print(f"Tokens 0-{start_idx_response-1}: Original prompt")
print(f"Tokens {start_idx_response}-{len(str_tokens)-1}: Generated continuation")

# Visualize features for the full sequence
ntok = len(str_tokens)
sae_vis, top_feature_indices = visualize_sae_features(
    sae_features_cpu[:ntok], 
    str_tokens[:ntok], 
    start_idx_response, 
    MODEL_ID, 
    num_features=8192, 
    goodfire_api_key=os.environ.get('GOODFIRE_API_KEY')
)

# # Decode steganographic message from the continuation only
# print("Capital-letter-based steganography decoded message:")
# print(decode_steganographic_message_firstletter(decoded_continuation))




# %% [markdown]
#### Control Experiment - capital-letter-based steganography

# %%


# Visualization Configuration
NUM_TOP_FEATURES = 100 # Number of top activating features to visualize
N_GENERATE_TOKENS = 500 # Number of tokens to generate after the prompt

# # Load prompt from file
# with open("talkative_probes_prompts/steg_attempt_firstletter_control.txt", "r") as f:
#     PROMPT = f.read()

full_conversation_tokens, start_idx_response, decoded_continuation = prompt_model_baseformat(model, tokenizer, model_device, PROMPT, N_GENERATE_TOKENS)

# --- Extract Activations for Full Sequence ---
print(f"Extracting activations for the full sequence from module: {TARGET_MODULE_PATH}")
with torch.no_grad():
    # Use the ObservableLanguageModel's forward method on the full sequence
    last_token_logits, activation_cache = model.forward(
        inputs=full_conversation_tokens.unsqueeze(0),
        cache_activations_at=[TARGET_MODULE_PATH]
    )

# Get activations and apply SAE
if TARGET_MODULE_PATH not in activation_cache:
    raise KeyError(f"Target module path '{TARGET_MODULE_PATH}' not found in activation cache.")
    
activations_val = activation_cache[TARGET_MODULE_PATH]

if activations_val.ndim == 3 and activations_val.shape[0] == 1:
    activations_val = activations_val.squeeze(0)

print(f"Activations extracted. Shape: {activations_val.shape}, Dtype: {activations_val.dtype}")

# Apply SAE to get features
sae_features = apply_sae_to_activations(sae, activations_val)
sae_features_cpu = sae_features.detach().float().cpu()

print(f"SAE features obtained. Shape: {sae_features.shape}, Dtype: {sae_features.dtype}")

# Prepare tokens for visualization (no separators)
str_tokens = prepare_tokens_for_visualization(tokenizer, full_conversation_tokens)

# Print the separation info
print(f"Tokens 0-{start_idx_response-1}: Original prompt")
print(f"Tokens {start_idx_response}-{len(str_tokens)-1}: Generated continuation")

# Visualize features for the full sequence
ntok = len(str_tokens)
sae_vis, top_feature_indices = visualize_sae_features(
    sae_features_cpu[:ntok], 
    str_tokens[:ntok], 
    start_idx_response, 
    MODEL_ID, 
    num_features=8192, 
    goodfire_api_key=os.environ.get('GOODFIRE_API_KEY')
)
# %% [markdown]




