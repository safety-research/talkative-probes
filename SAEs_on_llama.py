# %%
# Dictionary Learning and Sparse Autoencoders on Llama

# This interactive script demonstrates how to use a Sparse Autoencoder (SAE) to extract and visualize monosemantic features from a Llama model's activations.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
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
try:
    import google.colab
    is_colab = True
except ImportError:
    is_colab = False

if is_colab:
    # Install required packages in Colab
    # Note: Uncomment the following lines if running in Colab
    # !uv pip install -U nnsight
    # !git clone https://github.com/saprmarks/dictionary_learning
    # %cd dictionary_learning
    # !uv pip install -r requirements.txt
    pass
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
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct" # Base model ID
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct" # Base model ID

# SAE Configuration
#SAE_REPO_ID = "Goodfire/Llama-3.1-8B-Instruct-SAE-l19" # SAE repository on Hugging Face
SAE_REPO_ID = "Goodfire/Llama-3.3-70B-Instruct-SAE-l50" # SAE repository on Hugging Face
SAE_LAYER = 50 # The layer the SAE was trained on (usually indicated in repo name)
# Common paths: Residual stream: model.layers[N].output[0], MLP out: model.layers[N].mlp.output
# Verify based on SAE training details if unsure.
TARGET_MODULE_PATH = f"model.layers.{SAE_LAYER}" 
# Expansion factor needed for Goodfire SAEs (determines d_hidden = d_model * expansion_factor)
# Check the SAE repo card or config if unsure
# Goodfire Llama-3.1-8B uses 16, Llama-3.1-70B uses 8
EXPANSION_FACTOR = 16 if "8B" in SAE_REPO_ID else 8 

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

# Visualization Configuration
NUM_TOP_FEATURES = 20 # Number of top activating features to visualize

# Input Prompt (Modify as needed)
PROMPT = """
Call me Ishmael. Some years ago--never mind how long precisely--having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation. Whenever I find myself growing grim about the mouth; whenever it is a damp, drizzly November in my soul; whenever I find myself involuntarily pausing before coffin warehouses, and bringing up the rear of every funeral I meet; and especially whenever my hypos get such an upper hand of me, that it requires a strong moral principle to prevent me from deliberately stepping into the street, and methodically knocking people's hats off--then, I account it high time to get to sea as soon as I can.
"""

print("--- Configuration Loaded ---")
print(f"Model ID: {MODEL_ID}")
print(f"SAE Repo ID: {SAE_REPO_ID}")
print(f"Target Layer: {SAE_LAYER}")
print(f"Target Module Path: {TARGET_MODULE_PATH}")
print(f"Expansion Factor (for Goodfire): {EXPANSION_FACTOR}")

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
    d_model=model_d_model, # Pass d_model from base model
    expansion_factor=EXPANSION_FACTOR # Pass expansion factor
    # Optional: specify weight_filename_sf, cfg_filename_sf, weight_filename_gf if non-standard
)

print("SAE loaded.")
# Print SAE details
print(f"SAE input dimension (d_in): {sae.d_in}")
print(f"SAE feature dimension (d_hidden): {sae.d_hidden}") # Using d_hidden now
# --- Extract Activations from Base Model ---

print(f"Extracting activations from module: {TARGET_MODULE_PATH}")

# Prepare inputs for the model's forward method
# Needs token IDs as a PyTorch tensor
# Use add_generation_prompt=True for instruct models if appropriate
# based on how the model expects input.
#input_tokens = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
# For instruct/chat models, consider using apply_chat_template:
chatformat = [{"role": "user", "content": PROMPT}]
chat_input = tokenizer.apply_chat_template(chatformat, return_tensors="pt")
input_tokens = chat_input


with torch.no_grad(): # Disable gradient calculation for inference
    # Use the ObservableLanguageModel's forward method
    # Pass the list containing the target module path to cache_activations_at
    last_token_logits, activation_cache = model.forward(
        inputs=input_tokens,
        cache_activations_at=[TARGET_MODULE_PATH]
    )

# Get the specific activation tensor from the cache dictionary
if TARGET_MODULE_PATH not in activation_cache:
    raise KeyError(f"Target module path '{TARGET_MODULE_PATH}' not found in activation cache. Cached keys: {list(activation_cache.keys())}")
    
activations_val = activation_cache[TARGET_MODULE_PATH]

# If activations have batch, sequence and hidden_dim (e.g., [1, seq_len, hidden_dim])
# often we need [seq_len, hidden_dim] for the SAE
if activations_val.ndim == 3 and activations_val.shape[0] == 1:
    activations_val = activations_val.squeeze(0)

print(f"Activations extracted. Shape: {activations_val.shape}, Dtype: {activations_val.dtype}")
# %%

#print(sae_features[0:2,0:3])
# --- Apply SAE to Activations ---

#print(sae.encoder_linear.bias[0:2])#,0:3])
# %%
# --- Apply SAE to Activations ---

print("Applying SAE to activations...")
with torch.no_grad():
    # Pass activations to the SAE's encode method
    # The encode method handles moving the input to the correct device/dtype
    sae_features = sae.encode(activations_val)
    # features will have shape [seq_len, d_hidden]

print(f"SAE features obtained. Shape: {sae_features.shape}, Dtype: {sae_features.dtype}")
print(torch.sum(sae_features,axis=-1)[0:10].detach().float().cpu().numpy(), "...")

# %%
# --- Analyze and Visualize SAE Features ---

print(f"Analyzing top {NUM_TOP_FEATURES} features...")
print(sae_features.shape,sae_features[0:2,0:3])

# Ensure features are on CPU for analysis/visualization
# remove the first few tokens
toremove = [{"role": "user", "content":"" }]
len_to_remove = len(tokenizer.apply_chat_template(toremove, return_tensors="pt")[0])-1
sae_features_cpu = sae_features.detach().cpu()[len_to_remove:]
str_tokens = [tokenizer.decode([t]) for t in input_tokens[0][len_to_remove:]]
print("removing",len_to_remove,"tokens")

# Find the top activating features based on max activation value across the sequence
max_feature_activations, _ = sae_features_cpu.abs().max(dim=0)
top_feature_indices = max_feature_activations.topk(NUM_TOP_FEATURES).indices

print(f"Top {NUM_TOP_FEATURES} feature indices: {top_feature_indices.tolist()}")
client = goodfire.Client(os.environ.get('GOODFIRE_API_KEY'))
featuresdict = client.features.lookup(top_feature_indices.tolist(),MODEL_ID)
featuresdictsorted = {i:featuresdict[i] for i in top_feature_indices.tolist()}
for i,(k,v) in enumerate(featuresdictsorted.items()):
    print(f"{i}: {k}: {v.label}")

# Prepare data for visualization
# We need the activation values of these specific top features for each token
# Shape required by colored_tokens_multi: [token, feature]
top_features_acts = sae_features_cpu[:, top_feature_indices]

# Get token strings
#tokens = tokenizer.encode(PROMPT)
#str_tokens = [tokenizer.decode([t]) for t in tokens] # Decode one by one for safety

print(f"Visualizing activations for top {NUM_TOP_FEATURES} features...")

# Generate visualization
vis_html = colored_tokens_multi(str_tokens, top_features_acts)

display(vis_html) # Display in Jupyter/IPython environment

print("--- Visualization Complete ---")

# %%
print(os.environ.get('GOODFIRE_API_KEY'))
# %%
import goodfire
#from goodfire import Variant
#from goodfire import Client

#variant = Variant("meta-llama/Llama-3.3-70B-Instruct")
variant = goodfire.Variant(MODEL_ID)





client = goodfire.Client(os.environ.get('GOODFIRE_API_KEY'))
#client = goodfire.Client()
pirate_features = client.features.search('pirate', MODEL_ID)
pirate_feature = pirate_features[0]
goodfirechatinput = chatformat 
print(goodfirechatinput)
#  [
#         {"role": "user", "content": "What do you think about pirates and whales"},
#         {"role": "assistant", "content": "I think pirates are cool and whales are cool"}
#     ],
# Analyze how features activate in text
inspector = client.features.inspect(
    goodfirechatinput,
    model=variant
)

# Get top activated features
for activation in inspector.top(k=5):
    print(f"{activation.feature.label}: {activation.activation}")

# %%


# %%
# --- Optional: Visualize Original Activations (Without SAE) ---
# This helps compare the sparsity and interpretability

# print("Analyzing and visualizing original activations (without SAE)...")

# Ensure original activations are on CPU
# activations_val_cpu = activations_val.detach().cpu()

# # Find top activating *neurons* in the original activation space
# max_neuron_activations, _ = activations_val_cpu.abs().max(dim=0)
# top_neuron_indices = max_neuron_activations.topk(NUM_TOP_FEATURES).indices

# print(f"Top {NUM_TOP_FEATURES} neuron indices (original): {top_neuron_indices.tolist()}")

# # Get activations of these top neurons for each token
# top_neuron_acts = activations_val_cpu[:, top_neuron_indices]

# # Generate visualization
# vis_html_orig = colored_tokens_multi(str_tokens, top_neuron_acts)
# display(vis_html_orig)

# print("--- Original Activation Visualization Complete ---")

# %%

# --- Observable Language Model Class Definition (from Goodfire Demo) ---


# %%

