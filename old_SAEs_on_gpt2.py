# %%
# Dictionary Learning and Sparse Autoencoders on GPT-2

# This interactive script demonstrates how to use a Sparse Autoencoder (SAE) to extract and visualize monosemantic features from a GPT-2 model's activations.
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
        print("not using cuda")
        #raise RuntimeError("not using cuda")
    gc.collect()
print(sys.executable)
print(sys.version)
# Get the SLURM environment variables
slurm_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', None)
print(slurm_gpus)
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
    pass
clear_output()

# %%
from nnsight import LanguageModel
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
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
from SAEclasses import *

# %%
# --- Configuration ---

# Base Model Configuration - GPT-2 models
MODEL_ID = "openai-community/gpt2"  # Options: "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"

# SAE Configuration for different GPT-2 models
if MODEL_ID == "openai-community/gpt2":
    # GPT-2 small (117M parameters, 768 hidden dim, 12 layers)
    SAE_REPO_ID = "jbloom/GPT2-Small-SAEs"  # Joseph Bloom's GPT-2 SAEs
    SAE_LAYER = 6  # Middle layer for GPT-2 small
    EXPANSION_FACTOR = 32  # As specified in the repository
    MODEL_HIDDEN_DIM = 768
    MODEL_NUM_LAYERS = 12
elif MODEL_ID == "gpt2-medium":
    # GPT-2 medium (345M parameters, 1024 hidden dim, 24 layers)
    SAE_REPO_ID = "openai/sparse_autoencoder"  # May need to check available SAEs
    SAE_LAYER = 12
    EXPANSION_FACTOR = 16  # Common expansion factor
    MODEL_HIDDEN_DIM = 1024
    MODEL_NUM_LAYERS = 24
elif MODEL_ID == "gpt2-large":
    # GPT-2 large (774M parameters, 1280 hidden dim, 36 layers)
    SAE_REPO_ID = "openai/sparse_autoencoder"
    SAE_LAYER = 18
    EXPANSION_FACTOR = 16
    MODEL_HIDDEN_DIM = 1280
    MODEL_NUM_LAYERS = 36
elif MODEL_ID == "gpt2-xl":
    # GPT-2 xl (1.5B parameters, 1600 hidden dim, 48 layers)
    SAE_REPO_ID = "openai/sparse_autoencoder"
    SAE_LAYER = 24
    EXPANSION_FACTOR = 16
    MODEL_HIDDEN_DIM = 1600
    MODEL_NUM_LAYERS = 48

# GPT-2 uses different module naming convention
TARGET_MODULE_PATH = f"h.{SAE_LAYER}"  # GPT-2 transformer blocks are named h.0, h.1, etc.

print("--- Configuration Loaded ---")
print(f"Model ID: {MODEL_ID}")
print(f"SAE Repo ID: {SAE_REPO_ID}")
print(f"Target Layer: {SAE_LAYER}")
print(f"Target Module Path: {TARGET_MODULE_PATH}")
print(f"Model Hidden Dim: {MODEL_HIDDEN_DIM}")
print(f"Expansion Factor: {EXPANSION_FACTOR}")

# %%
# --- Load Model and Tokenizer ---

print(f"Loading GPT-2 model: {MODEL_ID}...")
# Use nnsight's LanguageModel for GPT-2
model = ObservableLanguageModel(MODEL_ID, device_map="auto", device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
tokenizer = model.tokenizer 

model_d_model = model.d_model # Get d_model inferred by ObservableLanguageModel
model_device = model.model_device # Get device from the wrapped nnsight model
model_dtype = model.model_dtype # Get dtype from the wrapped nnsight model

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer pad_token set to eos_token")

print(f"GPT-2 model loaded. D_model={model_d_model}, Device={model_device}, Dtype={model_dtype}")

# %%
# --- Simple SAE Loading for GPT-2 ---

class SimpleGPT2SAE(nn.Module):
    """Simple SAE class for GPT-2 compatible with standard SAE formats"""
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        
        # Standard SAE components
        self.encoder = nn.Linear(d_in, d_hidden, bias=False)
        self.decoder = nn.Linear(d_hidden, d_in, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_in))
        
    def encode(self, x):
        """Encode input to sparse features"""
        return torch.relu(self.encoder(x - self.bias))
    
    def decode(self, features):
        """Decode features back to original space"""
        return self.decoder(features) + self.bias
    
    def forward(self, x):
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction

def load_gpt2_sae(repo_id, layer_num, d_model, expansion_factor):
    """Load SAE for GPT-2 from HuggingFace repository"""
    print(f"Loading SAE from {repo_id} for layer {layer_num}...")
    
    try:
        # Try to load from jbloom's repository format
        if "jbloom" in repo_id:
            # jbloom's SAEs are stored as individual layer files
            filename = f"layer_{layer_num}.safetensors"
            weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
            weights = load_file(weights_path)
            
            # Create SAE with correct dimensions
            d_hidden = d_model * expansion_factor
            sae = SimpleGPT2SAE(d_model, d_hidden)
            
            # Load weights (adjust key names if needed)
            if 'encoder.weight' in weights:
                sae.encoder.weight.data = weights['encoder.weight']
            if 'decoder.weight' in weights:
                sae.decoder.weight.data = weights['decoder.weight']
            if 'bias' in weights:
                sae.bias.data = weights['bias']
                
        else:
            # Try OpenAI format or other formats
            # This may need adjustment based on actual file structure
            config_file = hf_hub_download(repo_id=repo_id, filename="config.json")
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            filename = f"sae_{layer_num}.safetensors"
            weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
            weights = load_file(weights_path)
            
            sae = SimpleGPT2SAE(config['d_in'], config['d_hidden'])
            sae.load_state_dict(weights)
            
    except Exception as e:
        print(f"Error loading SAE: {e}")
        print("Creating a random SAE for demonstration...")
        d_hidden = d_model * expansion_factor
        sae = SimpleGPT2SAE(d_model, d_hidden)
    
    return sae.to(model_device).to(model_dtype)

# Load the SAE
sae = load_gpt2_sae(SAE_REPO_ID, SAE_LAYER, model_d_model, EXPANSION_FACTOR)
print(f"SAE loaded: input dimension (d_in): {sae.d_in}, feature dimension (d_hidden): {sae.d_hidden}")

# %%
# --- Modified helper functions for GPT-2 with proper nnsight syntax ---

def generate_gpt2_output(model, prompt, n_generate_tokens):
    """Generate continuation from GPT-2 model using nnsight context."""
    print(f"Generating {n_generate_tokens} tokens...")
    
    # Tokenize input
    inputs = model.tokenizer(prompt, return_tensors="pt").to(model_device)
    chat_input = inputs.input_ids
    initial_prompt_len = chat_input.shape[1]
    
    # Use nnsight generate context
    with model._model.generate(chat_input, max_new_tokens=n_generate_tokens) as tracer:
        generated_output = model._model.generator.output.save()
    
    # Get full conversation tokens
    full_conversation_tokens = generated_output[0]  # Shape: [sequence_length]
    print(f"Full conversation length: {full_conversation_tokens.shape[0]} tokens")
    
    # For GPT-2, the response starts right after the prompt
    start_idx_response = initial_prompt_len
    
    # Decode the generated text
    decoded_prompt = model.tokenizer.decode(full_conversation_tokens[:start_idx_response].cpu())
    decoded_answer = model.tokenizer.decode(full_conversation_tokens[start_idx_response:].cpu())
    
    print(f"Prompt: {decoded_prompt}")
    print(f"Generated text: {decoded_answer[:200]}...")
    
    return full_conversation_tokens, start_idx_response, decoded_answer

def extract_gpt2_activations(model, full_conversation_tokens, target_module_path):
    """Extract activations from GPT-2 model using nnsight trace context."""
    print(f"Extracting activations for the full conversation from module: {target_module_path}")
    
    with torch.no_grad():
        with model._model.trace(full_conversation_tokens.unsqueeze(0)):
            # Access the specific transformer block for GPT-2
            if "h." in target_module_path:
                layer_num = int(target_module_path.split('.')[1])
                # GPT-2 transformer blocks are accessed as transformer.h[layer_num]
                activations = model.transformer.h[layer_num].output[0].save()
            else:
                raise ValueError(f"Unsupported module path: {target_module_path}")
    
    # Ensure activations are in the right shape [seq_len, hidden_dim]
    if activations.ndim == 3 and activations.shape[0] == 1:
        activations = activations.squeeze(0)
    
    print(f"Activations extracted. Shape: {activations.shape}, Dtype: {activations.dtype}")
    return activations

def apply_sae_to_activations(sae, activations_val):
    """Apply SAE to the extracted activations."""
    print("Applying SAE to activations...")
    with torch.no_grad():
        sae_features = sae.encode(activations_val)
    
    print(f"SAE features obtained. Shape: {sae_features.shape}, Dtype: {sae_features.dtype}")
    return sae_features

def create_feature_mask(sae_features, start_idx_response=None, analyze_only_response=True):
    """Create a mask for SAE features based on analysis scope."""
    sae_features_cpu = sae_features.detach().cpu()
    
    if analyze_only_response and start_idx_response is not None:
        # Create a response-only mask (1 for response tokens, 0 for everything else)
        mask = torch.zeros(sae_features_cpu.shape[0], device='cpu')
        mask[start_idx_response:] = 1
        print(f"Analyzing only the generated response ({mask.sum().item()} tokens)")
    else:
        # Create mask (1 for all tokens)
        mask = torch.ones(sae_features_cpu.shape[0], device='cpu')
        print("Analyzing all tokens in the conversation")
    
    # Apply mask to features
    masked_sae_features = sae_features_cpu * mask.unsqueeze(-1)  # Expand mask to feature dim
    
    return masked_sae_features, sae_features_cpu

# %%
# --- Main Analysis with proper nnsight syntax ---

# Configuration
NUM_TOP_FEATURES = 100
N_GENERATE_TOKENS = 200

# Input Prompt
PROMPT = "The quick brown fox jumps over the lazy dog. This sentence contains"

# Generate text using nnsight context
full_conversation_tokens, start_idx_response, generated_text = generate_gpt2_output(model, PROMPT, N_GENERATE_TOKENS)

# Extract activations using nnsight trace context
activations = extract_gpt2_activations(model, full_conversation_tokens, TARGET_MODULE_PATH)

# Apply SAE
sae_features = apply_sae_to_activations(sae, activations)

# Create feature mask
analyze_only_response = True
masked_sae_features, sae_features_cpu = create_feature_mask(
    sae_features, 
    start_idx_response=start_idx_response, 
    analyze_only_response=analyze_only_response
)

# Get token strings for visualization
str_tokens = [model.tokenizer.decode([t]) for t in full_conversation_tokens]
display_tokens = str_tokens[start_idx_response:] if analyze_only_response else str_tokens

# Find top activating features
max_feature_activations, _ = masked_sae_features.abs().max(dim=0)
top_feature_indices = max_feature_activations.topk(NUM_TOP_FEATURES).indices

print(f"Top {NUM_TOP_FEATURES} features found")
print(f"Generated text: {generated_text}")

# %%
# --- Visualization ---

# Get activations for visualization
if analyze_only_response:
    vis_features = masked_sae_features[start_idx_response:start_idx_response+len(display_tokens), :]
else:
    vis_features = masked_sae_features[:len(display_tokens), :]

vis_features = vis_features[:, top_feature_indices[:20]]  # Show top 20 features

# Create visualization
print("Creating visualization...")
try:
    vis_html = colored_tokens_multi(display_tokens, vis_features.numpy())
    display(vis_html)
    print("--- Feature Visualization Complete ---")
except Exception as e:
    print(f"Visualization error: {e}")
    print("Feature activation summary:")
    for i, feat_idx in enumerate(top_feature_indices[:10]):
        max_activation = max_feature_activations[feat_idx].item()
        print(f"Feature {feat_idx.item()}: max activation = {max_activation:.4f}")

print("--- GPT-2 SAE Analysis Complete ---")

# %%
