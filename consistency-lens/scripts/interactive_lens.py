#!/usr/bin/env python3
"""
Interactive script for experimenting with encoder/decoder and activation patching.

Usage:
1. First, get an interactive GPU allocation with SLURM:
   salloc -p gpu --gres=gpu:1 --time=2:00:00 --mem=32G

2. Then run this script interactively:
   uv run python scripts/interactive_lens.py --config conf/gpt2_frozen.yaml

3. In VSCode, you can use the Python Interactive window:
   - Select all code and run with Shift+Enter
   - Or use # %% to create cells
"""

# %%
import torch
import torch.nn.functional as F
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
from transformers import AutoTokenizer

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from lens.models.decoder import Decoder, DecoderConfig, Generated
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.orig import OrigWrapper
from lens.training.loop import train_step
from lens.data.dataset import ActivationDataset
from lens.data.collate import collate

# %% [markdown]
# # Interactive Lens Experimentation
# This notebook allows you to experiment with the encoder/decoder architecture

# %%
def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file."""
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(config_name=Path(config_path).stem)
    return cfg

def setup_models(cfg: DictConfig, checkpoint_path: Optional[str] = None):
    """Set up the models and tokenizer from config."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Extract model names and layer
    model_name = cfg.model_name
    tokenizer_name = cfg.get('tokenizer_name', model_name)
    layer = cfg.get('layer_l', 5)
    
    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Initialize decoder
    decoder_train_cfg = cfg.get('trainable_components', {}).get('decoder', {})
    decoder_config = DecoderConfig(
        model_name=model_name,
        **decoder_train_cfg
    )
    decoder = Decoder(decoder_config).to(device)
    
    # Initialize encoder
    encoder_train_cfg = cfg.get('trainable_components', {}).get('encoder', {})
    encoder_config = EncoderConfig(
        model_name=model_name,
        **encoder_train_cfg
    )
    encoder = Encoder(encoder_config).to(device)
    
    # Initialize original model wrapper
    orig = OrigWrapper(model_name, load_in_8bit=False)
    orig.model.to(device)
    
    # Set decoder prompt if configured
    prompt = cfg.get('decoder_prompt', 'The activation <embed> represents:')
    decoder.set_prompt(prompt, tokenizer)
    print(f"Set decoder prompt: {prompt}")
    
    # Initialize encoder soft prompt from text if specified
    encoder_soft_prompt_text = encoder_train_cfg.get('soft_prompt_init_text')
    if encoder_soft_prompt_text:
        encoder.set_soft_prompt_from_text(encoder_soft_prompt_text, tokenizer)
        print(f"Initialized encoder soft prompt from text: {encoder_soft_prompt_text}")
    
    # Load checkpoint if provided
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'decoder' in checkpoint:
            decoder.load_state_dict(checkpoint['decoder'])
        if 'encoder' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder'])
        print("Checkpoint loaded successfully!")
    
    return decoder, encoder, orig, tokenizer, device, layer

def extract_activation(orig: OrigWrapper, text: str, tokenizer, layer: int, 
                      token_pos: int = -1, device='cuda') -> torch.Tensor:
    """Extract activation from the original model at specified layer and position."""
    # Tokenize input
    input_ids = tokenizer(text, return_tensors='pt')['input_ids'].to(device)
    
    # Get activation using forward_with_replacement (like in training)
    with torch.no_grad():
        # Create a dummy activation to get the right shape
        dummy_act = torch.zeros(1, orig.model.config.hidden_size).to(device)
        
        # Run forward pass to extract activation
        outputs = orig.model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer + 1]  # +1 because index 0 is embeddings
        
        # Extract at specific position
        if token_pos == -1:
            # Last token
            activation = hidden_states[:, -1, :]
        else:
            activation = hidden_states[:, token_pos, :]
    
    return activation.squeeze(0)  # Remove batch dimension

def generate_text_from_activation(decoder: Decoder, encoder: Encoder, 
                                 activation: torch.Tensor, tokenizer,
                                 max_length: int = 50, tau: float = 1.0,
                                 decode_method: str = "generate_soft") -> Tuple[str, Generated, torch.Tensor]:
    """Generate text from activation and reconstruct it back."""
    # Ensure activation has batch dimension
    if activation.dim() == 1:
        activation = activation.unsqueeze(0)
    
    # Generate text with decoder using specified method
    if decode_method == "generate_soft":
        gen = decoder.generate_soft(activation, max_length=max_length, gumbel_tau=tau)
    elif decode_method == "generate_soft_kv_cached":
        gen = decoder.generate_soft_kv_cached(activation, max_length=max_length, gumbel_tau=tau)
    elif decode_method == "generate_soft_kv_flash":
        gen = decoder.generate_soft_kv_flash(activation, max_length=max_length, gumbel_tau=tau)
    elif decode_method == "generate_soft_chkpt":
        gen = decoder.generate_soft_chkpt(activation, max_length=max_length, gumbel_tau=tau,
                                         checkpoint_every_n_tokens=decoder.config.checkpoint_every_n_tokens)
    else:
        raise ValueError(f"Unknown decode method: {decode_method}")
    
    # Decode generated tokens to text
    generated_text = tokenizer.decode(gen.hard_token_ids[0])
    
    # Reconstruct activation with encoder
    reconstructed = encoder(gen.generated_text_embeddings)
    
    return generated_text, gen, reconstructed

def run_full_train_step(batch: Dict[str, torch.Tensor], decoder: Decoder, encoder: Encoder, 
                       orig: OrigWrapper, config: DictConfig, tokenizer, device='cuda') -> Dict[str, torch.Tensor]:
    """Run a full training step to see all losses."""
    # Move batch to device
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # Get training parameters from config
    t_text = config.get('t_text', 10)
    tau = config.get('gumbel_tau_schedule', {}).get('value', 1.0)
    alpha = config.get('alpha_schedule', {}).get('value', 0.1)
    lm_base_weight = config.get('lm_base_weight', 1.0)
    kl_base_weight = config.get('kl_base_weight', 1.0)
    entropy_weight = config.get('entropy_weight', 0.0)
    mse_weight = config.get('mse_weight', 0.0)
    
    # Prepare loss function arguments
    loss_fns = {
        "tau": tau,
        "T_text": t_text,
        "alpha": alpha,
        "lm_base_weight": lm_base_weight,
        "kl_base_weight": kl_base_weight,
        "entropy_weight": entropy_weight,
        "mse_weight": mse_weight,
    }
    
    # Run training step
    losses = train_step(
        batch,
        {"dec": decoder, "enc": encoder, "orig": orig},
        loss_fns,
        lm_loss_natural_prefix=config.get('lm_loss_natural_prefix'),
        tokenizer=tokenizer,
        cached_prefix_ids=None,
        resample_ablation=config.get('resample_ablation', True)
    )
    
    return losses

def patch_activations(act1: torch.Tensor, act2: torch.Tensor, 
                     alpha: float = 0.5) -> torch.Tensor:
    """Interpolate between two activation tensors."""
    return (1 - alpha) * act1 + alpha * act2

def analyze_activations(act1: torch.Tensor, act2: torch.Tensor) -> Dict[str, float]:
    """Analyze differences between two activation tensors."""
    diff = act1 - act2
    return {
        'mean_abs_diff': diff.abs().mean().item(),
        'max_abs_diff': diff.abs().max().item(),
        'cosine_similarity': F.cosine_similarity(act1.flatten(), act2.flatten(), dim=0).item(),
        'l2_distance': torch.norm(diff, p=2).item(),
        'relative_l2': (torch.norm(diff, p=2) / torch.norm(act1, p=2)).item()
    }

def activation_steering(content_act: torch.Tensor, style_act: torch.Tensor, 
                       strength: float = 0.5) -> torch.Tensor:
    """Steer content activation towards style activation."""
    # Simple version: add style direction to content
    style_direction = style_act - content_act
    return content_act + strength * style_direction

def load_activation_dataset(activation_dir: str, max_samples: Optional[int] = None):
    """Load pre-computed activations from disk."""
    dataset = ActivationDataset(activation_dir, max_samples=max_samples)
    return dataset

# %% [markdown]
# ## Example Usage

# %%
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, default='conf/gpt2_frozen.yaml',
#                        help='Path to configuration file')
#     parser.add_argument('--checkpoint', type=str, default=None,
#                        help='Path to checkpoint file')
#     args = parser.parse_args()
    
#     # Load configuration
#     print(f"Loading config from {args.config}")
from contextlib import nullcontext
with nullcontext():
    cfg = load_config("../conf/gpt2_frozen.yaml")
    print("Configuration loaded!")
    print(OmegaConf.to_yaml(cfg))
    
    # Setup models
    decoder, encoder, orig, tokenizer, device, layer = setup_models(cfg, None)
    print(f"\nModels initialized on {device}")
    print(f"Model: {cfg.model_name}")
    print(f"Layer: {layer}")
    decoder.set_prompt("one -> two; three -> four; five -> six; <embed> ->", tokenizer)
    # %% [markdown]
    # ## Interactive Examples
    # Run these cells to experiment with different functionalities
    
    # %%
    token_pos = -1
    # Example 1: Extract activations from text
    text = "The quick brown fox"
    activation = extract_activation(orig, text, tokenizer, layer)
    print(f"Text: {text}, extract at layer {layer} and token position {token_pos}")
    print(f"Activation shape: {activation.shape}")
    print(f"Activation norm: {torch.norm(activation).item():.4f}")
    
    # %%

    activation = decoder.base.get_input_embeddings().weight[tokenizer.encode("seven", add_special_tokens=False)][0]
    print(f"activation: {activation.shape}")
    # Example 2: Generate text from activation and reconstruct
    generated_text, gen, reconstructed = generate_text_from_activation(
        decoder, encoder, activation*1, tokenizer, max_length=50, tau=1.0, decode_method="generate_soft_kv_cached"
    )
    print(f"Original text: {text}")
    print(f"Generated text: {generated_text}")
    print(f"Reconstruction error: {torch.norm(activation - reconstructed.squeeze()).item():.4f}")
    
    # %%
    # Example 3: Try different generation methods
    print("Comparing generation methods:")
    methods = ["generate_soft", "generate_soft_kv_cached"]
    if cfg.get('decoder', {}).get('use_flash_attention', False):
        methods.append("generate_soft_kv_flash")
    
    for method in methods:
        try:
            gen_text, _, _ = generate_text_from_activation(
                decoder, encoder, activation, tokenizer, 
                max_length=20, tau=1.0, decode_method=method
            )
            print(f"{method}: {gen_text}")
        except Exception as e:
            print(f"{method}: Error - {e}")
    
    # %%
    # Example 4: Interpolate between two different texts
    text1 = "The cat sat on the mat"
    text2 = "Scientists discovered a new planet"
    
    act1 = extract_activation(orig, text1, tokenizer, layer)
    act2 = extract_activation(orig, text2, tokenizer, layer)
    
    print("Interpolation experiment:")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        interpolated = patch_activations(act1, act2, alpha=alpha)
        gen_text, _, _ = generate_text_from_activation(
            decoder, encoder, interpolated, tokenizer, max_length=15, tau=0.8
        )
        print(f"α={alpha}: {gen_text}")
    
    # %%
    # Example 5: Analyze activation differences
    diff_stats = analyze_activations(act1, act2)
    print("Activation analysis:")
    for key, value in diff_stats.items():
        print(f"  {key}: {value:.4f}")
    
    # %%
    # Example 6: Activation steering experiment
    content_text = "The stock market showed"
    style_text = "Once upon a time in a magical"
    
    content_act = extract_activation(orig, content_text, tokenizer, layer)
    style_act = extract_activation(orig, style_text, tokenizer, layer)
    
    print("Activation steering:")
    for strength in [0.0, 0.3, 0.5, 0.8]:
        steered = activation_steering(content_act, style_act, strength=strength)
        gen_text, _, _ = generate_text_from_activation(
            decoder, encoder, steered, tokenizer, max_length=20, tau=0.8
        )
        print(f"Strength {strength}: {gen_text}")
    
    # %%
    # Example 7: Extract from different positions
    text = "Artificial intelligence is transforming the world"
    tokens = tokenizer(text, return_tensors='pt')['input_ids']
    print(f"Text: {text}")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokens[0])}")
    
    print("\nExtracting from different token positions:")
    for pos in range(min(len(tokens[0]), 5)):  # First 5 positions
        act = extract_activation(orig, text, tokenizer, layer, token_pos=pos)
        gen_text, _, _ = generate_text_from_activation(
            decoder, encoder, act, tokenizer, max_length=15, tau=1.0
        )
        token = tokenizer.decode([tokens[0][pos]])
        print(f"Position {pos} ('{token}'): {gen_text}")
    
    # %%
    # Example 8: Temperature exploration
    text = "The future of technology"
    act = extract_activation(orig, text, tokenizer, layer)
    
    print(f"Temperature exploration for: '{text}'")
    for tau in [0.5, 0.8, 1.0, 1.5, 2.0]:
        gen_text, _, _ = generate_text_from_activation(
            decoder, encoder, act, tokenizer, max_length=20, tau=tau
        )
        print(f"τ={tau}: {gen_text}")
    
    # %%
    # Example 9: Load and examine a batch from the dataset
    if hasattr(cfg, 'activation_dumper') and cfg.activation_dumper.get('output_dir'):
        activation_dir = cfg.activation_dumper['output_dir']
        print(f"Loading activations from: {activation_dir}")
        
        try:
            # Load a few samples
            dataset = load_activation_dataset(activation_dir, max_samples=10)
            if len(dataset) > 0:
                # Create a small batch
                from torch.utils.data import DataLoader
                loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate)
                batch = next(iter(loader))
                
                print(f"Loaded batch with keys: {list(batch.keys())}")
                print(f"Batch size: {batch['A'].shape[0]}")
                
                # Run a training step to see losses
                print("\nRunning training step on batch:")
                losses = run_full_train_step(batch, decoder, encoder, orig, cfg, tokenizer, device)
                
                print("Losses:")
                for loss_name, loss_val in losses.items():
                    if isinstance(loss_val, torch.Tensor):
                        print(f"  {loss_name}: {loss_val.item():.4f}")
        except Exception as e:
            print(f"Could not load dataset: {e}")
    
    # %%
    # Example 10: Prompt manipulation
    print(f"Current prompt: {decoder.prompt_text}")
    
    # Try different prompts
    prompts = [
        "Explain: <embed>.",
        "The meaning is: <embed>.",
        "It represents: <embed>.",
        "In other words: <embed>."
    ]
    
    text = "Democracy means"
    act = extract_activation(orig, text, tokenizer, layer)
    
    print(f"\nTrying different prompts for: '{text}'")
    original_prompt = decoder.prompt_text
    for prompt in prompts:
        decoder.set_prompt(prompt, tokenizer)
        gen_text, _, _ = generate_text_from_activation(
            decoder, encoder, act, tokenizer, max_length=20, tau=1.0
        )
        print(f"{prompt} -> {gen_text}")
    
    # Restore original prompt
    decoder.set_prompt(original_prompt, tokenizer)
    
    # %%
    print("\n" + "="*50)
    print("Interactive environment ready!")
    print("Available objects:")
    print("- decoder: Generate text from activations")
    print("- encoder: Reconstruct activations from generated text")
    print("- orig: Original model for extracting activations")
    print("- tokenizer: Encode/decode text")
    print("- device: Current compute device")
    print("- layer: Current layer being analyzed")
    print("\nKey functions:")
    print("- extract_activation(): Get activations from text")
    print("- generate_text_from_activation(): Full pipeline")
    print("- patch_activations(): Interpolate activations")
    print("- analyze_activations(): Compare activations")
    print("- activation_steering(): Steer one activation towards another")
    print("- load_activation_dataset(): Load pre-computed activations")
    print("- run_full_train_step(): See all training losses")
    print("="*50)
# %%

decoder.prompt_text
# %%
