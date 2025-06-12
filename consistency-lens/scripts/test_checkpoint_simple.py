#!/usr/bin/env python3
"""Simple test to check if checkpoint loading changes model outputs."""

import torch
import sys
from pathlib import Path
import argparse
from omegaconf import OmegaConf
import logging

from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.utils.checkpoint_manager import CheckpointManager
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


def main(checkpoint_path):
    """Test checkpoint loading."""
    
    # Hardcode config for GPT-2
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Decoder config from your training
    decoder_cfg = {
        "base_model": True,
        "projection_layer": True,
        "eye_init": True,
        "output_head": False,
        "use_kv_cache": True,
        "per_layer_projections": True,
        "patch_all_layers": True,
        "end_to_end": True,
        "detach_after_each_sample": False
    }
    
    # Encoder config
    encoder_cfg = {
        "base_model": True,
        "use_base_model": True,
        "projection_layer": True,
        "eye_init": True,
        "output_layer": 5,
        "soft_prompt_length": 0
    }
    
    decoder_prompt = "<|endoftext|>Short explanation of <embed>. Language, topic, sentiment, claims, speaker, style, etc:"
    
    log.info("=== Creating Fresh Models ===")
    
    # Create fresh models
    decoder_fresh = Decoder(DecoderConfig(model_name=model_name, **decoder_cfg))
    encoder_fresh = Encoder(EncoderConfig(model_name=model_name, **encoder_cfg))
    
    # Set prompt
    decoder_fresh.set_prompt(decoder_prompt, tokenizer)
    
    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder_fresh = decoder_fresh.to(device)
    encoder_fresh = encoder_fresh.to(device)
    
    # Test with random activation
    test_activation = torch.randn(4, 768).to(device)  # batch_size=4, hidden_size=768
    
    with torch.no_grad():
        # Generate with fresh models
        gen_fresh = decoder_fresh.generate_soft(test_activation, max_length=8, gumbel_tau=10)
        recon_fresh = encoder_fresh(gen_fresh.generated_text_embeddings)
        
        # Get some statistics
        gen_norm_fresh = gen_fresh.generated_text_embeddings.norm(dim=-1).mean()
        recon_norm_fresh = recon_fresh.norm(dim=-1).mean()
        mse_fresh = torch.nn.functional.mse_loss(recon_fresh, test_activation)
    
    log.info(f"Fresh models:")
    log.info(f"  Generated embeddings norm: {gen_norm_fresh:.4f}")
    log.info(f"  Reconstruction norm: {recon_norm_fresh:.4f}")
    log.info(f"  MSE loss: {mse_fresh:.6f}")
    
    log.info("\n=== Loading Checkpoint ===")
    
    # Create new models
    decoder_loaded = Decoder(DecoderConfig(model_name=model_name, **decoder_cfg))
    encoder_loaded = Encoder(EncoderConfig(model_name=model_name, **encoder_cfg))
    
    # Set prompt BEFORE loading
    decoder_loaded.set_prompt(decoder_prompt, tokenizer)
    
    # Load checkpoint
    checkpoint_manager = CheckpointManager({'enabled': True}, log, None, 1)
    rec = checkpoint_manager.load_checkpoint(
        checkpoint_path,
        models={'decoder': decoder_loaded, 'encoder': encoder_loaded},
        optimizer=None,
        map_location='cpu'
    )
    
    log.info(f"Loaded checkpoint from step {rec.get('step', 'unknown')}")
    
    # Move to device AFTER loading
    decoder_loaded = decoder_loaded.to(device)
    encoder_loaded = encoder_loaded.to(device)
    
    # Test with same activation
    with torch.no_grad():
        gen_loaded = decoder_loaded.generate_soft(test_activation, max_length=8, gumbel_tau=10)
        recon_loaded = encoder_loaded(gen_loaded.generated_text_embeddings)
        
        gen_norm_loaded = gen_loaded.generated_text_embeddings.norm(dim=-1).mean()
        recon_norm_loaded = recon_loaded.norm(dim=-1).mean()
        mse_loaded = torch.nn.functional.mse_loss(recon_loaded, test_activation)
    
    log.info(f"Loaded models:")
    log.info(f"  Generated embeddings norm: {gen_norm_loaded:.4f}")
    log.info(f"  Reconstruction norm: {recon_norm_loaded:.4f}")
    log.info(f"  MSE loss: {mse_loaded:.6f}")
    
    log.info("\n=== Comparison ===")
    
    # Compare outputs
    gen_diff = (gen_fresh.generated_text_embeddings - gen_loaded.generated_text_embeddings).abs()
    recon_diff = (recon_fresh - recon_loaded).abs()
    
    log.info(f"Generation difference - max: {gen_diff.max():.6f}, mean: {gen_diff.mean():.6f}")
    log.info(f"Reconstruction difference - max: {recon_diff.max():.6f}, mean: {recon_diff.mean():.6f}")
    log.info(f"MSE difference: {mse_loaded - mse_fresh:+.6f}")
    
    # Check if outputs are exactly the same (they shouldn't be if models are trained)
    if gen_diff.max() < 1e-6:
        log.warning("Generated embeddings are nearly identical - models might not be loading correctly!")
    else:
        log.info("Models are producing different outputs (expected for trained models)")
    
    # Test moving to device first
    log.info("\n=== Testing Device-First Loading ===")
    
    decoder_device_first = Decoder(DecoderConfig(model_name=model_name, **decoder_cfg))
    encoder_device_first = Encoder(EncoderConfig(model_name=model_name, **encoder_cfg))
    decoder_device_first.set_prompt(decoder_prompt, tokenizer)
    
    # Move to device FIRST
    decoder_device_first = decoder_device_first.to(device)
    encoder_device_first = encoder_device_first.to(device)
    
    # Then load
    rec2 = checkpoint_manager.load_checkpoint(
        checkpoint_path,
        models={'decoder': decoder_device_first, 'encoder': encoder_device_first},
        optimizer=None,
        map_location=device
    )
    
    with torch.no_grad():
        gen_device_first = decoder_device_first.generate_soft(test_activation, max_length=8, gumbel_tau=10)
        recon_device_first = encoder_device_first(gen_device_first.generated_text_embeddings)
        mse_device_first = torch.nn.functional.mse_loss(recon_device_first, test_activation)
    
    log.info(f"Device-first MSE: {mse_device_first:.6f}")
    log.info(f"CPU->Device vs Device-first MSE diff: {mse_device_first - mse_loaded:.6f}")
    
    # Compare the two loaded versions
    recon_loaded_diff = (recon_loaded - recon_device_first).abs()
    log.info(f"CPU->Device vs Device-first recon diff - max: {recon_loaded_diff.max():.6f}, mean: {recon_loaded_diff.mean():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    
    main(args.checkpoint)