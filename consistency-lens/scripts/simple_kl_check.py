#!/usr/bin/env python3
"""Simple check for KL loss jump when loading checkpoint."""

import torch
import numpy as np
from pathlib import Path
import argparse
from omegaconf import OmegaConf
import logging

from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.orig import OrigWrapper
from lens.utils.checkpoint_manager import CheckpointManager
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def test_kl_on_random_data(models, config, device, batch_size=8, seq_len=16):
    """Test KL loss on random data."""
    decoder, encoder, orig_model = models
    
    # Create random activations (simulating layer 5 outputs)
    activations = torch.randn(batch_size, seq_len, 768).to(device)  # GPT-2 hidden size
    indices = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(device)
    
    with torch.no_grad():
        # Generate with decoder
        gen_result = decoder.generate_soft(
            activations, 
            max_length=config.get('t_text', 8),
            gumbel_tau=10.0
        )
        
        # Encode back
        reconstructions = encoder(gen_result.generated_text_embeddings)
        
        # Get logits
        orig_logits = orig_model(activations, indices=indices)
        recon_logits = orig_model(reconstructions, indices=indices)
        
        # Compute KL
        kl_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(recon_logits, dim=-1),
            torch.nn.functional.softmax(orig_logits, dim=-1),
            reduction='batchmean'
        )
        
        # MSE for reference
        mse_loss = torch.nn.functional.mse_loss(reconstructions, activations)
        
    return kl_loss.item(), mse_loss.item()


def main(checkpoint_path, config_path, device='cuda'):
    """Test checkpoint loading and KL loss."""
    
    # Load config
    cfg = OmegaConf.load(config_path)
    with open(config_path.replace('wider1p0multigpu2chgprompt', 'frozen'), 'r') as f:
        # Load base config to get model_name
        base_cfg = OmegaConf.load(f)
    
    # Merge configs
    cfg = OmegaConf.merge(base_cfg, cfg)
    config = OmegaConf.to_container(cfg, resolve=True)
    
    model_name = config.get('model_name', 'openai-community/gpt2')
    tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer_name", model_name))
    
    # Extract configs
    decoder_cfg = config['trainable_components']['decoder']
    encoder_cfg = config['trainable_components']['encoder']
    
    log.info("=== Test 1: Fresh models ===")
    
    # Initialize fresh models
    decoder1 = Decoder(DecoderConfig(model_name=model_name, **decoder_cfg))
    encoder1 = Encoder(EncoderConfig(model_name=model_name, **encoder_cfg))
    orig_model = OrigWrapper(model_name, load_in_8bit=False)
    
    # Set decoder prompt
    if 'decoder_prompt' in config and config['decoder_prompt']:
        decoder1.set_prompt(config['decoder_prompt'], tokenizer)
        log.info(f"Set decoder prompt: '{config['decoder_prompt'][:50]}...'")
    
    # Move to device
    decoder1 = decoder1.to(device)
    encoder1 = encoder1.to(device)
    orig_model = orig_model.to(device)
    
    # Test fresh models
    kl_fresh, mse_fresh = test_kl_on_random_data(
        (decoder1, encoder1, orig_model), config, device
    )
    log.info(f"Fresh models - KL: {kl_fresh:.6f}, MSE: {mse_fresh:.6f}")
    
    log.info("\n=== Test 2: Load checkpoint (CPU first) ===")
    
    # Initialize new models
    decoder2 = Decoder(DecoderConfig(model_name=model_name, **decoder_cfg))
    encoder2 = Encoder(EncoderConfig(model_name=model_name, **encoder_cfg))
    
    # Set decoder prompt BEFORE loading
    if 'decoder_prompt' in config and config['decoder_prompt']:
        decoder2.set_prompt(config['decoder_prompt'], tokenizer)
    
    # Load checkpoint on CPU
    checkpoint_manager = CheckpointManager({'enabled': True}, log, None, 1)
    rec = checkpoint_manager.load_checkpoint(
        checkpoint_path,
        models={'decoder': decoder2, 'encoder': encoder2},
        optimizer=None,
        map_location='cpu'
    )
    log.info(f"Loaded checkpoint from step {rec.get('step', 'unknown')}")
    
    # Move to device AFTER loading
    decoder2 = decoder2.to(device)
    encoder2 = encoder2.to(device)
    
    # Test loaded models
    kl_loaded, mse_loaded = test_kl_on_random_data(
        (decoder2, encoder2, orig_model), config, device
    )
    log.info(f"Loaded models - KL: {kl_loaded:.6f}, MSE: {mse_loaded:.6f}")
    
    log.info("\n=== Test 3: Load checkpoint (device first) ===")
    
    # Initialize new models
    decoder3 = Decoder(DecoderConfig(model_name=model_name, **decoder_cfg))
    encoder3 = Encoder(EncoderConfig(model_name=model_name, **encoder_cfg))
    
    # Set decoder prompt
    if 'decoder_prompt' in config and config['decoder_prompt']:
        decoder3.set_prompt(config['decoder_prompt'], tokenizer)
    
    # Move to device BEFORE loading
    decoder3 = decoder3.to(device)
    encoder3 = encoder3.to(device)
    
    # Load checkpoint
    rec3 = checkpoint_manager.load_checkpoint(
        checkpoint_path,
        models={'decoder': decoder3, 'encoder': encoder3},
        optimizer=None,
        map_location=device
    )
    
    # Test
    kl_device_first, mse_device_first = test_kl_on_random_data(
        (decoder3, encoder3, orig_model), config, device
    )
    log.info(f"Device-first models - KL: {kl_device_first:.6f}, MSE: {mse_device_first:.6f}")
    
    log.info("\n=== Summary ===")
    log.info(f"KL Fresh:          {kl_fresh:.6f}")
    log.info(f"KL CPU->Device:    {kl_loaded:.6f} (diff: {kl_loaded - kl_fresh:+.6f})")
    log.info(f"KL Device->Load:   {kl_device_first:.6f} (diff: {kl_device_first - kl_fresh:+.6f})")
    log.info(f"MSE Fresh:         {mse_fresh:.6f}")
    log.info(f"MSE CPU->Device:   {mse_loaded:.6f} (diff: {mse_loaded - mse_fresh:+.6f})")
    log.info(f"MSE Device->Load:  {mse_device_first:.6f} (diff: {mse_device_first - mse_fresh:+.6f})")
    
    # Check parameters
    log.info("\n=== Parameter check ===")
    for (n1, p1), (n2, p2), (n3, p3) in zip(
        decoder1.named_parameters()[:5],  # Just check first few
        decoder2.named_parameters()[:5],
        decoder3.named_parameters()[:5]
    ):
        diff_1_2 = (p1 - p2).abs().max().item()
        diff_1_3 = (p1 - p3).abs().max().item()
        diff_2_3 = (p2 - p3).abs().max().item()
        
        if max(diff_1_2, diff_1_3, diff_2_3) > 1e-7:
            log.info(f"{n1}: diff fresh-loaded={diff_1_2:.2e}, fresh-device={diff_1_3:.2e}, loaded-device={diff_2_3:.2e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to checkpoint")
    parser.add_argument("--config", default="conf/gpt2_frozen_e6_wider1p0multigpu2chgprompt.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    main(args.checkpoint, args.config, args.device)