#!/usr/bin/env python3
"""Diagnostic script to identify why loss is higher when resuming from checkpoint."""

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
from lens.data.dataset import ActivationDataset
from lens.training.schedules import get_schedule_value
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def compute_losses_on_batch(models, batch, config, device, step=0):
    """Compute losses on a single batch."""
    decoder, encoder, orig_model = models
    
    # Get schedule values
    tau = get_schedule_value(
        config['gumbel_tau_schedule'], 
        step, 
        config.get('max_train_steps', 1000),
        current_epoch=0,
        steps_per_epoch=None,
        grad_accum_steps=1
    )
    alpha = get_schedule_value(
        config['alpha_schedule'],
        step,
        config.get('max_train_steps', 1000),
        current_epoch=0,
        steps_per_epoch=None,
        grad_accum_steps=1
    )
    
    # Move batch to device
    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    
    # Forward pass
    with torch.no_grad():
        # Get activations
        activations = batch['A'].float()
        
        # Generate text with decoder
        gen_result = decoder.generate_soft(
            activations, 
            max_length=config.get('t_text', 8),
            gumbel_tau=tau
        )
        
        # Encode back to activations
        reconstructions = encoder(gen_result.generated_text_embeddings)
        
        # Compute MSE loss
        mse_loss = torch.nn.functional.mse_loss(reconstructions, activations)
        
        # Get logits from both original and reconstructed
        orig_logits = orig_model(activations, indices=batch['indices'])
        recon_logits = orig_model(reconstructions, indices=batch['indices'])
        
        # Compute KL divergence
        kl_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(recon_logits, dim=-1),
            torch.nn.functional.softmax(orig_logits, dim=-1),
            reduction='batchmean'
        )
        
        # Language modeling loss (simplified)
        lm_loss = torch.tensor(0.0).to(device)  # Placeholder
        
        # Total loss
        total_loss = kl_loss * config.get('kl_base_weight', 1.0) + alpha * lm_loss + mse_loss * config.get('mse_weight', 0.0)
        
    return {
        'total': total_loss.item(),
        'mse': mse_loss.item(),
        'kl': kl_loss.item(),
        'lm': lm_loss.item()
    }


def diagnose_checkpoint(checkpoint_path, config_path, device='cuda'):
    """Diagnose checkpoint loading issues."""
    
    # Load config
    cfg = OmegaConf.load(config_path)
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Load tokenizer
    tokenizer_name = config.get("tokenizer_name", config['model_name'])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Initialize models on CPU
    model_name = config['model_name']
    decoder_cfg = config['trainable_components']['decoder']
    encoder_cfg = config['trainable_components']['encoder']
    
    log.info("Initializing models on CPU...")
    decoder = Decoder(DecoderConfig(model_name=model_name, **decoder_cfg))
    encoder = Encoder(EncoderConfig(model_name=model_name, **encoder_cfg))
    orig_model = OrigWrapper(model_name, load_in_8bit=False)
    
    # Set prompts if configured
    if 'decoder_prompt' in config and config['decoder_prompt']:
        decoder.set_prompt(config['decoder_prompt'], tokenizer)
    
    # Create dataset
    activation_dir = config.get('activation_dir', config['activation_dumper']['output_dir'])
    dataset = ActivationDataset(
        activation_dir,
        indices_in_dataset="indices" in config['activation_dumper']['features_to_save'],
        train=True,
        max_samples=100  # Just use a few samples for testing
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    
    # Get a batch
    batch = next(iter(dataloader))
    
    log.info("\n=== Testing Fresh Models ===")
    # Move models to device
    decoder_fresh = decoder.to(device)
    encoder_fresh = encoder.to(device)
    orig_model_fresh = orig_model.to(device)
    
    # Compute losses with fresh models
    losses_fresh = compute_losses_on_batch(
        (decoder_fresh, encoder_fresh, orig_model_fresh),
        batch, config, device
    )
    log.info(f"Fresh model losses: {losses_fresh}")
    
    # Save model outputs for comparison
    with torch.no_grad():
        activations = batch['A'].float().to(device)
        gen_fresh = decoder_fresh.generate_soft(activations[:1], max_length=8, gumbel_tau=10)
        recon_fresh = encoder_fresh(gen_fresh.generated_text_embeddings)
    
    log.info("\n=== Loading Checkpoint ===")
    # Now load checkpoint
    checkpoint_manager = CheckpointManager({'enabled': True}, log, None, 1)
    
    # Reset models to CPU state
    decoder_cpu = Decoder(DecoderConfig(model_name=model_name, **decoder_cfg))
    encoder_cpu = Encoder(EncoderConfig(model_name=model_name, **encoder_cfg))
    if 'decoder_prompt' in config and config['decoder_prompt']:
        decoder_cpu.set_prompt(config['decoder_prompt'], tokenizer)
    
    # Load checkpoint on CPU
    log.info(f"Loading checkpoint from {checkpoint_path}")
    rec = checkpoint_manager.load_checkpoint(
        checkpoint_path,
        models={'decoder': decoder_cpu, 'encoder': encoder_cpu},
        optimizer=None,
        map_location='cpu'
    )
    log.info(f"Checkpoint step: {rec.get('step', 'unknown')}")
    
    # Move to device after loading
    decoder_loaded = decoder_cpu.to(device)
    encoder_loaded = encoder_cpu.to(device)
    orig_model_loaded = orig_model.to(device)  # This wasn't in checkpoint
    
    # Compute losses with loaded models
    losses_loaded = compute_losses_on_batch(
        (decoder_loaded, encoder_loaded, orig_model_loaded),
        batch, config, device, step=rec.get('step', 0)
    )
    log.info(f"Loaded model losses: {losses_loaded}")
    
    # Compare outputs
    with torch.no_grad():
        gen_loaded = decoder_loaded.generate_soft(activations[:1], max_length=8, gumbel_tau=10)
        recon_loaded = encoder_loaded(gen_loaded.generated_text_embeddings)
    
    log.info("\n=== Comparison ===")
    log.info(f"Loss difference: {losses_loaded['total'] - losses_fresh['total']:.6f}")
    log.info(f"MSE difference: {losses_loaded['mse'] - losses_fresh['mse']:.6f}")
    log.info(f"KL difference: {losses_loaded['kl'] - losses_fresh['kl']:.6f}")
    
    # Check parameter differences
    log.info("\n=== Parameter Comparison ===")
    for (name_fresh, param_fresh), (name_loaded, param_loaded) in zip(
        decoder_fresh.named_parameters(), decoder_loaded.named_parameters()
    ):
        if name_fresh == name_loaded:
            diff = (param_fresh - param_loaded).abs().max().item()
            if diff > 1e-6:
                log.info(f"Decoder {name_fresh}: max diff = {diff}")
    
    # Check reconstruction differences
    recon_diff = (recon_fresh - recon_loaded).abs().max().item()
    log.info(f"\nReconstruction max diff: {recon_diff}")
    
    # Try loading to device directly
    log.info("\n=== Testing Direct Device Loading ===")
    decoder_direct = Decoder(DecoderConfig(model_name=model_name, **decoder_cfg))
    encoder_direct = Encoder(EncoderConfig(model_name=model_name, **encoder_cfg))
    if 'decoder_prompt' in config and config['decoder_prompt']:
        decoder_direct.set_prompt(config['decoder_prompt'], tokenizer)
    
    # Load directly to device
    rec_direct = checkpoint_manager.load_checkpoint(
        checkpoint_path,
        models={'decoder': decoder_direct, 'encoder': encoder_direct},
        optimizer=None,
        map_location=device
    )
    
    decoder_direct = decoder_direct.to(device)
    encoder_direct = encoder_direct.to(device)
    
    losses_direct = compute_losses_on_batch(
        (decoder_direct, encoder_direct, orig_model),
        batch, config, device, step=rec.get('step', 0)
    )
    log.info(f"Direct device loading losses: {losses_direct}")
    log.info(f"Direct vs CPU->GPU loss diff: {losses_direct['total'] - losses_loaded['total']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--config", default="conf/config.yaml", help="Path to config file")
    parser.add_argument("--device", default="cuda", help="Device to use")
    args = parser.parse_args()
    
    diagnose_checkpoint(args.checkpoint, args.config, args.device)