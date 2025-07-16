#!/usr/bin/env python3
"""Check KL loss with real activation data."""

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
from lens.utils.path_utils import resolve_path
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def compute_kl_loss(models, batch, config, device):
    """Compute KL loss on a batch."""
    decoder, encoder, orig_model = models
    
    # Move batch to device
    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    
    with torch.no_grad():
        # Get activations
        activations = batch['A'].float()
        
        # Generate with decoder
        gen_result = decoder.generate_soft(
            activations, 
            max_length=config.get('t_text', 8),
            gumbel_tau=10.0
        )
        
        # Encode back
        reconstructions = encoder(gen_result.generated_text_embeddings)
        
        # Get logits
        orig_logits = orig_model(activations, indices=batch['indices'])
        recon_logits = orig_model(reconstructions, indices=batch['indices'])
        
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
    
    # Load config with defaults
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    
    config_dir = str(Path(config_path).parent.absolute())
    config_name = Path(config_path).stem
    
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
    
    config = OmegaConf.to_container(cfg, resolve=True)
    
    model_name = config['model_name']
    tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer_name", model_name))
    
    # Setup activation directory (same as in training script)
    layer_l = config.get('layer_l', 5)
    cli_activation_dir = config.get('activation_dir')
    base_activation_dir_str = cli_activation_dir if cli_activation_dir is not None else config['activation_dumper']['output_dir']
    base_activation_path = resolve_path(base_activation_dir_str)
    model_name_clean = config['model_name'].replace("/", "_")
    activation_dir = str(base_activation_path.parent / model_name_clean / f"layer_{layer_l}" / base_activation_path.name)
    
    log.info(f"Loading activations from: {activation_dir}")
    
    # Create dataset
    dataset = ActivationDataset(
        activation_dir,
        indices_in_dataset="indices" in config['activation_dumper']['features_to_save'],
        train=True,
        max_samples=100  # Just use a few samples
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    
    # Get a batch
    batch = next(iter(dataloader))
    
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
        log.info(f"Set decoder prompt")
    
    # Move to device
    decoder1 = decoder1.to(device)
    encoder1 = encoder1.to(device)
    orig_model = orig_model.to(device)
    
    # Test fresh models
    kl_fresh, mse_fresh = compute_kl_loss(
        (decoder1, encoder1, orig_model), batch, config, device
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
    kl_loaded, mse_loaded = compute_kl_loss(
        (decoder2, encoder2, orig_model), batch, config, device
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
    kl_device_first, mse_device_first = compute_kl_loss(
        (decoder3, encoder3, orig_model), batch, config, device
    )
    log.info(f"Device-first models - KL: {kl_device_first:.6f}, MSE: {mse_device_first:.6f}")
    
    log.info("\n=== Summary ===")
    log.info(f"KL Fresh:          {kl_fresh:.6f}")
    log.info(f"KL CPU->Device:    {kl_loaded:.6f} (diff: {kl_loaded - kl_fresh:+.6f})")
    log.info(f"KL Device->Load:   {kl_device_first:.6f} (diff: {kl_device_first - kl_fresh:+.6f})")
    
    # Check on multiple batches
    log.info("\n=== Testing on more batches ===")
    total_kl_fresh = 0
    total_kl_loaded = 0
    num_batches = 5
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        kl_f, _ = compute_kl_loss((decoder1, encoder1, orig_model), batch, config, device)
        kl_l, _ = compute_kl_loss((decoder2, encoder2, orig_model), batch, config, device)
        total_kl_fresh += kl_f
        total_kl_loaded += kl_l
        log.info(f"Batch {i}: Fresh KL={kl_f:.6f}, Loaded KL={kl_l:.6f}, diff={kl_l-kl_f:+.6f}")
    
    log.info(f"Average KL Fresh: {total_kl_fresh/num_batches:.6f}")
    log.info(f"Average KL Loaded: {total_kl_loaded/num_batches:.6f}")
    log.info(f"Average difference: {(total_kl_loaded-total_kl_fresh)/num_batches:+.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to checkpoint")
    parser.add_argument("--config", default="conf/gpt2_frozen_e6_wider1p0multigpu2chgprompt.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    main(args.checkpoint, args.config, args.device)