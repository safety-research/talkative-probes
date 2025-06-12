#!/usr/bin/env python3
"""Simple KL test using the exact same setup as training."""

import torch
import sys
from pathlib import Path
import argparse
from omegaconf import OmegaConf
import logging

# Import from training script
sys.path.insert(0, str(Path(__file__).parent))
import importlib
train_module = importlib.import_module("01_train")

from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.orig import OrigWrapper
from lens.utils.checkpoint_manager import CheckpointManager
from lens.utils.path_utils import resolve_path
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def test_checkpoint_kl(checkpoint_path, config_path, device='cuda'):
    """Test KL with checkpoint loading."""
    
    # Load config with defaults like in training
    from hydra import compose, initialize_config_dir
    config_dir = str(Path(config_path).parent.absolute())
    config_name = Path(config_path).stem
    
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
    
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Get paths like in training script
    model_name = config['model_name']
    layer_l = config.get('layer_l', 5)
    cli_activation_dir = config.get('activation_dir')
    base_activation_dir_str = cli_activation_dir if cli_activation_dir is not None else config['activation_dumper']['output_dir']
    base_activation_path = resolve_path(base_activation_dir_str)
    model_name_clean = config['model_name'].replace("/", "_")
    activation_dir = str(base_activation_path.parent / model_name_clean / f"layer_{layer_l}" / base_activation_path.name)
    
    log.info(f"Using activation dir: {activation_dir}")
    
    # Load tokenizer
    tokenizer_name = config.get("tokenizer_name", model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Prepare dataset using the same function as training
    train_loader, val_loader, train_ds, val_ds = train_module._prepare_dataloaders(
        config=config,
        activation_dir=activation_dir,
        effective_val_activation_dir=None,
        max_train_samples_req=100,  # Just a few samples
        max_val_samples_req=None,
        log=log
    )
    
    # Get configs
    decoder_cfg = config['trainable_components']['decoder']
    encoder_cfg = config['trainable_components']['encoder']
    
    # Get a batch
    batch = next(iter(train_loader))
    
    log.info("\n=== Test 1: Fresh models ===")
    
    # Initialize fresh models
    decoder1 = Decoder(DecoderConfig(model_name=model_name, **decoder_cfg))
    encoder1 = Encoder(EncoderConfig(model_name=model_name, **encoder_cfg))
    orig_model1 = OrigWrapper(model_name, load_in_8bit=False)
    
    # Set decoder prompt
    if 'decoder_prompt' in config and config['decoder_prompt']:
        decoder1.set_prompt(config['decoder_prompt'], tokenizer)
    
    # Move to device
    decoder1 = decoder1.to(device)
    encoder1 = encoder1.to(device)
    orig_model1 = orig_model1.to(device)
    
    # Move batch to device
    batch_dev = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    
    # Test with fresh models
    with torch.no_grad():
        activations = batch_dev['A'].float()
        B = activations.shape[0]
        
        # Create indices like in training
        indices = torch.arange(B, device=device)
        
        gen1 = decoder1.generate_soft(activations, max_length=config.get('t_text', 8), gumbel_tau=10)
        recon1 = encoder1(gen1.generated_text_embeddings)
        
        # The training loop uses orig.model not orig directly
        # And it passes embeddings through the model, not activations
        # For KL test, we need to get logits from positions corresponding to the activations
        # This is complex - let's simplify by just comparing reconstructions
        pass
        
        # For now, just compare MSE between activations and reconstructions
        # KL computation in training is more complex with full forward passes
        mse1 = torch.nn.functional.mse_loss(recon1, activations)
        
        # Simple norm comparison
        act_norm1 = activations.norm(dim=-1).mean()
        recon_norm1 = recon1.norm(dim=-1).mean()
    
    log.info(f"Fresh models - KL: {kl1:.6f}, MSE: {mse1:.6f}")
    
    log.info("\n=== Test 2: Load checkpoint ===")
    
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
    
    # Test
    with torch.no_grad():
        gen2 = decoder2.generate_soft(activations, max_length=config.get('t_text', 8), gumbel_tau=10)
        recon2 = encoder2(gen2.generated_text_embeddings)
        
        recon_logits2 = orig_model1(recon2, indices=indices)
        
        kl2 = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(recon_logits2, dim=-1),
            torch.nn.functional.softmax(orig_logits1, dim=-1),
            reduction='batchmean'
        )
        mse2 = torch.nn.functional.mse_loss(recon2, activations)
    
    log.info(f"Loaded models - KL: {kl2:.6f}, MSE: {mse2:.6f}")
    log.info(f"KL difference: {kl2 - kl1:+.6f}")
    log.info(f"MSE difference: {mse2 - mse1:+.6f}")
    
    # Check reconstruction differences
    recon_diff = (recon1 - recon2).abs()
    log.info(f"Reconstruction diff - max: {recon_diff.max():.6f}, mean: {recon_diff.mean():.6f}")
    
    # Test on more batches
    log.info("\n=== Testing on 5 batches ===")
    kl_diffs = []
    for i, batch in enumerate(train_loader):
        if i >= 5:
            break
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        with torch.no_grad():
            A = batch['A'].float()
            
            # Fresh
            g1 = decoder1.generate_soft(A, max_length=config.get('t_text', 8), gumbel_tau=10)
            r1 = encoder1(g1.generated_text_embeddings)
            B = A.shape[0]
            indices = torch.arange(B, device=device)
            orig_l = orig_model1(A, indices=indices)
            rec_l1 = orig_model1(r1, indices=indices)
            kl_fresh = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(rec_l1, dim=-1),
                torch.nn.functional.softmax(orig_l, dim=-1),
                reduction='batchmean'
            )
            
            # Loaded
            g2 = decoder2.generate_soft(A, max_length=config.get('t_text', 8), gumbel_tau=10)
            r2 = encoder2(g2.generated_text_embeddings)
            rec_l2 = orig_model1(r2, indices=indices)
            kl_loaded = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(rec_l2, dim=-1),
                torch.nn.functional.softmax(orig_l, dim=-1),
                reduction='batchmean'
            )
            
            diff = kl_loaded - kl_fresh
            kl_diffs.append(diff.item())
            log.info(f"Batch {i}: Fresh={kl_fresh:.6f}, Loaded={kl_loaded:.6f}, Diff={diff:+.6f}")
    
    avg_diff = sum(kl_diffs) / len(kl_diffs)
    log.info(f"Average KL difference: {avg_diff:+.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default="conf/gpt2_frozen_e6_wider1p0multigpu2chgprompt.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    test_checkpoint_kl(args.checkpoint, args.config, args.device)