#!/usr/bin/env python3
"""Diagnose why KL loss jumps when resuming from checkpoint."""

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


def detailed_kl_analysis(orig_logits, recon_logits, prefix=""):
    """Detailed analysis of KL divergence components."""
    # Get probabilities
    orig_probs = torch.nn.functional.softmax(orig_logits, dim=-1)
    recon_log_probs = torch.nn.functional.log_softmax(recon_logits, dim=-1)
    recon_probs = torch.nn.functional.softmax(recon_logits, dim=-1)
    
    # Compute KL
    kl = torch.nn.functional.kl_div(recon_log_probs, orig_probs, reduction='none')
    kl_per_token = kl.sum(dim=-1)  # Sum over vocab
    kl_mean = kl_per_token.mean()
    
    # Compute other metrics
    # Cross entropy from original to reconstructed
    ce = -(orig_probs * recon_log_probs).sum(dim=-1).mean()
    
    # Entropy of original distribution
    orig_entropy = -(orig_probs * torch.log(orig_probs + 1e-10)).sum(dim=-1).mean()
    
    # L2 distance between logits
    logit_l2 = (orig_logits - recon_logits).pow(2).mean()
    
    # Top-1 accuracy
    orig_top1 = orig_logits.argmax(dim=-1)
    recon_top1 = recon_logits.argmax(dim=-1)
    top1_acc = (orig_top1 == recon_top1).float().mean()
    
    # Temperature of distributions (lower = more peaked)
    orig_temp = orig_probs.max(dim=-1)[0].mean()
    recon_temp = recon_probs.max(dim=-1)[0].mean()
    
    log.info(f"{prefix}KL divergence: {kl_mean:.6f}")
    log.info(f"{prefix}Cross entropy: {ce:.6f}")
    log.info(f"{prefix}Original entropy: {orig_entropy:.6f}")
    log.info(f"{prefix}Logit L2 distance: {logit_l2:.6f}")
    log.info(f"{prefix}Top-1 accuracy: {top1_acc:.4f}")
    log.info(f"{prefix}Orig max prob: {orig_temp:.4f}, Recon max prob: {recon_temp:.4f}")
    
    return {
        'kl': kl_mean.item(),
        'ce': ce.item(),
        'orig_entropy': orig_entropy.item(),
        'logit_l2': logit_l2.item(),
        'top1_acc': top1_acc.item(),
        'orig_temp': orig_temp.item(),
        'recon_temp': recon_temp.item()
    }


def test_checkpoint_kl(checkpoint_path, config_path, device='cuda'):
    """Test KL divergence behavior with checkpoint loading."""
    
    # Load config
    cfg = OmegaConf.load(config_path)
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Get model name from config - handle nested structure
    if 'model_name' in config:
        model_name = config['model_name']
    elif 'model' in config and 'name' in config['model']:
        model_name = config['model']['name']
    else:
        # Default for GPT-2 configs
        model_name = 'gpt2'
    
    # Load tokenizer
    tokenizer_name = config.get("tokenizer_name", model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Initialize models
    decoder_cfg = config['trainable_components']['decoder']
    encoder_cfg = config['trainable_components']['encoder']
    
    log.info("\n=== Test 1: Fresh models moved to device ===")
    decoder1 = Decoder(DecoderConfig(model_name=model_name, **decoder_cfg))
    encoder1 = Encoder(EncoderConfig(model_name=model_name, **encoder_cfg))
    orig_model1 = OrigWrapper(model_name, load_in_8bit=False)
    
    if 'decoder_prompt' in config and config['decoder_prompt']:
        decoder1.set_prompt(config['decoder_prompt'], tokenizer)
    
    # Move to device
    decoder1 = decoder1.to(device)
    encoder1 = encoder1.to(device)
    orig_model1 = orig_model1.to(device)
    
    # Get test data
    activation_dir = config.get('activation_dir', config['activation_dumper']['output_dir'])
    dataset = ActivationDataset(
        activation_dir,
        indices_in_dataset="indices" in config['activation_dumper']['features_to_save'],
        train=True,
        max_samples=100
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    batch = next(iter(dataloader))
    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    
    # Test with fresh models
    with torch.no_grad():
        activations = batch['A'].float()
        gen1 = decoder1.generate_soft(activations, max_length=config.get('t_text', 8), gumbel_tau=10)
        recon1 = encoder1(gen1.generated_text_embeddings)
        
        orig_logits1 = orig_model1(activations, indices=batch['indices'])
        recon_logits1 = orig_model1(recon1, indices=batch['indices'])
    
    kl_metrics1 = detailed_kl_analysis(orig_logits1, recon_logits1, "Fresh models: ")
    
    log.info("\n=== Test 2: Load checkpoint on CPU, then move to device ===")
    decoder2 = Decoder(DecoderConfig(model_name=model_name, **decoder_cfg))
    encoder2 = Encoder(EncoderConfig(model_name=model_name, **encoder_cfg))
    
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
    
    # Now move to device
    decoder2 = decoder2.to(device)
    encoder2 = encoder2.to(device)
    
    # Test with loaded models
    with torch.no_grad():
        gen2 = decoder2.generate_soft(activations, max_length=config.get('t_text', 8), gumbel_tau=10)
        recon2 = encoder2(gen2.generated_text_embeddings)
        
        recon_logits2 = orig_model1(recon2, indices=batch['indices'])
    
    kl_metrics2 = detailed_kl_analysis(orig_logits1, recon_logits2, "Loaded models: ")
    
    log.info("\n=== Test 3: Move to device first, then load checkpoint ===")
    decoder3 = Decoder(DecoderConfig(model_name=model_name, **decoder_cfg))
    encoder3 = Encoder(EncoderConfig(model_name=model_name, **encoder_cfg))
    
    if 'decoder_prompt' in config and config['decoder_prompt']:
        decoder3.set_prompt(config['decoder_prompt'], tokenizer)
    
    # Move to device FIRST
    decoder3 = decoder3.to(device)
    encoder3 = encoder3.to(device)
    
    # Then load checkpoint
    rec3 = checkpoint_manager.load_checkpoint(
        checkpoint_path,
        models={'decoder': decoder3, 'encoder': encoder3},
        optimizer=None,
        map_location=device
    )
    
    # Test
    with torch.no_grad():
        gen3 = decoder3.generate_soft(activations, max_length=config.get('t_text', 8), gumbel_tau=10)
        recon3 = encoder3(gen3.generated_text_embeddings)
        
        recon_logits3 = orig_model1(recon3, indices=batch['indices'])
    
    kl_metrics3 = detailed_kl_analysis(orig_logits1, recon_logits3, "Device-first models: ")
    
    log.info("\n=== Comparison ===")
    log.info(f"KL Fresh vs CPU->GPU: {kl_metrics2['kl'] - kl_metrics1['kl']:.6f}")
    log.info(f"KL Fresh vs GPU->load: {kl_metrics3['kl'] - kl_metrics1['kl']:.6f}")
    log.info(f"KL CPU->GPU vs GPU->load: {kl_metrics3['kl'] - kl_metrics2['kl']:.6f}")
    
    # Check if reconstructions are different
    log.info("\n=== Reconstruction differences ===")
    recon_diff_1_2 = (recon1 - recon2).abs()
    recon_diff_1_3 = (recon1 - recon3).abs()
    recon_diff_2_3 = (recon2 - recon3).abs()
    
    log.info(f"Recon diff Fresh vs CPU->GPU: max={recon_diff_1_2.max():.6f}, mean={recon_diff_1_2.mean():.6f}")
    log.info(f"Recon diff Fresh vs GPU->load: max={recon_diff_1_3.max():.6f}, mean={recon_diff_1_3.mean():.6f}")
    log.info(f"Recon diff CPU->GPU vs GPU->load: max={recon_diff_2_3.max():.6f}, mean={recon_diff_2_3.mean():.6f}")
    
    # Check parameter differences
    log.info("\n=== Parameter check ===")
    for (n1, p1), (n2, p2), (n3, p3) in zip(
        decoder1.named_parameters(), 
        decoder2.named_parameters(),
        decoder3.named_parameters()
    ):
        diff_1_2 = (p1 - p2).abs().max().item()
        diff_1_3 = (p1 - p3).abs().max().item()
        diff_2_3 = (p2 - p3).abs().max().item()
        
        if max(diff_1_2, diff_1_3, diff_2_3) > 1e-6:
            log.info(f"Decoder {n1}: diff 1-2={diff_1_2:.2e}, 1-3={diff_1_3:.2e}, 2-3={diff_2_3:.2e}")
    
    # Test with different tau values
    log.info("\n=== Testing different tau values ===")
    for tau in [1.0, 5.0, 10.0, 20.0]:
        with torch.no_grad():
            gen_tau = decoder2.generate_soft(activations[:4], max_length=config.get('t_text', 8), gumbel_tau=tau)
            recon_tau = encoder2(gen_tau.generated_text_embeddings)
            recon_logits_tau = orig_model1(recon_tau, indices=batch['indices'][:4])
            
        kl_tau = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(recon_logits_tau, dim=-1),
            torch.nn.functional.softmax(orig_logits1[:4], dim=-1),
            reduction='batchmean'
        )
        log.info(f"  tau={tau}: KL={kl_tau:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--config", default="conf/config.yaml", help="Path to config file")
    parser.add_argument("--device", default="cuda", help="Device to use")
    args = parser.parse_args()
    
    test_checkpoint_kl(args.checkpoint, args.config, args.device)