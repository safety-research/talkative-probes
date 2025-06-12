#!/usr/bin/env python3
"""Check if soft prompts are being saved/loaded correctly."""

import torch
from pathlib import Path
import argparse
from omegaconf import OmegaConf
import logging

from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.utils.checkpoint_manager import CheckpointManager
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def check_soft_prompts(checkpoint_path, config_path):
    """Check soft prompt parameters in checkpoint."""
    
    # Load config
    cfg = OmegaConf.load(config_path)
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Load checkpoint raw
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    log.info("=== Checkpoint Contents ===")
    log.info(f"Keys: {list(checkpoint.keys())}")
    log.info(f"Step: {checkpoint.get('step', 'unknown')}")
    
    # Check decoder state dict
    if 'models' in checkpoint and 'decoder' in checkpoint['models']:
        decoder_state = checkpoint['models']['decoder']
        log.info("\n=== Decoder State Dict ===")
        for key in decoder_state.keys():
            if 'prompt' in key.lower() or 'soft' in key.lower():
                param = decoder_state[key]
                log.info(f"{key}: shape={param.shape}, mean={param.mean():.6f}, std={param.std():.6f}")
    
    # Check encoder state dict  
    if 'models' in checkpoint and 'encoder' in checkpoint['models']:
        encoder_state = checkpoint['models']['encoder']
        log.info("\n=== Encoder State Dict ===")
        for key in encoder_state.keys():
            if 'prompt' in key.lower() or 'soft' in key.lower() or 'embed' in key.lower():
                param = encoder_state[key]
                log.info(f"{key}: shape={param.shape}, mean={param.mean():.6f}, std={param.std():.6f}")
    
    # Now test loading
    log.info("\n=== Testing Checkpoint Loading ===")
    
    # Initialize models
    model_name = config['model_name']
    decoder_cfg = config['trainable_components']['decoder']
    encoder_cfg = config['trainable_components']['encoder']
    tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer_name", model_name))
    
    # Create fresh models
    decoder_fresh = Decoder(DecoderConfig(model_name=model_name, **decoder_cfg))
    encoder_fresh = Encoder(EncoderConfig(model_name=model_name, **encoder_cfg))
    
    # Set prompts
    if 'decoder_prompt' in config and config['decoder_prompt']:
        decoder_fresh.set_prompt(config['decoder_prompt'], tokenizer)
        log.info(f"Set decoder prompt: '{config['decoder_prompt']}'")
    
    # Log fresh soft prompt values
    log.info("\n=== Fresh Model Soft Prompts ===")
    for name, param in decoder_fresh.named_parameters():
        if 'prompt' in name.lower() or 'soft' in name.lower():
            log.info(f"Decoder {name}: shape={param.shape}, mean={param.mean():.6f}, std={param.std():.6f}")
    
    for name, param in encoder_fresh.named_parameters():
        if 'prompt' in name.lower() or 'soft' in name.lower() or 'embed' in name.lower():
            log.info(f"Encoder {name}: shape={param.shape}, mean={param.mean():.6f}, std={param.std():.6f}")
    
    # Load checkpoint
    checkpoint_manager = CheckpointManager({'enabled': True}, log, None, 1)
    rec = checkpoint_manager.load_checkpoint(
        checkpoint_path,
        models={'decoder': decoder_fresh, 'encoder': encoder_fresh},
        optimizer=None,
        map_location='cpu'
    )
    
    # Log loaded soft prompt values
    log.info("\n=== After Loading Checkpoint ===")
    for name, param in decoder_fresh.named_parameters():
        if 'prompt' in name.lower() or 'soft' in name.lower():
            log.info(f"Decoder {name}: shape={param.shape}, mean={param.mean():.6f}, std={param.std():.6f}")
            
            # Check if it matches checkpoint
            if 'models' in checkpoint and 'decoder' in checkpoint['models']:
                if name in checkpoint['models']['decoder']:
                    ckpt_param = checkpoint['models']['decoder'][name]
                    diff = (param - ckpt_param).abs().max().item()
                    log.info(f"  Diff from checkpoint: {diff}")
    
    for name, param in encoder_fresh.named_parameters():
        if 'prompt' in name.lower() or 'soft' in name.lower() or 'embed' in name.lower():
            log.info(f"Encoder {name}: shape={param.shape}, mean={param.mean():.6f}, std={param.std():.6f}")
            
            # Check if it matches checkpoint
            if 'models' in checkpoint and 'encoder' in checkpoint['models']:
                if name in checkpoint['models']['encoder']:
                    ckpt_param = checkpoint['models']['encoder'][name]
                    diff = (param - ckpt_param).abs().max().item()
                    log.info(f"  Diff from checkpoint: {diff}")
    
    # Test prompt re-initialization
    log.info("\n=== Testing Prompt Re-initialization ===")
    if 'decoder_prompt' in config and config['decoder_prompt']:
        # Get soft prompts before re-init
        soft_prompts_before = {}
        for name, param in decoder_fresh.named_parameters():
            if 'soft_prompt' in name:
                soft_prompts_before[name] = param.clone()
        
        # Re-initialize
        decoder_fresh.set_prompt(config['decoder_prompt'], tokenizer)
        
        # Check differences
        for name, param in decoder_fresh.named_parameters():
            if 'soft_prompt' in name and name in soft_prompts_before:
                diff = (param - soft_prompts_before[name]).abs().max().item()
                log.info(f"Prompt re-init changed {name}: max diff = {diff}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--config", default="conf/config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    check_soft_prompts(args.checkpoint, args.config)