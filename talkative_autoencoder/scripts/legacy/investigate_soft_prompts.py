#!/usr/bin/env python3
"""Investigate soft prompt embedding mismatch between checkpoint and fresh models."""

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


def main(checkpoint_path, config_path):
    """Investigate soft prompt embeddings in checkpoint vs fresh models."""
    
    # Load config
    from hydra import compose, initialize_config_dir
    config_dir = str(Path(config_path).parent.absolute())
    config_name = Path(config_path).stem
    
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
    
    config = OmegaConf.to_container(cfg, resolve=True)
    
    model_name = config['model_name']
    tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer_name", model_name))
    
    # Extract configs
    decoder_cfg = config['trainable_components']['decoder']
    encoder_cfg = config['trainable_components']['encoder']
    
    log.info("=== Loading raw checkpoint ===")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Check what's in the checkpoint
    if 'encoder' in checkpoint:
        encoder_state = checkpoint['encoder']
        log.info(f"Encoder state_dict keys: {list(encoder_state.keys())}")
        if 'soft_prompt_embeddings' in encoder_state:
            soft_prompt_shape = encoder_state['soft_prompt_embeddings'].shape
            log.info(f"Encoder soft_prompt_embeddings shape: {soft_prompt_shape}")
    
    if 'decoder' in checkpoint:
        decoder_state = checkpoint['decoder']
        prompt_keys = [k for k in decoder_state.keys() if 'prompt' in k]
        log.info(f"Decoder prompt-related keys: {prompt_keys}")
        for key in prompt_keys:
            if isinstance(decoder_state[key], torch.Tensor):
                log.info(f"  {key} shape: {decoder_state[key].shape}")
    
    log.info("\n=== Creating fresh models ===")
    
    # Initialize fresh models
    decoder_fresh = Decoder(DecoderConfig(model_name=model_name, **decoder_cfg))
    encoder_fresh = Encoder(EncoderConfig(model_name=model_name, **encoder_cfg))
    
    # Check encoder config
    log.info(f"\nEncoder config soft_prompt_length: {encoder_cfg.get('soft_prompt_length', 0)}")
    
    # Check what's in fresh encoder
    encoder_params = {name: param.shape for name, param in encoder_fresh.named_parameters()}
    log.info(f"\nFresh encoder parameters: {encoder_params}")
    
    # Set decoder prompt
    decoder_prompt = config.get('decoder_prompt', '')
    if decoder_prompt:
        log.info(f"\nSetting decoder prompt: {decoder_prompt}")
        decoder_fresh.set_prompt(decoder_prompt, tokenizer)
        
        # Check decoder after setting prompt
        decoder_params = {name: param.shape for name, param in decoder_fresh.named_parameters()}
        log.info(f"\nFresh decoder parameters after set_prompt: {decoder_params}")
    
    # Now try loading the checkpoint
    log.info("\n=== Loading checkpoint into fresh models ===")
    
    decoder_load = Decoder(DecoderConfig(model_name=model_name, **decoder_cfg))
    encoder_load = Encoder(EncoderConfig(model_name=model_name, **encoder_cfg))
    
    # Set prompt before loading
    if decoder_prompt:
        decoder_load.set_prompt(decoder_prompt, tokenizer)
    
    try:
        checkpoint_manager = CheckpointManager({'enabled': True}, log, None, 1)
        rec = checkpoint_manager.load_checkpoint(
            checkpoint_path,
            models={'decoder': decoder_load, 'encoder': encoder_load},
            optimizer=None,
            map_location='cpu'
        )
        log.info(f"Successfully loaded checkpoint from step {rec.get('step', 'unknown')}")
    except Exception as e:
        log.error(f"Error loading checkpoint: {e}")
        log.info("\n=== Attempting manual state_dict loading ===")
        
        # Try loading encoder state manually
        if 'encoder' in checkpoint:
            try:
                missing, unexpected = encoder_load.load_state_dict(checkpoint['encoder'], strict=False)
                log.info(f"Encoder - Missing keys: {missing}")
                log.info(f"Encoder - Unexpected keys: {unexpected}")
            except Exception as e2:
                log.error(f"Manual encoder load failed: {e2}")
        
        # Try loading decoder state manually
        if 'decoder' in checkpoint:
            try:
                missing, unexpected = decoder_load.load_state_dict(checkpoint['decoder'], strict=False)
                log.info(f"Decoder - Missing keys: {missing}")
                log.info(f"Decoder - Unexpected keys: {unexpected}")
            except Exception as e2:
                log.error(f"Manual decoder load failed: {e2}")
    
    # Check if soft_prompt_length is 0 but checkpoint has soft prompts
    if encoder_cfg.get('soft_prompt_length', 0) == 0 and 'soft_prompt_embeddings' in checkpoint.get('encoder', {}):
        log.warning("\n!!! MISMATCH: Checkpoint has soft_prompt_embeddings but config has soft_prompt_length=0")
        log.warning("This is likely causing the KL loss jump!")
        
        # Suggest fix
        soft_prompt_len = checkpoint['encoder']['soft_prompt_embeddings'].shape[0]
        log.info(f"\nSuggested fix: Set soft_prompt_length={soft_prompt_len} in encoder config")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default="conf/gpt2_frozen_e6_wider1p0multigpu2chgprompt.yaml")
    args = parser.parse_args()
    
    main(args.checkpoint, args.config)