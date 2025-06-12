#!/usr/bin/env python3
"""
Test script to verify checkpoint saving and loading works correctly with prompt initialization.
Tests that prompts are properly saved and restored.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile

# Add the parent directory to the path so we can import lens modules
sys.path.insert(0, str(Path(__file__).parent))

from lens.utils.checkpoint_manager import CheckpointManager
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from transformers import AutoTokenizer
import logging


def test_prompt_preservation():
    """Test that prompts are preserved across checkpoint save/load."""
    print("="*60)
    print("Testing Prompt Preservation in Checkpoints")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    model_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Test prompt texts
        decoder_prompt_text = "The activation is <embed>:"
        encoder_prompt_text = "Activation analysis:"
        
        # Create decoder with soft prompt
        print("\n1. Creating decoder with soft prompt...")
        decoder_config = DecoderConfig(
            model_name=model_name,
            n_prompt_tokens=10,  # Enable soft prompts
            base_model=False,
            projection_layer=True,
            output_head=False,
            embedding_head=False,
            eye_init=True,
            trainable_prompts=True
        )
        decoder = Decoder(decoder_config).to(device)
        
        # Set decoder prompt from text
        decoder.set_prompt(decoder_prompt_text, tokenizer)
        print(f"Decoder prompt set to: '{decoder_prompt_text}'")
        
        # Get the decoder prompt embeddings for comparison
        decoder_prompt_before = decoder.prompt_left_emb.data.clone() if hasattr(decoder, 'prompt_left_emb') and decoder.prompt_left_emb is not None else None
        
        # Create encoder with soft prompt
        print("\n2. Creating encoder with soft prompt...")
        encoder_config = EncoderConfig(
            model_name=model_name,
            output_layer=6,
            base_model=False,
            projection_layer=True,
            use_base_model=True,
            embedding_head=False,
            eye_init=True,
            soft_prompt_length=0,  # Will be determined by text
            trainable_soft_prompt=True,
            soft_prompt_init_text=encoder_prompt_text  # Initialize from text
        )
        encoder = Encoder(encoder_config)
        # Initialize soft prompt from text if specified
        if encoder_config.soft_prompt_init_text:
            encoder.set_soft_prompt_from_text(encoder_config.soft_prompt_init_text, tokenizer)
        encoder = encoder.to(device)
        print(f"Encoder prompt set to: '{encoder_prompt_text}'")
        print(f"Encoder has soft_prompt_embeddings: {hasattr(encoder, 'soft_prompt_embeddings') and encoder.soft_prompt_embeddings is not None}")
        
        # Get the encoder prompt embeddings for comparison
        encoder_prompt_before = encoder.soft_prompt_embeddings.data.clone() if hasattr(encoder, 'soft_prompt_embeddings') and encoder.soft_prompt_embeddings is not None else None
        
        # Setup checkpoint manager
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        config = {
            'checkpoint': {
                'enabled': True,
                'output_dir': str(checkpoint_dir),
                'save_components': {
                    'models': True,
                    'optimizer': False,
                    'scheduler': False,
                    'config': True,
                    'metrics': False
                }
            }
        }
        
        checkpoint_manager = CheckpointManager(config, logger)
        
        # Save checkpoint
        print("\n3. Saving checkpoint...")
        saved_path = checkpoint_manager.save_checkpoint(
            step=100,
            epoch=1,
            models={'decoder': decoder, 'encoder': encoder},
            config={
                'decoder_prompt': decoder_prompt_text,
                'encoder_prompt': encoder_prompt_text
            }
        )
        
        print(f"Checkpoint saved to: {saved_path}")
        
        # Create new models with random initialization
        print("\n4. Creating fresh models with random initialization...")
        decoder2_config = DecoderConfig(
            model_name=model_name,
            n_prompt_tokens=10,  # Same length but not initialized
            base_model=False,
            projection_layer=True,
            output_head=False,
            embedding_head=False,
            eye_init=True,
            trainable_prompts=True
        )
        decoder2 = Decoder(decoder2_config).to(device)
        
        # Don't set any prompt - let it be random
        # Create dummy prompt parameters with random values (matching the checkpoint sizes)
        decoder2.prompt_left_emb = nn.Parameter(torch.randn(4, 768, device=device) * 0.1)
        decoder2.prompt_right_emb = nn.Parameter(torch.randn(1, 768, device=device) * 0.1)
        # Also need to set prompt_ids to match
        decoder2.prompt_ids.resize_(6)  # 4 + 1 + 1 = 6 total
        
        encoder2_config = EncoderConfig(
            model_name=model_name,
            output_layer=6,
            base_model=False,
            projection_layer=True,
            use_base_model=True,
            embedding_head=False,
            eye_init=True,
            soft_prompt_length=4,  # Same length as saved checkpoint
            trainable_soft_prompt=True
        )
        encoder2 = Encoder(encoder2_config).to(device)
        
        # Get their initial prompt values (should be random)
        decoder_prompt_random = decoder2.prompt_left_emb.data.clone() if hasattr(decoder2, 'prompt_left_emb') and decoder2.prompt_left_emb is not None else None
        encoder_prompt_random = encoder2.soft_prompt_embeddings.data.clone() if hasattr(encoder2, 'soft_prompt_embeddings') and encoder2.soft_prompt_embeddings is not None else None
        
        # Load checkpoint
        print("\n5. Loading checkpoint...")
        loaded_data = checkpoint_manager.load_checkpoint(
            str(saved_path),
            models={'decoder': decoder2, 'encoder': encoder2},
            map_location=device
        )
        
        # Get loaded prompt values
        decoder_prompt_after = decoder2.prompt_left_emb.data.clone() if hasattr(decoder2, 'prompt_left_emb') and decoder2.prompt_left_emb is not None else None
        encoder_prompt_after = encoder2.soft_prompt_embeddings.data.clone() if hasattr(encoder2, 'soft_prompt_embeddings') and encoder2.soft_prompt_embeddings is not None else None
        
        # Verify prompts were loaded correctly
        print("\n6. Verifying prompt preservation...")
        
        # Initialize match variables
        decoder_match = False
        encoder_match = False
        
        # Debug: print what we have
        print(f"decoder_prompt_before: {decoder_prompt_before.shape if decoder_prompt_before is not None else None}")
        print(f"decoder_prompt_after: {decoder_prompt_after.shape if decoder_prompt_after is not None else None}")
        print(f"decoder_prompt_random: {decoder_prompt_random.shape if decoder_prompt_random is not None else None}")
        print(f"encoder_prompt_before: {encoder_prompt_before.shape if encoder_prompt_before is not None else None}")
        print(f"encoder_prompt_after: {encoder_prompt_after.shape if encoder_prompt_after is not None else None}")
        print(f"encoder_prompt_random: {encoder_prompt_random.shape if encoder_prompt_random is not None else None}")
        
        # Check decoder prompt
        if decoder_prompt_before is not None and decoder_prompt_after is not None:
            decoder_match = torch.allclose(decoder_prompt_before, decoder_prompt_after, rtol=1e-5, atol=1e-8)
            decoder_changed = not torch.allclose(decoder_prompt_random, decoder_prompt_after, rtol=1e-5, atol=1e-8)
            
            if decoder_match and decoder_changed:
                print("✅ Decoder prompt: Successfully loaded (matches original, differs from random)")
            else:
                print("❌ Decoder prompt: Loading failed")
                if not decoder_match:
                    print(f"  - Loaded prompt doesn't match original (max diff: {(decoder_prompt_before - decoder_prompt_after).abs().max().item():.6f})")
                if not decoder_changed:
                    print("  - Loaded prompt is same as random initialization")
        else:
            print("❌ Decoder prompt: Not found in model")
        
        # Check encoder prompt  
        if encoder_prompt_before is not None and encoder_prompt_after is not None:
            encoder_match = torch.allclose(encoder_prompt_before, encoder_prompt_after, rtol=1e-5, atol=1e-8)
            encoder_changed = not torch.allclose(encoder_prompt_random, encoder_prompt_after, rtol=1e-5, atol=1e-8)
            
            if encoder_match and encoder_changed:
                print("✅ Encoder prompt: Successfully loaded (matches original, differs from random)")
            else:
                print("❌ Encoder prompt: Loading failed")
                if not encoder_match:
                    print(f"  - Loaded prompt doesn't match original (max diff: {(encoder_prompt_before - encoder_prompt_after).abs().max().item():.6f})")
                if not encoder_changed:
                    print("  - Loaded prompt is same as random initialization")
        else:
            print("❌ Encoder prompt: Not found in model")
        
        # Test generation with loaded prompts
        print("\n7. Testing generation with loaded prompts...")
        try:
            # Test decoder generation
            dummy_input = torch.randn(1, 768, device=device)
            output = decoder2.generate_soft(dummy_input, max_length=8, gumbel_tau=1.0)
            print("✅ Decoder generation works with loaded prompt")
            
            # Test encoder
            encoder_out = encoder2(output.generated_text_embeddings)
            print("✅ Encoder works with loaded prompt")
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return False
        
        # Check config was saved
        if 'config' in loaded_data:
            config_match = (
                loaded_data['config'].get('decoder_prompt') == decoder_prompt_text and
                loaded_data['config'].get('encoder_prompt') == encoder_prompt_text
            )
            if config_match:
                print("✅ Prompt texts preserved in config")
            else:
                print("❌ Prompt texts not preserved in config")
        
        return decoder_match and encoder_match


def test_checkpoint_with_different_prompt_configs():
    """Test checkpoint loading with different prompt configurations."""
    print("\n" + "="*60)
    print("Testing Checkpoint Compatibility with Different Prompt Configs")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'gpt2'
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Create models with prompts
        print("\n1. Creating models with prompts...")
        decoder_config = DecoderConfig(
            model_name=model_name,
            n_prompt_tokens=10,
            base_model=False,
            projection_layer=True,
            output_head=False,
            embedding_head=False,
            eye_init=True,
            trainable_prompts=True
        )
        decoder = Decoder(decoder_config).to(device)
        
        encoder_config = EncoderConfig(
            model_name=model_name,
            output_layer=6,
            base_model=False,
            projection_layer=True,
            use_base_model=True,
            embedding_head=False,
            eye_init=True,
            soft_prompt_length=8,
            trainable_soft_prompt=True
        )
        encoder = Encoder(encoder_config).to(device)
        
        # Save checkpoint
        logger = logging.getLogger(__name__)
        config = {
            'checkpoint': {
                'enabled': True,
                'output_dir': str(checkpoint_dir)
            }
        }
        checkpoint_manager = CheckpointManager(config, logger)
        
        saved_path = checkpoint_manager.save_checkpoint(
            step=100,
            epoch=1,
            models={'decoder': decoder, 'encoder': encoder}
        )
        
        # Test 1: Load into models without prompts (should fail gracefully)
        print("\n2. Testing load into models without prompts...")
        decoder_no_prompt = Decoder(DecoderConfig(
            model_name=model_name,
            n_prompt_tokens=0,  # No prompt
            base_model=False,
            projection_layer=True,
            output_head=False,
            embedding_head=False,
            eye_init=True,
            trainable_prompts=True
        )).to(device)
        
        encoder_no_prompt = Encoder(EncoderConfig(
            model_name=model_name,
            output_layer=6,
            base_model=False,
            projection_layer=True,
            use_base_model=True,
            embedding_head=False,
            eye_init=True,
            soft_prompt_length=0,  # No prompt
            trainable_soft_prompt=True
        )).to(device)
        
        try:
            loaded_data = checkpoint_manager.load_checkpoint(
                str(saved_path),
                models={'decoder': decoder_no_prompt, 'encoder': encoder_no_prompt},
                map_location=device
            )
            print("❌ Unexpected: Checkpoint loaded successfully despite incompatible prompt configs")
            return False
        except Exception as e:
            print(f"✅ Expected failure when loading prompts into models without prompts: {type(e).__name__}")
            # This is expected behavior - can't load prompt parameters into models without prompts
        
        # Test 2: Load into models with different prompt lengths
        print("\n3. Testing load into models with different prompt lengths...")
        decoder_diff_prompt = Decoder(DecoderConfig(
            model_name=model_name,
            n_prompt_tokens=5,  # Different length
            base_model=False,
            projection_layer=True,
            output_head=False,
            embedding_head=False,
            eye_init=True,
            trainable_prompts=True
        )).to(device)
        
        encoder_diff_prompt = Encoder(EncoderConfig(
            model_name=model_name,
            output_layer=6,
            base_model=False,
            projection_layer=True,
            use_base_model=True,
            embedding_head=False,
            eye_init=True,
            soft_prompt_length=12,  # Different length
            trainable_soft_prompt=True
        )).to(device)
        
        try:
            loaded_data = checkpoint_manager.load_checkpoint(
                str(saved_path),
                models={'decoder': decoder_diff_prompt, 'encoder': encoder_diff_prompt},
                map_location=device
            )
            print("❌ Unexpected: Checkpoint loaded successfully despite mismatched prompt lengths")
            return False
        except Exception as e:
            print(f"✅ Expected failure when loading prompts with mismatched lengths: {type(e).__name__}")
            # This is expected behavior - can't load prompt parameters with different shapes
        
        return True


if __name__ == "__main__":
    # Run tests
    prompt_test_pass = test_prompt_preservation()
    compat_test_pass = test_checkpoint_with_different_prompt_configs()
    
    # Summary
    print("\n" + "="*60)
    print("PROMPT CHECKPOINT TEST SUMMARY")
    print("="*60)
    
    if prompt_test_pass and compat_test_pass:
        print("✅ All prompt tests passed!")
        sys.exit(0)
    else:
        print("❌ Some prompt tests failed!")
        sys.exit(1)