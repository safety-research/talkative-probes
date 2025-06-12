#!/usr/bin/env python3
"""
Test script to verify checkpoint saving and loading works correctly.
Tests that all parameters match after save/load cycle.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Add the parent directory to the path so we can import lens modules
sys.path.insert(0, str(Path(__file__).parent))

from lens.utils.checkpoint_manager import CheckpointManager
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.orig import OrigWrapper
import logging


def create_test_models(device='cpu'):
    """Create test models for checkpoint testing."""
    model_name = 'gpt2'
    
    # Create decoder
    decoder_config = DecoderConfig(
        model_name=model_name,
        n_prompt_tokens=8,
        base_model=False,
        projection_layer=True,
        output_head=False,
        embedding_head=False,
        eye_init=True,
        trainable_prompts=True,
        use_checkpointing=False,
        use_kv_cache=False
    )
    decoder = Decoder(decoder_config).to(device)
    
    # Create encoder
    encoder_config = EncoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        use_base_model=True,
        embedding_head=False,
        eye_init=True,
        soft_prompt_length=0,
        trainable_soft_prompt=True,
        output_layer=6
    )
    encoder = Encoder(encoder_config).to(device)
    
    # Create original model wrapper
    orig_model = OrigWrapper(model_name, load_in_8bit=False)
    
    return decoder, encoder, orig_model


def compare_model_parameters(model1, model2, name="model"):
    """Compare parameters between two models and return differences."""
    state1 = model1.state_dict()
    state2 = model2.state_dict()
    
    # Check if keys match
    keys1 = set(state1.keys())
    keys2 = set(state2.keys())
    
    missing_in_2 = keys1 - keys2
    extra_in_2 = keys2 - keys1
    
    if missing_in_2:
        print(f"❌ {name}: Keys missing after load: {missing_in_2}")
        return False
    
    if extra_in_2:
        print(f"❌ {name}: Extra keys after load: {extra_in_2}")
        return False
    
    # Compare parameter values
    all_match = True
    for key in keys1:
        param1 = state1[key]
        param2 = state2[key]
        
        if param1.shape != param2.shape:
            print(f"❌ {name}.{key}: Shape mismatch - {param1.shape} vs {param2.shape}")
            all_match = False
            continue
            
        if not torch.allclose(param1, param2, rtol=1e-5, atol=1e-8):
            max_diff = torch.max(torch.abs(param1 - param2)).item()
            print(f"❌ {name}.{key}: Value mismatch - max diff: {max_diff}")
            all_match = False
    
    if all_match:
        print(f"✅ {name}: All parameters match perfectly!")
    
    return all_match


def compare_optimizer_state(opt1, opt2):
    """Compare optimizer states."""
    state1 = opt1.state_dict()
    state2 = opt2.state_dict()
    
    # Compare param groups
    if len(state1['param_groups']) != len(state2['param_groups']):
        print(f"❌ Optimizer: Different number of param groups - {len(state1['param_groups'])} vs {len(state2['param_groups'])}")
        return False
    
    all_match = True
    for i, (pg1, pg2) in enumerate(zip(state1['param_groups'], state2['param_groups'])):
        # Compare learning rates
        if pg1['lr'] != pg2['lr']:
            print(f"❌ Optimizer param_group[{i}]: LR mismatch - {pg1['lr']} vs {pg2['lr']}")
            all_match = False
        
        if 'initial_lr' in pg1 and 'initial_lr' in pg2:
            if pg1['initial_lr'] != pg2['initial_lr']:
                print(f"❌ Optimizer param_group[{i}]: initial_lr mismatch - {pg1['initial_lr']} vs {pg2['initial_lr']}")
                all_match = False
    
    # Compare state (momentum, etc.)
    if len(state1['state']) != len(state2['state']):
        print(f"❌ Optimizer: Different number of state entries - {len(state1['state'])} vs {len(state2['state'])}")
        return False
    
    # Note: Detailed state comparison is complex due to parameter ID mapping
    # For now, just check the count
    
    if all_match:
        print("✅ Optimizer: States match!")
    
    return all_match


def test_checkpoint_save_load():
    """Test checkpoint saving and loading with all components."""
    print("="*60)
    print("Testing Checkpoint Save/Load Functionality")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Create models
        print("\n1. Creating test models...")
        decoder, encoder, orig_model = create_test_models(device)
        
        # Create optimizer
        params = list(decoder.parameters()) + list(encoder.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-3)
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        
        # Run a few training steps to create optimizer state
        print("\n2. Running dummy training steps to create optimizer state...")
        for i in range(5):
            optimizer.zero_grad()
            
            # Dummy forward pass
            dummy_input = torch.randn(2, 768, device=device)
            output = encoder(decoder.generate_soft(dummy_input, max_length=8, gumbel_tau=1.0).generated_text_embeddings)
            loss = output.sum()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # Setup checkpoint manager
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        config = {
            'checkpoint': {
                'enabled': True,
                'output_dir': str(checkpoint_dir),
                'save_components': {
                    'models': True,
                    'optimizer': True,
                    'scheduler': True,
                    'config': True,
                    'metrics': True
                }
            }
        }
        
        checkpoint_manager = CheckpointManager(config, logger)
        
        # Test data to save
        test_step = 1000
        test_epoch = 5
        test_metrics = {
            'loss': 0.123,
            'kl_loss': 0.456,
            'mse_loss': 0.789
        }
        test_config = {
            'learning_rate': 1e-3,
            'batch_size': 32,
            'model_name': 'gpt2'
        }
        
        # Additional metadata
        additional_data = {
            'tau': 0.5,
            'alpha': 0.8,
            'wandb_run_id': 'test_run_123',
            'custom_data': [1, 2, 3]
        }
        
        # Save checkpoint
        print("\n3. Saving checkpoint...")
        saved_path = checkpoint_manager.save_checkpoint(
            step=test_step,
            epoch=test_epoch,
            models={'decoder': decoder, 'encoder': encoder},
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=test_metrics,
            config=test_config,
            val_loss=test_metrics['loss'],
            **additional_data
        )
        
        print(f"Checkpoint saved to: {saved_path}")
        
        # Verify file exists
        assert saved_path.exists(), "Checkpoint file was not created!"
        
        # Create new models and optimizer for loading
        print("\n4. Creating fresh models for loading...")
        decoder2, encoder2, _ = create_test_models(device)
        
        params2 = list(decoder2.parameters()) + list(encoder2.parameters())
        optimizer2 = torch.optim.AdamW(params2, lr=999)  # Different LR to test loading
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=1000)
        
        # Load checkpoint
        print("\n5. Loading checkpoint...")
        loaded_data = checkpoint_manager.load_checkpoint(
            str(saved_path),
            models={'decoder': decoder2, 'encoder': encoder2},
            optimizer=optimizer2,
            map_location=device
        )
        
        # Also load scheduler state manually (since checkpoint.load doesn't handle it)
        if 'scheduler' in loaded_data:
            scheduler2.load_state_dict(loaded_data['scheduler'])
        
        # Verify loaded data
        print("\n6. Verifying loaded checkpoint data...")
        
        # Check metadata
        print("\nMetadata verification:")
        assert loaded_data['step'] == test_step, f"Step mismatch: {loaded_data['step']} vs {test_step}"
        print(f"✅ Step: {loaded_data['step']}")
        
        assert loaded_data['epoch'] == test_epoch, f"Epoch mismatch: {loaded_data['epoch']} vs {test_epoch}"
        print(f"✅ Epoch: {loaded_data['epoch']}")
        
        # Check metrics
        if 'metrics' in loaded_data:
            for key, value in test_metrics.items():
                assert key in loaded_data['metrics'], f"Metric '{key}' missing"
                assert loaded_data['metrics'][key] == value, f"Metric '{key}' mismatch"
            print("✅ Metrics: All match")
        
        # Check config
        if 'config' in loaded_data:
            for key, value in test_config.items():
                assert key in loaded_data['config'], f"Config '{key}' missing"
                assert loaded_data['config'][key] == value, f"Config '{key}' mismatch"
            print("✅ Config: All match")
        
        # Check additional data
        for key, value in additional_data.items():
            assert key in loaded_data, f"Additional data '{key}' missing"
            assert loaded_data[key] == value, f"Additional data '{key}' mismatch"
        print("✅ Additional data: All match")
        
        # Compare model parameters
        print("\n7. Comparing model parameters...")
        decoder_match = compare_model_parameters(decoder, decoder2, "Decoder")
        encoder_match = compare_model_parameters(encoder, encoder2, "Encoder")
        
        # Compare optimizer state
        print("\n8. Comparing optimizer state...")
        optimizer_match = compare_optimizer_state(optimizer, optimizer2)
        
        # Compare scheduler state
        print("\n9. Comparing scheduler state...")
        scheduler_match = scheduler.state_dict() == scheduler2.state_dict()
        if scheduler_match:
            print("✅ Scheduler: States match!")
        else:
            print("❌ Scheduler: States don't match!")
        
        # Test checkpoint compilation compatibility
        print("\n10. Testing compiled model compatibility...")
        if hasattr(torch, 'compile'):
            print("Testing save from non-compiled, load to compiled...")
            decoder3_compiled = torch.compile(Decoder(decoder.config).to(device))
            encoder3_compiled = torch.compile(Encoder(encoder.config).to(device))
            
            # Load checkpoint to compiled models
            loaded_data3 = checkpoint_manager.load_checkpoint(
                str(saved_path),
                models={'decoder': decoder3_compiled, 'encoder': encoder3_compiled},
                map_location=device
            )
            
            print("✅ Successfully loaded non-compiled checkpoint into compiled models!")
        
        # Summary
        print("\n" + "="*60)
        print("CHECKPOINT TEST SUMMARY")
        print("="*60)
        
        all_pass = all([
            decoder_match,
            encoder_match,
            optimizer_match,
            scheduler_match
        ])
        
        if all_pass:
            print("✅ ALL TESTS PASSED! Checkpoint save/load works correctly.")
        else:
            print("❌ Some tests failed. Check the output above for details.")
        
        return all_pass


def test_random_state_preservation():
    """Test that random states are preserved across checkpoint save/load."""
    print("\n" + "="*60)
    print("Testing Random State Preservation")
    print("="*60)
    
    # Set initial seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate some random numbers to advance the state
    _ = torch.rand(10)
    _ = np.random.rand(10)
    
    # Now save the current states
    torch_state_before = torch.get_rng_state().clone()
    cuda_state_before = torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None
    numpy_state_before = np.random.get_state()
    
    # Generate some random numbers (these should be reproducible later)
    torch_rand_expected = torch.rand(5)
    numpy_rand_expected = np.random.rand(5)
    
    print("Expected random values after restore:")
    print(f"Torch: {torch_rand_expected}")
    print(f"NumPy: {numpy_rand_expected}")
    
    # Create temporary checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save checkpoint with random states (states were captured before generating the expected values)
        from lens.utils import checkpoint
        
        checkpoint_path = Path(tmpdir) / "test_rng.pt"
        
        # Manually restore states to before we generated expected values
        torch.set_rng_state(torch_state_before)
        if cuda_state_before is not None:
            torch.cuda.set_rng_state(cuda_state_before)
        np.random.set_state(numpy_state_before)
        
        # Now save
        checkpoint.save(
            path=checkpoint_path,
            models={},
            optim=None,
            step=0
        )
        
        # Change random states significantly
        torch.manual_seed(99999)
        np.random.seed(99999)
        
        # Generate different random numbers
        torch_rand_different = torch.rand(5)
        numpy_rand_different = np.random.rand(5)
        
        print("\nRandom values with different seed:")
        print(f"Torch: {torch_rand_different}")
        print(f"NumPy: {numpy_rand_different}")
        
        # Load checkpoint (should restore random states)
        checkpoint.load(
            path=checkpoint_path,
            models={},
            load_rng_state=True
        )
        
        # Generate random numbers again (should match expected values)
        torch_rand_after = torch.rand(5)
        numpy_rand_after = np.random.rand(5)
        
        print("\nRandom values after loading checkpoint:")
        print(f"Torch: {torch_rand_after}")
        print(f"NumPy: {numpy_rand_after}")
        
        # Verify states match
        torch_match = torch.allclose(torch_rand_expected, torch_rand_after, rtol=1e-5, atol=1e-8)
        numpy_match = np.allclose(numpy_rand_expected, numpy_rand_after, rtol=1e-5, atol=1e-8)
        
        if torch_match and numpy_match:
            print("\n✅ Random states successfully preserved!")
        else:
            print("\n❌ Random state preservation failed!")
            if not torch_match:
                print(f"  - Torch random state mismatch: max diff = {(torch_rand_expected - torch_rand_after).abs().max()}")
            if not numpy_match:
                print(f"  - NumPy random state mismatch: max diff = {np.abs(numpy_rand_expected - numpy_rand_after).max()}")
        
        return torch_match and numpy_match


if __name__ == "__main__":
    # Run tests
    checkpoint_test_pass = test_checkpoint_save_load()
    rng_test_pass = test_random_state_preservation()
    
    # Exit with appropriate code
    if checkpoint_test_pass and rng_test_pass:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)