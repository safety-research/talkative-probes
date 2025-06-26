#!/usr/bin/env python3
"""Simple test to verify cache regeneration logging without loading models."""

import logging
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

import torch
from datasets import Dataset as HFDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lens.data.on_the_fly_datasets import RankInMemoryTrainingCache

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


def create_tiny_pretokenized_dataset(num_samples: int, seq_len: int = 128) -> HFDataset:
    """Create a tiny pretokenized dataset for testing."""
    data = {
        'input_ids': [
            [i % 50000 + j for j in range(seq_len)]  # Create distinct patterns
            for i in range(num_samples)
        ]
    }
    return HFDataset.from_dict(data)


def test_cycling_detection_logging():
    """Test that cycling detection and logging works correctly."""
    print("=== Testing Cycling Detection and Logging ===\n")
    
    # Create mock model that simulates activation generation
    mock_orig_wrapper = Mock()
    mock_orig_wrapper.to = Mock(return_value=None)
    
    # Mock the activation generation to return dummy data
    def mock_get_activations(input_ids, layer_idx, min_pos_to_select_from, no_grad):
        batch_size = input_ids.shape[0]
        # Return dummy activations and positions
        activations = torch.randn(batch_size, 768)  # Simulate hidden dimension
        positions = torch.randint(min_pos_to_select_from, input_ids.shape[1], (batch_size,))
        return activations, positions
    
    mock_orig_wrapper.get_activations_at_positions = Mock(side_effect=mock_get_activations)
    
    # Test configuration
    dataset_sizes = [50, 100, 200]  # Different dataset sizes
    cache_size = 150  # Fixed cache size
    
    for dataset_size in dataset_sizes:
        print(f"\n--- Test with dataset size {dataset_size}, cache size {cache_size} ---")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and save pretokenized dataset
            dataset = create_tiny_pretokenized_dataset(dataset_size)
            dataset_path = Path(temp_dir) / "pretokenized"
            dataset.save_to_disk(str(dataset_path))
            
            # Create logger that captures output
            captured_logs = []
            
            class LogCapture:
                def __init__(self, captured_list):
                    self.captured = captured_list
                
                def info(self, msg):
                    self.captured.append(('INFO', msg))
                    print(f"[INFO] {msg}")
                
                def warning(self, msg):
                    self.captured.append(('WARNING', msg))
                    print(f"[WARNING] {msg}")
                
                def error(self, msg):
                    self.captured.append(('ERROR', msg))
                    print(f"[ERROR] {msg}")
            
            capture_logger = LogCapture(captured_logs)
            
            # Create cache instance
            cache = RankInMemoryTrainingCache(
                orig_model_for_gen=mock_orig_wrapper,
                pretok_dataset_path=str(dataset_path.parent),
                pretok_split_name="pretokenized",
                on_the_fly_config={
                    'layer_l': 5,
                    'min_pos': 5,
                    'generation_batch_size': 10
                },
                generation_device=torch.device('cpu'),  # Use CPU for testing
                rank=0,
                world_size=1,
                initial_cache_size=0,  # Don't fill initially
                logger=capture_logger
            )
            
            # Regenerate cache
            cache.regenerate_cache(num_samples_to_generate=cache_size)
            
            # Check logs for cycling warnings (only WARNING level)
            cycling_warnings = [(level, log) for level, log in captured_logs if level == 'WARNING' and ('cycle' in log.lower() or 'cycling' in log.lower())]
            
            print(f"\n  Dataset size: {dataset_size}")
            print(f"  Cache size: {cache_size}")
            print(f"  Expected cycles: {cache_size / dataset_size:.1f}")
            print(f"  Cycling warnings found: {len(cycling_warnings)}")
            
            # Verify expectations
            if cache_size > dataset_size:
                assert len(cycling_warnings) > 0, f"Expected cycling warnings but found none"
                print("  ✅ Cycling warnings correctly generated")
            else:
                assert len(cycling_warnings) == 0, f"Unexpected cycling warnings: {cycling_warnings}"
                print("  ✅ No cycling warnings (as expected)")
            
            # Display sample warning messages
            if cycling_warnings:
                print("\n  Sample warning messages:")
                for level, msg in cycling_warnings[:3]:
                    print(f"    - {msg}")


def test_exact_cycle_counting():
    """Test that cycle counting is accurate."""
    print("\n\n=== Testing Exact Cycle Counting ===\n")
    
    # Create mock model
    mock_orig_wrapper = Mock()
    mock_orig_wrapper.to = Mock(return_value=None)
    
    call_count = 0
    samples_generated = []
    
    def mock_get_activations(input_ids, layer_idx, min_pos_to_select_from, no_grad):
        nonlocal call_count
        call_count += 1
        batch_size = input_ids.shape[0]
        # Track which samples are being used
        for i in range(batch_size):
            sample_id = input_ids[i, 0].item()  # Use first token as ID
            samples_generated.append(sample_id)
        # Return dummy data
        activations = torch.randn(batch_size, 768)
        positions = torch.randint(min_pos_to_select_from, input_ids.shape[1], (batch_size,))
        return activations, positions
    
    mock_orig_wrapper.get_activations_at_positions = Mock(side_effect=mock_get_activations)
    
    # Test with exact multiples
    test_cases = [
        (20, 20, 1.0),   # Exact match - 1 cycle
        (20, 40, 2.0),   # Exact 2 cycles
        (20, 50, 2.5),   # 2.5 cycles
        (30, 100, 3.33), # 3.33 cycles
    ]
    
    for dataset_size, cache_size, expected_cycles in test_cases:
        print(f"\n--- Dataset: {dataset_size}, Cache: {cache_size}, Expected cycles: {expected_cycles:.2f} ---")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Reset tracking
            call_count = 0
            samples_generated.clear()
            
            # Create dataset
            dataset = create_tiny_pretokenized_dataset(dataset_size)
            dataset_path = Path(temp_dir) / "pretokenized"
            dataset.save_to_disk(str(dataset_path))
            
            # Create cache
            cache = RankInMemoryTrainingCache(
                orig_model_for_gen=mock_orig_wrapper,
                pretok_dataset_path=str(dataset_path.parent),
                pretok_split_name="pretokenized",
                on_the_fly_config={
                    'layer_l': 5,
                    'min_pos': 5,
                    'generation_batch_size': 5  # Small batch for accurate counting
                },
                generation_device=torch.device('cpu'),
                rank=0,
                world_size=1,
                initial_cache_size=0,
                logger=log
            )
            
            # Regenerate
            cache.regenerate_cache(num_samples_to_generate=cache_size)
            
            # Analyze sample usage
            from collections import Counter
            sample_counter = Counter(samples_generated)
            
            # Calculate actual cycles
            min_appearances = min(sample_counter.values()) if sample_counter else 0
            max_appearances = max(sample_counter.values()) if sample_counter else 0
            
            print(f"  Total samples generated: {len(samples_generated)}")
            print(f"  Unique samples used: {len(sample_counter)}")
            print(f"  Min times a sample appeared: {min_appearances}")
            print(f"  Max times a sample appeared: {max_appearances}")
            print(f"  Expected full cycles: {int(expected_cycles)}")
            print(f"  Actual min cycles: {min_appearances}")
            
            # Verify
            assert len(samples_generated) == cache_size, f"Expected {cache_size} samples, got {len(samples_generated)}"
            assert len(cache) == cache_size, f"Cache size mismatch"
            
            # For exact multiples, all samples should appear the same number of times
            if cache_size % dataset_size == 0:
                assert min_appearances == max_appearances, f"Uneven cycling for exact multiple"
                assert min_appearances == int(expected_cycles), f"Cycle count mismatch"
                print("  ✅ Exact cycling verified")
            else:
                # For non-exact multiples, some samples appear one more time
                assert max_appearances - min_appearances <= 1, f"Cycling imbalance too large"
                print("  ✅ Partial cycling verified")


if __name__ == "__main__":
    print("Starting cache regeneration cycling tests...\n")
    
    try:
        test_cycling_detection_logging()
        test_exact_cycle_counting()
        
        print("\n" + "="*60)
        print("✅ All cycling detection tests passed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)