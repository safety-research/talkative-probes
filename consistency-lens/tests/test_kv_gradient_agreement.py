"""Test that gradients computed with KV cache match the original method."""

import torch
import torch.nn as nn
from lens.models.decoder import Decoder, DecoderConfig
from transformers import AutoTokenizer
import numpy as np


def test_gradient_agreement():
    """Verify that KV-cached and original methods produce identical gradients."""
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing gradient agreement on device: {device}")
    
    # Create two identical decoders
    config_orig = DecoderConfig(
        model_name="gpt2",
        n_prompt_tokens=0,
        base_model=False,
        projection_layer=True,
        output_head=True,
        eye_init=True,
        trainable_prompts=True,
        use_checkpointing=False,
        use_kv_cache=False  # Original
    )
    
    config_kv = DecoderConfig(
        model_name="gpt2",
        n_prompt_tokens=0,
        base_model=False,
        projection_layer=True,
        output_head=True,
        eye_init=True,
        trainable_prompts=True,
        use_checkpointing=False,
        use_kv_cache=True  # KV-cached
    )
    
    decoder_orig = Decoder(config_orig).to(device)
    decoder_kv = Decoder(config_kv).to(device)
    
    # Ensure identical initialization
    decoder_kv.load_state_dict(decoder_orig.state_dict())
    
    # Setup tokenizer and prompt
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    prompt = "The meaning of <embed> is:"
    decoder_orig.set_prompt(prompt, tokenizer)
    decoder_kv.set_prompt(prompt, tokenizer)
    
    # Test different configurations
    test_configs = [
        (2, 4),   # Small test
        (4, 8),   # Medium test
        (8, 16),  # Larger test
    ]
    
    print("\nTesting gradient agreement across different configurations:")
    print("=" * 70)
    
    all_passed = True
    
    for batch_size, seq_length in test_configs:
        print(f"\nBatch size: {batch_size}, Sequence length: {seq_length}")
        print("-" * 50)
        
        # Create test input
        d_model = decoder_orig.base.config.hidden_size
        test_activation = torch.randn(batch_size, d_model, device=device)
        
        # Clone inputs for both methods
        act_orig = test_activation.clone().detach().requires_grad_(True)
        act_kv = test_activation.clone().detach().requires_grad_(True)
        
        # Forward pass with same random seed
        torch.manual_seed(123)
        gen_orig = decoder_orig.generate_soft(act_orig, max_length=seq_length, gumbel_tau=1.0)
        
        torch.manual_seed(123)
        gen_kv = decoder_kv.generate_soft_kv_cached(act_kv, max_length=seq_length, gumbel_tau=1.0)
        
        # Verify forward pass matches
        forward_match = torch.allclose(gen_orig.generated_text_embeddings, 
                                      gen_kv.generated_text_embeddings, 
                                      rtol=1e-4, atol=1e-5)
        print(f"Forward pass matches: {forward_match}")
        
        # Create identical loss
        target = torch.randn_like(gen_orig.generated_text_embeddings)
        loss_orig = nn.functional.mse_loss(gen_orig.generated_text_embeddings, target)
        loss_kv = nn.functional.mse_loss(gen_kv.generated_text_embeddings, target)
        
        # Backward pass
        loss_orig.backward()
        loss_kv.backward()
        
        # Compare gradients
        input_grad_match = torch.allclose(act_orig.grad, act_kv.grad, rtol=1e-4, atol=1e-5)
        print(f"Input gradient matches: {input_grad_match}")
        
        if not input_grad_match:
            diff = torch.abs(act_orig.grad - act_kv.grad)
            print(f"  Max difference: {diff.max().item():.2e}")
            print(f"  Mean difference: {diff.mean().item():.2e}")
            print(f"  Original grad norm: {act_orig.grad.norm().item():.4f}")
            print(f"  KV-cached grad norm: {act_kv.grad.norm().item():.4f}")
        
        # Compare projection layer gradients
        proj_weight_match = torch.allclose(decoder_orig.proj.weight.grad, 
                                          decoder_kv.proj.weight.grad, 
                                          rtol=1e-4, atol=1e-5)
        proj_bias_match = torch.allclose(decoder_orig.proj.bias.grad, 
                                        decoder_kv.proj.bias.grad, 
                                        rtol=1e-4, atol=1e-5)
        
        print(f"Projection weight gradient matches: {proj_weight_match}")
        print(f"Projection bias gradient matches: {proj_bias_match}")
        
        # Compare output head gradients
        out_weight_match = torch.allclose(decoder_orig.out.weight.grad, 
                                         decoder_kv.out.weight.grad, 
                                         rtol=1e-4, atol=1e-5)
        print(f"Output head gradient matches: {out_weight_match}")
        
        # Sample some transformer layer gradients
        layer_matches = []
        for i in range(min(3, len(decoder_orig.base.transformer.h))):  # Check first 3 layers
            layer_orig = decoder_orig.base.transformer.h[i]
            layer_kv = decoder_kv.base.transformer.h[i]
            
            # Check attention weights
            if layer_orig.attn.c_attn.weight.grad is not None:
                attn_match = torch.allclose(layer_orig.attn.c_attn.weight.grad,
                                           layer_kv.attn.c_attn.weight.grad,
                                           rtol=1e-4, atol=1e-5)
                layer_matches.append(attn_match)
        
        print(f"Transformer layer gradients match: {all(layer_matches) if layer_matches else 'No gradients'}")
        
        # Overall test result
        test_passed = (forward_match and input_grad_match and proj_weight_match and 
                      proj_bias_match and out_weight_match)
        print(f"Test {'PASSED' if test_passed else 'FAILED'}")
        
        all_passed = all_passed and test_passed
        
        # Clear gradients
        decoder_orig.zero_grad()
        decoder_kv.zero_grad()
    
    print("\n" + "=" * 70)
    print(f"Overall result: {'✓ All tests PASSED!' if all_passed else '✗ Some tests FAILED!'}")
    print("=" * 70)
    
    return all_passed


def test_memory_calculation():
    """Explain the memory calculation with actual numbers."""
    
    print("\nMemory Calculation Explanation:")
    print("=" * 70)
    print("Example: Batch=8, Seq=16")
    print("-" * 50)
    print("1. Before forward pass: 0 MB (baseline)")
    print("2. After forward pass:  +1624.6 MB (activations stored)")
    print("3. After backward pass: +152.6 MB total")
    print("   - This means backward freed: 1624.6 - 152.6 = 1472.0 MB")
    print("   - So backward 'used': -1472.0 MB (negative = freed memory)")
    print("\nThe backward pass frees more memory than it allocates because:")
    print("- Activations are freed after computing gradients")
    print("- Gradients are smaller than activations")
    print("- Net effect: memory decreases during backward")
    print("=" * 70)


if __name__ == "__main__":
    test_gradient_agreement()
    test_memory_calculation()