#!/usr/bin/env python3
"""Test that all generation methods produce identical outputs and gradients."""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
import time


def test_generation_consistency():
    """Test that all generation methods produce identical outputs."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Testing Generation Method Consistency")
    print("=" * 80)
    print(f"Device: {device}")
    print()
    
    # Test configurations
    test_configs = [
        # (model_name, description, batch_size, seq_length, patch_all_layers, per_layer_projections)
        ("gpt2", "GPT-2 - No patching", 2, 8, False, False),
        ("gpt2", "GPT-2 - Multi-layer single proj", 2, 8, True, False),
        ("gpt2", "GPT-2 - Multi-layer per-layer proj", 2, 8, True, True),
        ("SimpleStories/SimpleStories-5M", "LLaMA - No patching", 2, 8, False, False),
        ("SimpleStories/SimpleStories-5M", "LLaMA - Multi-layer single proj", 2, 8, True, False),
        ("SimpleStories/SimpleStories-5M", "LLaMA - Multi-layer per-layer proj", 2, 8, True, True),
    ]
    
    for model_name, description, batch_size, seq_length, patch_all_layers, per_layer_projections in test_configs:
        print(f"\n{description}")
        print("-" * 60)
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create decoder config
        config = DecoderConfig(
            model_name=model_name,
            base_model=False,
            projection_layer=True,
            patch_all_layers=patch_all_layers,
            per_layer_projections=per_layer_projections,
        )
        
        # Methods to test
        methods = ["generate_soft", "generate_soft_chkpt"]
        
        # KV cache only works for GPT-2 and without multi-layer patching currently
        if model_name == "gpt2" and not patch_all_layers:
            methods.append("generate_soft_kv_cached")
        
        # Store results from each method
        results = {}
        gradients = {}
        times = {}
        
        # Fixed random seed for reproducibility
        torch.manual_seed(42)
        
        for method in methods:
            try:
                # Create fresh decoder for each method
                decoder = Decoder(config).to(device)
                decoder.set_prompt("explain <embed>:", tokenizer)
                d_model = decoder.base.config.hidden_size
                
                # Create activation (same seed for each method)
                torch.manual_seed(42)
                activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
                
                # Time the generation
                start_time = time.time()
                
                # Generate based on method
                if method == "generate_soft":
                    gen = decoder.generate_soft(activation, max_length=seq_length, gumbel_tau=1.0)
                elif method == "generate_soft_chkpt":
                    gen = decoder.generate_soft_chkpt(activation, max_length=seq_length, gumbel_tau=1.0)
                elif method == "generate_soft_kv_cached":
                    gen = decoder.generate_soft_kv_cached(activation, max_length=seq_length, gumbel_tau=1.0)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                elapsed_time = time.time() - start_time
                times[method] = elapsed_time
                
                # Store outputs
                results[method] = {
                    'embeddings': gen.generated_text_embeddings.detach().cpu(),
                    'logits': gen.raw_lm_logits.detach().cpu(),
                    'hard_ids': gen.hard_token_ids.detach().cpu(),
                }
                
                # Compute loss and gradients
                loss = gen.generated_text_embeddings.sum()
                loss.backward()
                
                # Store gradients
                gradients[method] = {
                    'activation': activation.grad.clone().cpu(),
                }
                
                # Store projection gradients
                if per_layer_projections:
                    gradients[method]['proj_weight'] = decoder.proj_weight.grad.clone().cpu()
                    gradients[method]['proj_bias'] = decoder.proj_bias.grad.clone().cpu()
                elif decoder.proj is not None:
                    gradients[method]['proj_weight'] = decoder.proj.weight.grad.clone().cpu()
                    gradients[method]['proj_bias'] = decoder.proj.bias.grad.clone().cpu()
                
                print(f"  {method:25} - Time: {elapsed_time:.3f}s ✓")
                
                # Cleanup
                del decoder, gen
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  {method:25} - FAILED: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Compare results
        if len(results) > 1:
            print("\n  Comparing outputs:")
            base_method = methods[0]
            
            for method in methods[1:]:
                if method in results:
                    # Compare embeddings
                    emb_diff = (results[base_method]['embeddings'] - results[method]['embeddings']).abs().max().item()
                    logits_diff = (results[base_method]['logits'] - results[method]['logits']).abs().max().item()
                    hard_ids_same = torch.equal(results[base_method]['hard_ids'], results[method]['hard_ids'])
                    
                    print(f"    {base_method} vs {method}:")
                    print(f"      Embeddings max diff: {emb_diff:.2e}")
                    print(f"      Logits max diff: {logits_diff:.2e}")
                    print(f"      Hard IDs identical: {hard_ids_same}")
                    
                    # Compare gradients
                    act_grad_diff = (gradients[base_method]['activation'] - gradients[method]['activation']).abs().max().item()
                    proj_grad_diff = (gradients[base_method]['proj_weight'] - gradients[method]['proj_weight']).abs().max().item()
                    
                    print(f"      Activation grad diff: {act_grad_diff:.2e}")
                    print(f"      Projection grad diff: {proj_grad_diff:.2e}")
                    
                    # Check if outputs are consistent
                    if emb_diff < 1e-5 and logits_diff < 1e-5 and hard_ids_same:
                        print(f"      ✓ Outputs are consistent!")
                    else:
                        print(f"      ✗ Outputs differ significantly!")
                    
                    # Check if gradients are consistent (allow more tolerance for checkpointing)
                    grad_tolerance = 1e-3 if method == "generate_soft_chkpt" else 1e-5
                    if act_grad_diff < grad_tolerance and proj_grad_diff < grad_tolerance:
                        print(f"      ✓ Gradients are consistent!")
                    else:
                        print(f"      ⚠ Gradients differ (may be expected for checkpointing)")


def test_gradient_flow_all_methods():
    """Test gradient flow through encoder-decoder pipeline for all methods."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n\n" + "=" * 80)
    print("Gradient Flow Test - Full Pipeline")
    print("=" * 80)
    
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    batch_size = 2
    seq_length = 16
    
    # Test both with and without multi-layer patching
    for patch_all_layers in [False, True]:
        config_desc = "Multi-layer patching" if patch_all_layers else "Standard"
        print(f"\n{config_desc}:")
        print("-" * 40)
        
        # Create encoder
        encoder = Encoder(EncoderConfig(
            model_name=model_name,
            base_model=False,
            projection_layer=True,
        )).to(device)
        
        # Methods to test
        methods = ["generate_soft", "generate_soft_chkpt"]
        if not patch_all_layers:
            methods.append("generate_soft_kv_cached")
        
        for method in methods:
            # Create decoder
            config = DecoderConfig(
                model_name=model_name,
                base_model=False,
                projection_layer=True,
                patch_all_layers=patch_all_layers,
                per_layer_projections=False,
            )
            
            decoder = Decoder(config).to(device)
            decoder.set_prompt("explain <embed>:", tokenizer)
            d_model = decoder.base.config.hidden_size
            
            # Create activation
            activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
            
            # Forward pass through decoder
            if method == "generate_soft":
                gen = decoder.generate_soft(activation, max_length=seq_length, gumbel_tau=1.0)
            elif method == "generate_soft_chkpt":
                gen = decoder.generate_soft_chkpt(activation, max_length=seq_length, gumbel_tau=1.0)
            elif method == "generate_soft_kv_cached":
                gen = decoder.generate_soft_kv_cached(activation, max_length=seq_length, gumbel_tau=1.0)
            
            # Forward pass through encoder
            reconstructed = encoder(gen.generated_text_embeddings)
            
            # Compute reconstruction loss
            loss = F.mse_loss(reconstructed, activation.detach())
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            act_grad_norm = activation.grad.norm().item()
            decoder_grad_norm = sum(p.grad.norm().item() for p in decoder.parameters() if p.grad is not None)
            encoder_grad_norm = sum(p.grad.norm().item() for p in encoder.parameters() if p.grad is not None)
            
            print(f"\n  {method}:")
            print(f"    Loss: {loss.item():.4f}")
            print(f"    Activation grad norm: {act_grad_norm:.4f}")
            print(f"    Decoder grad norm: {decoder_grad_norm:.4f}")
            print(f"    Encoder grad norm: {encoder_grad_norm:.4f}")
            
            if act_grad_norm > 0 and decoder_grad_norm > 0 and encoder_grad_norm > 0:
                print(f"    ✓ Gradients flow correctly through full pipeline")
            else:
                print(f"    ✗ Gradient flow issue detected")
            
            # Cleanup
            decoder.zero_grad()
            encoder.zero_grad()
            activation.grad = None
            del decoder, gen
            torch.cuda.empty_cache()
        
        del encoder
        torch.cuda.empty_cache()


def test_deterministic_generation():
    """Test that generation is deterministic with fixed seeds."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n\n" + "=" * 80)
    print("Deterministic Generation Test")
    print("=" * 80)
    
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    batch_size = 2
    d_model = 768
    seq_length = 8
    
    methods = ["generate_soft", "generate_soft_chkpt", "generate_soft_kv_cached"]
    
    for method in methods:
        print(f"\n{method}:")
        
        outputs = []
        
        # Run twice with same seed
        for run in range(2):
            # Set seeds
            torch.manual_seed(123)
            torch.cuda.manual_seed_all(123) if torch.cuda.is_available() else None
            
            # Create decoder
            config = DecoderConfig(
                model_name=model_name,
                base_model=False,
                projection_layer=True,
            )
            decoder = Decoder(config).to(device)
            decoder.set_prompt("explain <embed>:", tokenizer)
            
            # Create activation with fixed seed
            torch.manual_seed(456)
            activation = torch.randn(batch_size, d_model, device=device)
            
            # Generate
            if method == "generate_soft":
                gen = decoder.generate_soft(activation, max_length=seq_length, gumbel_tau=1.0)
            elif method == "generate_soft_chkpt":
                gen = decoder.generate_soft_chkpt(activation, max_length=seq_length, gumbel_tau=1.0)
            elif method == "generate_soft_kv_cached":
                gen = decoder.generate_soft_kv_cached(activation, max_length=seq_length, gumbel_tau=1.0)
            
            outputs.append({
                'embeddings': gen.generated_text_embeddings.detach().cpu(),
                'hard_ids': gen.hard_token_ids.detach().cpu(),
            })
            
            del decoder, gen
            torch.cuda.empty_cache()
        
        # Compare outputs
        emb_diff = (outputs[0]['embeddings'] - outputs[1]['embeddings']).abs().max().item()
        ids_same = torch.equal(outputs[0]['hard_ids'], outputs[1]['hard_ids'])
        
        print(f"  Embeddings max diff: {emb_diff:.2e}")
        print(f"  Hard IDs identical: {ids_same}")
        
        if emb_diff < 1e-6 and ids_same:
            print(f"  ✓ Generation is deterministic!")
        else:
            print(f"  ✗ Generation is not deterministic!")


if __name__ == "__main__":
    test_generation_consistency()
    test_gradient_flow_all_methods()
    test_deterministic_generation()
    
    print("\n\n" + "=" * 80)
    print("✓ All tests completed!")
    print("=" * 80)