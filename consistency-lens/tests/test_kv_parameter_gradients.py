"""Test that parameter gradients are computed correctly with KV cache."""

import torch
import torch.nn as nn
from lens.models.decoder import Decoder, DecoderConfig
from transformers import AutoTokenizer
import numpy as np


def test_parameter_gradients():
    """Test gradients w.r.t. model parameters with KV caching."""
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing parameter gradients on device: {device}")
    
    # Create two identical decoders with trainable base model
    config_orig = DecoderConfig(
        model_name="gpt2",
        n_prompt_tokens=0,
        base_model=True,  # Make base model trainable!
        projection_layer=True,
        output_head=True,
        embedding_head=True,  # Also train embeddings
        eye_init=True,
        trainable_prompts=True,
        use_checkpointing=False,
        use_kv_cache=False  # Original
    )
    
    config_kv = DecoderConfig(
        model_name="gpt2",
        n_prompt_tokens=0,
        base_model=True,  # Make base model trainable!
        projection_layer=True,
        output_head=True,
        embedding_head=True,  # Also train embeddings
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
    
    # Test configuration
    batch_size = 4
    seq_length = 8
    d_model = decoder_orig.base.config.hidden_size
    
    print(f"\nTesting with batch_size={batch_size}, seq_length={seq_length}")
    print("=" * 70)
    
    # Create test input
    test_activation = torch.randn(batch_size, d_model, device=device)
    
    # Clone inputs for both methods
    act_orig = test_activation.clone().detach().requires_grad_(True)
    act_kv = test_activation.clone().detach().requires_grad_(True)
    
    # Forward pass with same random seed
    torch.manual_seed(123)
    gen_orig = decoder_orig.generate_soft(act_orig, max_length=seq_length, gumbel_tau=1.0)
    
    torch.manual_seed(123)
    gen_kv = decoder_kv.generate_soft_kv_cached(act_kv, max_length=seq_length, gumbel_tau=1.0)
    
    # Create identical loss
    target = torch.randn_like(gen_orig.generated_text_embeddings)
    loss_orig = nn.functional.mse_loss(gen_orig.generated_text_embeddings, target)
    loss_kv = nn.functional.mse_loss(gen_kv.generated_text_embeddings, target)
    
    # Backward pass
    loss_orig.backward()
    loss_kv.backward()
    
    # Check various parameter gradients
    print("Checking parameter gradients:")
    print("-" * 50)
    
    # 1. Projection layer
    proj_weight_diff = torch.abs(decoder_orig.proj.weight.grad - decoder_kv.proj.weight.grad).max().item()
    proj_bias_diff = torch.abs(decoder_orig.proj.bias.grad - decoder_kv.proj.bias.grad).max().item()
    print(f"Projection weight max diff: {proj_weight_diff:.2e}")
    print(f"Projection bias max diff: {proj_bias_diff:.2e}")
    
    # 2. Output head
    out_weight_diff = torch.abs(decoder_orig.out.weight.grad - decoder_kv.out.weight.grad).max().item()
    print(f"Output head weight max diff: {out_weight_diff:.2e}")
    
    # 3. Embedding layers
    input_emb_orig = decoder_orig.base.get_input_embeddings()
    input_emb_kv = decoder_kv.base.get_input_embeddings()
    if input_emb_orig.weight.grad is not None:
        emb_diff = torch.abs(input_emb_orig.weight.grad - input_emb_kv.weight.grad).max().item()
        print(f"Input embeddings max diff: {emb_diff:.2e}")
        print(f"  Original grad norm: {input_emb_orig.weight.grad.norm().item():.4f}")
        print(f"  KV-cached grad norm: {input_emb_kv.weight.grad.norm().item():.4f}")
    
    # 4. Transformer layers
    print("\nTransformer layer gradients:")
    layer_diffs = []
    
    for i, (layer_orig, layer_kv) in enumerate(zip(decoder_orig.base.transformer.h, 
                                                   decoder_kv.base.transformer.h)):
        # Check attention weights
        if layer_orig.attn.c_attn.weight.grad is not None:
            attn_diff = torch.abs(layer_orig.attn.c_attn.weight.grad - 
                                 layer_kv.attn.c_attn.weight.grad).max().item()
            layer_diffs.append(('attn', i, attn_diff))
            
            # Also check gradient norms
            orig_norm = layer_orig.attn.c_attn.weight.grad.norm().item()
            kv_norm = layer_kv.attn.c_attn.weight.grad.norm().item()
            
            if i < 3:  # Print first 3 layers
                print(f"  Layer {i} attention: max diff={attn_diff:.2e}, "
                      f"orig_norm={orig_norm:.4f}, kv_norm={kv_norm:.4f}")
        
        # Check MLP weights
        if hasattr(layer_orig.mlp, 'c_fc') and layer_orig.mlp.c_fc.weight.grad is not None:
            mlp_diff = torch.abs(layer_orig.mlp.c_fc.weight.grad - 
                                layer_kv.mlp.c_fc.weight.grad).max().item()
            layer_diffs.append(('mlp', i, mlp_diff))
    
    # Summary statistics
    if layer_diffs:
        max_diff = max(d[2] for d in layer_diffs)
        avg_diff = sum(d[2] for d in layer_diffs) / len(layer_diffs)
        print(f"\nTransformer layers summary:")
        print(f"  Max gradient difference: {max_diff:.2e}")
        print(f"  Avg gradient difference: {avg_diff:.2e}")
        print(f"  Layers with gradients: {len(layer_diffs)}")
    
    # Test with different tolerance levels
    tolerances = [1e-3, 1e-4, 1e-5, 1e-6]
    print("\nGradient agreement at different tolerances:")
    print("-" * 50)
    
    for tol in tolerances:
        matches = []
        
        # Check all parameter groups
        if decoder_orig.proj.weight.grad is not None:
            matches.append(torch.allclose(decoder_orig.proj.weight.grad, 
                                        decoder_kv.proj.weight.grad, rtol=tol, atol=tol))
        
        if decoder_orig.out.weight.grad is not None:
            matches.append(torch.allclose(decoder_orig.out.weight.grad, 
                                        decoder_kv.out.weight.grad, rtol=tol, atol=tol))
        
        # Check a few transformer layers
        for i in range(min(3, len(decoder_orig.base.transformer.h))):
            layer_orig = decoder_orig.base.transformer.h[i]
            layer_kv = decoder_kv.base.transformer.h[i]
            
            if layer_orig.attn.c_attn.weight.grad is not None:
                matches.append(torch.allclose(layer_orig.attn.c_attn.weight.grad,
                                            layer_kv.attn.c_attn.weight.grad, 
                                            rtol=tol, atol=tol))
        
        match_rate = sum(matches) / len(matches) if matches else 0
        print(f"  Tolerance {tol:.0e}: {match_rate*100:.1f}% of gradients match")
    
    print("\n" + "=" * 70)
    
    # Final verdict
    all_close = all_close = proj_weight_diff < 1e-4 and out_weight_diff < 1e-4
    if layer_diffs:
        all_close = all_close and max(d[2] for d in layer_diffs) < 1e-4
    
    print(f"Result: {'✓ Parameter gradients match!' if all_close else '✗ Parameter gradients differ!'}")
    print("=" * 70)
    
    return all_close


def test_full_training_step():
    """Test a full training step with all components."""
    
    from lens.models.encoder import Encoder, EncoderConfig
    from lens.training.loop import train_step
    
    print("\n\nTesting full training step with parameter gradients:")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create models with trainable parameters
    dec_config_orig = DecoderConfig(
        model_name="gpt2",
        base_model=True,  # Train everything
        projection_layer=True,
        output_head=True,
        embedding_head=True,
        use_kv_cache=False
    )
    
    dec_config_kv = DecoderConfig(
        model_name="gpt2",
        base_model=True,  # Train everything
        projection_layer=True,
        output_head=True,
        embedding_head=True,
        use_kv_cache=True
    )
    
    enc_config = EncoderConfig(
        model_name="gpt2",
        base_model=True,  # Train encoder too
        use_base_model=True,
        projection_layer=True,
        embedding_head=True,
        output_layer=-1
    )
    
    # Create models
    decoder_orig = Decoder(dec_config_orig).to(device)
    decoder_kv = Decoder(dec_config_kv).to(device)
    encoder_orig = Encoder(enc_config).to(device)
    encoder_kv = Encoder(enc_config).to(device)
    
    # Sync parameters
    decoder_kv.load_state_dict(decoder_orig.state_dict())
    encoder_kv.load_state_dict(encoder_orig.state_dict())
    
    # Setup
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    decoder_orig.set_prompt("The meaning of <embed> is:", tokenizer)
    decoder_kv.set_prompt("The meaning of <embed> is:", tokenizer)
    
    # Create batch
    batch_size = 4
    d_model = 768
    batch = {
        "A": torch.randn(batch_size, d_model, device=device),
        "A_prime": torch.randn(batch_size, d_model, device=device),
        "input_ids_A": torch.randint(0, 1000, (batch_size, 32), device=device),
        "layer_idx": torch.tensor([5] * batch_size, device=device),
        "token_pos_A": torch.tensor([10] * batch_size, device=device)
    }
    
    models_orig = {"dec": decoder_orig, "enc": encoder_orig, "orig": None}
    models_kv = {"dec": decoder_kv, "enc": encoder_kv, "orig": None}
    
    loss_fns = {
        "t_text": 8,
        "tau": 1.0,
        "alpha": 0.1,
        "lm_base_weight": 0.0,
        "kl_base_weight": 0.0,
        "entropy_weight": 0.01,  # Small entropy weight
        "mse_weight": 1.0
    }
    
    # Run training steps
    torch.manual_seed(456)
    losses_orig = train_step(batch.copy(), models_orig, loss_fns)
    
    torch.manual_seed(456)
    losses_kv = train_step(batch.copy(), models_kv, loss_fns)
    
    # Backward
    losses_orig['total'].backward()
    losses_kv['total'].backward()
    
    # Compare some key gradients
    print("Gradient comparison after full training step:")
    print("-" * 50)
    
    # Decoder projection
    proj_match = torch.allclose(decoder_orig.proj.weight.grad, 
                               decoder_kv.proj.weight.grad, rtol=1e-4, atol=1e-5)
    print(f"Decoder projection gradients match: {proj_match}")
    
    # Encoder projection  
    enc_proj_match = torch.allclose(encoder_orig.proj.weight.grad,
                                   encoder_kv.proj.weight.grad, rtol=1e-4, atol=1e-5)
    print(f"Encoder projection gradients match: {enc_proj_match}")
    
    # Sample transformer layer
    if decoder_orig.base.transformer.h[0].attn.c_attn.weight.grad is not None:
        layer0_match = torch.allclose(decoder_orig.base.transformer.h[0].attn.c_attn.weight.grad,
                                     decoder_kv.base.transformer.h[0].attn.c_attn.weight.grad,
                                     rtol=1e-4, atol=1e-5)
        print(f"Decoder transformer layer 0 gradients match: {layer0_match}")
    
    print("=" * 70)
    print(f"Full training step: {'✓ PASSED' if proj_match and enc_proj_match else '✗ FAILED'}")
    print("=" * 70)


if __name__ == "__main__":
    test_parameter_gradients()
    test_full_training_step()