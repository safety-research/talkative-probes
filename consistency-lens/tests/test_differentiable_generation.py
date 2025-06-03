"""Test that generate_soft and generate_soft_chkpt produce equivalent results."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
import numpy as np
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.training.loop import train_step


def test_generation_equivalence():
    """Test that both generation methods produce identical forward and backward passes."""
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Use a small model for testing
    model_name = "sshleifer/tiny-gpt2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create two decoders with identical initialization
    decoder_config_original = DecoderConfig(
        model_name=model_name,
        n_prompt_tokens=0,
        base_model=False,
        projection_layer=True,
        output_head=True,
        embedding_head=False,
        eye_init=True,
        trainable_prompts=True,
        use_checkpointing=False
    )
    
    decoder_config_diff = DecoderConfig(
        model_name=model_name,
        n_prompt_tokens=0,
        base_model=False,
        projection_layer=True,
        output_head=True,
        embedding_head=False,
        eye_init=True,
        trainable_prompts=True,
        use_checkpointing=True
    )
    
    # Initialize decoders
    decoder_original = Decoder(decoder_config_original).to(device)
    decoder_diff = Decoder(decoder_config_diff).to(device)
    
    # Copy weights to ensure identical initialization
    decoder_diff.load_state_dict(decoder_original.state_dict())
    
    # Create encoder (same for both)
    encoder_config = EncoderConfig(
        model_name=model_name,
        base_model=False,
        use_base_model=True,
        projection_layer=True,
        embedding_head=False,
        eye_init=True,
        soft_prompt_length=0,
        trainable_soft_prompt=True,
        soft_prompt_init_std=0.1,
        soft_prompt_init_text=None,
        output_layer=-1,
        stop_grad_aprime=False
    )
    
    encoder_original = Encoder(encoder_config).to(device)
    encoder_diff = Encoder(encoder_config).to(device)
    encoder_diff.load_state_dict(encoder_original.state_dict())
    
    # Initialize tokenizer and set prompts
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    prompt = "The meaning of <embed> is:"
    decoder_original.set_prompt(prompt, tokenizer)
    decoder_diff.set_prompt(prompt, tokenizer)
    
    # Create test activation
    batch_size = 2
    d_model = decoder_original.base.config.hidden_size
    test_activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    
    # Test parameters
    max_length = 8
    gumbel_tau = 1.0
    
    print("Testing generation methods...")
    
    # Forward pass - original method
    print("\n1. Testing forward pass equivalence...")
    torch.manual_seed(123)  # Reset seed for Gumbel noise
    gen_original = decoder_original.generate_soft(
        test_activation.clone(),
        max_length=max_length,
        gumbel_tau=gumbel_tau
    )
    
    # Forward pass - differentiable method
    torch.manual_seed(123)  # Same seed for identical Gumbel noise
    gen_diff = decoder_diff.generate_soft_chkpt(
        test_activation.clone(),
        max_length=max_length,
        gumbel_tau=gumbel_tau,
        checkpoint_every_n_tokens=4
    )
    
    # Check forward pass equivalence
    print(f"Generated embeddings match: {torch.allclose(gen_original.generated_text_embeddings, gen_diff.generated_text_embeddings, rtol=1e-5, atol=1e-6)}")
    print(f"Raw logits match: {torch.allclose(gen_original.raw_lm_logits, gen_diff.raw_lm_logits, rtol=1e-5, atol=1e-6)}")
    print(f"Hard token IDs match: {torch.equal(gen_original.hard_token_ids, gen_diff.hard_token_ids)}")
    
    # Test backward pass
    print("\n2. Testing backward pass equivalence...")
    
    # Create identical loss targets
    target = torch.randn_like(gen_original.generated_text_embeddings)
    
    # Backward pass - original
    loss_original = nn.functional.mse_loss(gen_original.generated_text_embeddings, target)
    grad_original = torch.autograd.grad(loss_original, test_activation, create_graph=True)[0]
    
    # Backward pass - differentiable
    loss_diff = nn.functional.mse_loss(gen_diff.generated_text_embeddings, target)
    grad_diff = torch.autograd.grad(loss_diff, test_activation, create_graph=True)[0]
    
    print(f"Losses match: {torch.allclose(loss_original, loss_diff, rtol=1e-5, atol=1e-6)}")
    print(f"Gradients match: {torch.allclose(grad_original, grad_diff, rtol=1e-5, atol=1e-6)}")
    
    # Test with full training step
    print("\n3. Testing full training step...")
    
    # Create models dict
    models_original = {
        "dec": decoder_original,
        "enc": encoder_original,
        "orig": None  # Skip for simplicity
    }
    
    models_diff = {
        "dec": decoder_diff,
        "enc": encoder_diff,
        "orig": None
    }
    
    # Create batch
    batch = {
        "A": test_activation.clone().detach().requires_grad_(True),
        "A_prime": test_activation.clone().detach().requires_grad_(True),
        "input_ids_A": torch.randint(0, 1000, (batch_size, 32), device=device),
        "layer_idx": torch.tensor([5] * batch_size, device=device),
        "token_pos_A": torch.tensor([10] * batch_size, device=device)
    }
    
    # Set loss functions
    loss_fns = {
        "T_text": max_length,
        "tau": gumbel_tau,
        "alpha": 0.1,
        "lm_base_weight": 1.0,
        "kl_base_weight": 0.0,  # Disable KL for this test
        "entropy_weight": 0.0,
        "mse_weight": 1.0
    }
    
    # Run training step with original method
    torch.manual_seed(456)
    losses_original = train_step(batch.copy(), models_original, loss_fns)
    
    # Run training step with differentiable method
    torch.manual_seed(456)
    losses_diff = train_step(batch.copy(), models_diff, loss_fns)
    
    print(f"Total losses match: {torch.allclose(losses_original['total'], losses_diff['total'], rtol=1e-4, atol=1e-5)}")
    print(f"MSE losses match: {torch.allclose(losses_original['mse'], losses_diff['mse'], rtol=1e-4, atol=1e-5)}")
    
    # Check gradient flow
    print("\n4. Testing gradient flow...")
    
    # Compute gradients
    losses_original['total'].backward()
    grad_proj_original = decoder_original.proj.weight.grad.clone()
    
    losses_diff['total'].backward()
    grad_proj_diff = decoder_diff.proj.weight.grad.clone()
    
    print(f"Projection gradients match: {torch.allclose(grad_proj_original, grad_proj_diff, rtol=1e-4, atol=1e-5)}")
    
    # Test memory usage difference
    print("\n5. Memory usage comparison...")
    print("(Note: Gradient checkpointing trades memory for compute)")
    print("The differentiable version should use less memory but take slightly longer")
    
    print("\n✓ All tests completed!")
    
    # Print any differences
    if not torch.allclose(gen_original.generated_text_embeddings, gen_diff.generated_text_embeddings, rtol=1e-5, atol=1e-6):
        diff = torch.abs(gen_original.generated_text_embeddings - gen_diff.generated_text_embeddings)
        print(f"\nMax embedding difference: {diff.max().item()}")
        print(f"Mean embedding difference: {diff.mean().item()}")


if __name__ == "__main__":
    test_generation_equivalence()