"""Test KV cache with A' generation path."""

import torch
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.training.loop import train_step
from transformers import AutoTokenizer


def test_aprime_path():
    """Test that KV cache works correctly for A' generation."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing A' generation with KV cache on device: {device}")
    
    # Create models with KV cache enabled
    dec_config = DecoderConfig(
        model_name="gpt2",
        n_prompt_tokens=0,
        base_model=False,
        projection_layer=True,
        output_head=True,
        eye_init=True,
        use_checkpointing=False,
        use_kv_cache=True  # Enable KV cache
    )
    
    enc_config = EncoderConfig(
        model_name="gpt2",
        base_model=False,
        use_base_model=True,
        projection_layer=True,
        eye_init=True,
        output_layer=-1,
        stop_grad_aprime=False
    )
    
    decoder = Decoder(dec_config).to(device)
    encoder = Encoder(enc_config).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    decoder.set_prompt("The meaning of <embed> is:", tokenizer)
    
    # Create batch
    batch_size = 4
    d_model = decoder.base.config.hidden_size
    batch = {
        "A": torch.randn(batch_size, d_model, device=device),
        "A_prime": torch.randn(batch_size, d_model, device=device),
        "input_ids_A": torch.randint(0, 1000, (batch_size, 32), device=device),
        "layer_idx": torch.tensor([5] * batch_size, device=device),
        "token_pos_A": torch.tensor([10] * batch_size, device=device)
    }
    
    models = {
        "dec": decoder,
        "enc": encoder,
        "orig": None  # No orig model for this test
    }
    
    # Test with different KL weights to trigger A' path
    print("\nTesting with kl_base_weight = 0 (no A' generation):")
    loss_fns_no_kl = {
        "T_text": 8,
        "tau": 1.0,
        "alpha": 0.1,
        "lm_base_weight": 0.0,
        "kl_base_weight": 0.0,  # No KL = no A' generation
        "entropy_weight": 0.0,
        "mse_weight": 1.0
    }
    
    try:
        losses = train_step(batch.copy(), models, loss_fns_no_kl)
        print(f"✓ Success - MSE loss: {losses['mse'].item():.4f}")
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
    
    print("\nTesting with kl_base_weight > 0 (triggers A' generation):")
    loss_fns_with_kl = {
        "T_text": 8,
        "tau": 1.0,
        "alpha": 0.1,
        "lm_base_weight": 0.0,
        "kl_base_weight": 1.0,  # Enable KL = triggers A' generation
        "entropy_weight": 0.0,
        "mse_weight": 1.0
    }
    
    try:
        losses = train_step(batch.copy(), models, loss_fns_with_kl)
        print(f"✓ Success - MSE loss: {losses['mse'].item():.4f}, KL loss: {losses['kl'].item():.4f}")
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test multiple forward passes to ensure KV cache handles repeated calls
    print("\nTesting multiple forward passes:")
    for i in range(3):
        try:
            losses = train_step(batch.copy(), models, loss_fns_with_kl)
            print(f"  Pass {i+1}: ✓ MSE={losses['mse'].item():.4f}, KL={losses['kl'].item():.4f}")
        except Exception as e:
            print(f"  Pass {i+1}: ✗ Failed - {str(e)}")
    
    print("\nTesting backward pass:")
    losses = train_step(batch.copy(), models, loss_fns_with_kl)
    losses['total'].backward()
    
    # Check gradients exist
    grad_exists = decoder.proj.weight.grad is not None
    print(f"Gradients computed: {'✓' if grad_exists else '✗'}")
    
    if grad_exists:
        print(f"Projection gradient norm: {decoder.proj.weight.grad.norm().item():.4f}")


if __name__ == "__main__":
    test_aprime_path()