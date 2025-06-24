"""Comprehensive test for KV cache functionality including the epoch boundary bug fix."""

import torch
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.orig import OrigWrapper
from lens.training.loop import train_step
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc


def test_all_kv_cache_scenarios():
    """Test KV cache in all scenarios that could trigger the ln_2 error."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Comprehensive KV Cache Test Suite")
    print("=" * 70)
    
    # Create models
    dec_config = DecoderConfig(
        model_name="gpt2",
        n_prompt_tokens=0,
        base_model=False,
        projection_layer=True,
        output_head=True,
        eye_init=True,
        use_kv_cache=True
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
    orig = OrigWrapper("gpt2").to(device)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    d_model = decoder.base.config.hidden_size
    
    print("\n1. Testing basic generation modes:")
    test_activation = torch.randn(2, d_model, device=device)
    
    # Test without override
    print("   - Without override model...")
    decoder.set_prompt("The meaning of <embed> is:", tokenizer)
    gen1 = decoder.generate_soft_kv_cached(test_activation, max_length=4, gumbel_tau=1.0)
    print(f"     ✓ Success: {gen1.generated_text_embeddings.shape}")
    
    # Test with OrigWrapper override
    print("   - With OrigWrapper override...")
    gen2 = decoder.generate_soft_kv_cached(
        test_activation, max_length=4, gumbel_tau=1.0, 
        override_model_base_and_out=orig
    )
    print(f"     ✓ Success: {gen2.generated_text_embeddings.shape}")
    
    # Test with raw model override
    print("   - With raw model override...")
    raw_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    gen3 = decoder.generate_soft_kv_cached(
        test_activation, max_length=4, gumbel_tau=1.0,
        override_model_base_and_out=raw_model
    )
    print(f"     ✓ Success: {gen3.generated_text_embeddings.shape}")
    
    print("\n2. Testing training loop integration:")
    models = {"dec": decoder, "enc": encoder, "orig": orig}
    
    # Test with different batch sizes (simulating epoch boundaries)
    for batch_idx, batch_size in enumerate([2, 4, 8]):
        print(f"\n   Batch {batch_idx + 1} (size={batch_size}):")
        
        batch = {
            "A": torch.randn(batch_size, d_model, device=device),
            "A_prime": torch.randn(batch_size, d_model, device=device),
            "input_ids_A": torch.randint(0, 1000, (batch_size, 32), device=device),
            "layer_idx": torch.tensor([5] * batch_size, device=device),
            "token_pos_A": torch.tensor([10] * batch_size, device=device)
        }
        
        loss_fns = {
            "t_text": 8,
            "tau": 1.0,
            "alpha": 0.1,
            "lm_base_weight": 0.0,
            "kl_base_weight": 1.0,  # Enable KL to trigger A' generation
            "entropy_weight": 0.0,
            "mse_weight": 1.0
        }
        
        try:
            losses = train_step(batch, models, loss_fns)
            losses['total'].backward()
            print(f"     ✓ Forward/backward pass successful")
            print(f"     MSE: {losses['mse'].item():.4f}, KL: {losses['kl'].item():.4f}")
            
            # Clear gradients
            decoder.zero_grad()
            encoder.zero_grad()
            
        except Exception as e:
            print(f"     ✗ Failed: {str(e)}")
            if "ln_2" in str(e):
                print("     \!\!\! This is the ln_2 error\!")
    
    print("\n3. Testing prompt changes (epoch boundary scenario):")
    prompts = [
        "The meaning of <embed> is:",
        "<|endoftext|>Short explanation of <embed>. Language, topic:",
        "Explain <embed> in simple terms:"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\n   Prompt {i+1}: '{prompt[:30]}...'")
        decoder.set_prompt(prompt, tokenizer)
        
        try:
            gen = decoder.generate_soft_kv_cached(
                test_activation, max_length=4, gumbel_tau=1.0,
                override_model_base_and_out=orig
            )
            print(f"     ✓ Generation successful: {gen.generated_text_embeddings.shape}")
        except Exception as e:
            print(f"     ✗ Failed: {str(e)}")
    
    print("\n4. Testing memory cleanup (garbage collection scenario):")
    # Simulate what might happen at epoch boundaries
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        gen = decoder.generate_soft_kv_cached(
            test_activation, max_length=4, gumbel_tau=1.0,
            override_model_base_and_out=orig
        )
        print("   ✓ Generation after GC successful")
    except Exception as e:
        print(f"   ✗ Failed after GC: {str(e)}")
    
    print("\n" + "=" * 70)
    print("All tests completed\!")


if __name__ == "__main__":
    test_all_kv_cache_scenarios()
