"""Simple test to verify KV-cached generation works correctly."""

import torch
from lens.models.decoder import Decoder, DecoderConfig
from transformers import AutoTokenizer


def test_basic_kv_cache():
    """Basic test that KV cache method runs without errors."""
    
    # Simple setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DecoderConfig(
        model_name="gpt2",
        n_prompt_tokens=0,
        base_model=False,
        projection_layer=True,
        output_head=True,
        embedding_head=False,
        eye_init=True,
        trainable_prompts=True,
        use_checkpointing=False
    )
    
    decoder = Decoder(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    decoder.set_prompt("The meaning of <embed> is:", tokenizer)
    
    # Test input
    batch_size = 2
    d_model = decoder.base.config.hidden_size
    test_activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    
    print("Testing generate_soft_kv_cached...")
    
    # Run generation
    gen = decoder.generate_soft_kv_cached(
        test_activation,
        max_length=4,
        gumbel_tau=1.0
    )
    
    print(f"Generated shape: {gen.generated_text_embeddings.shape}")
    print(f"Logits shape: {gen.raw_lm_logits.shape}")
    print(f"Hard IDs shape: {gen.hard_token_ids.shape}")
    
    # Test backward pass
    loss = gen.generated_text_embeddings.sum()
    loss.backward()
    
    print(f"Gradient exists: {test_activation.grad is not None}")
    print("✓ Basic test passed!")


if __name__ == "__main__":
    test_basic_kv_cache()