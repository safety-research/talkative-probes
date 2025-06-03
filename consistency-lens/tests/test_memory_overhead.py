"""Estimate memory overhead saved by gradient checkpointing in generation."""

import torch
import gc
from lens.models.decoder import Decoder, DecoderConfig
from transformers import AutoTokenizer
import numpy as np


def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def measure_generation_memory(use_differentiable=False, model_name="gpt2", batch_size=8, max_length=32):
    """Measure memory usage for generation with and without checkpointing."""
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create decoder
    config = DecoderConfig(
        model_name=model_name,
        n_prompt_tokens=0,
        base_model=False,
        projection_layer=True,
        output_head=True,
        embedding_head=False,
        eye_init=True,
        trainable_prompts=True,
        use_checkpointing=use_differentiable
    )
    
    decoder = Decoder(config).to(device)
    
    # Set up tokenizer and prompt
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    decoder.set_prompt("The meaning of <embed> is:", tokenizer)
    
    # Create test activation
    d_model = decoder.base.config.hidden_size
    test_activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    
    # Measure memory before generation
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    mem_before = get_gpu_memory()
    
    # Generate
    if use_differentiable:
        gen = decoder.generate_soft_chkpt(
            test_activation,
            max_length=max_length,
            gumbel_tau=1.0,
            checkpoint_every_n_tokens=4
        )
    else:
        gen = decoder.generate_soft(
            test_activation,
            max_length=max_length,
            gumbel_tau=1.0
        )
    
    # Create a loss to ensure computation graph is built
    loss = gen.generated_text_embeddings.sum()
    
    # Measure memory after forward pass (before backward)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    mem_after_forward = get_gpu_memory()
    
    # Backward pass
    loss.backward()
    
    # Measure memory after backward pass
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    mem_after_backward = get_gpu_memory()
    
    # Cleanup
    del decoder, gen, loss, test_activation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return {
        'before': mem_before,
        'after_forward': mem_after_forward,
        'after_backward': mem_after_backward,
        'forward_overhead': mem_after_forward - mem_before,
        'backward_overhead': mem_after_backward - mem_after_forward,
        'total_overhead': mem_after_backward - mem_before
    }


def analyze_memory_scaling():
    """Analyze how memory scales with sequence length."""
    
    if not torch.cuda.is_available():
        print("CUDA not available. Memory analysis requires GPU.")
        return
    
    print("Memory Overhead Analysis for Differentiable Generation")
    print("=" * 70)
    
    # Test different configurations
    configs = [
        ("gpt2", 4, 16),
        ("gpt2", 8, 16),
        ("gpt2", 8, 32),
        ("gpt2", 16, 32),
        ("gpt2", 32, 32),
    ]
    
    for model_name, batch_size, max_length in configs:
        print(f"\nModel: {model_name}, Batch: {batch_size}, Seq Length: {max_length}")
        print("-" * 50)
        
        # Measure without checkpointing
        mem_original = measure_generation_memory(
            use_differentiable=False,
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length
        )
        
        # Measure with checkpointing
        mem_checkpoint = measure_generation_memory(
            use_differentiable=True,
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length
        )
        
        # Calculate savings
        forward_saved = mem_original['forward_overhead'] - mem_checkpoint['forward_overhead']
        backward_saved = mem_original['backward_overhead'] - mem_checkpoint['backward_overhead']
        total_saved = mem_original['total_overhead'] - mem_checkpoint['total_overhead']
        
        # Print results
        print(f"Original method:")
        print(f"  Forward:  {mem_original['forward_overhead']:.1f} MB")
        print(f"  Backward: {mem_original['backward_overhead']:.1f} MB")
        print(f"  Total:    {mem_original['total_overhead']:.1f} MB")
        
        print(f"Checkpointed method:")
        print(f"  Forward:  {mem_checkpoint['forward_overhead']:.1f} MB")
        print(f"  Backward: {mem_checkpoint['backward_overhead']:.1f} MB")
        print(f"  Total:    {mem_checkpoint['total_overhead']:.1f} MB")
        
        print(f"Memory saved:")
        print(f"  Forward:  {forward_saved:.1f} MB ({forward_saved/mem_original['forward_overhead']*100:.1f}%)")
        print(f"  Backward: {backward_saved:.1f} MB ({backward_saved/mem_original['backward_overhead']*100:.1f}%)")
        print(f"  Total:    {total_saved:.1f} MB ({total_saved/mem_original['total_overhead']*100:.1f}%)")


def theoretical_analysis():
    """Theoretical analysis of memory savings."""
    
    print("\n" + "=" * 70)
    print("Theoretical Memory Analysis")
    print("=" * 70)
    
    # GPT-2 parameters
    d_model = 768
    vocab_size = 50257
    bytes_per_param = 2  # bfloat16
    
    print(f"\nGPT-2 Model Parameters:")
    print(f"- Hidden dimension: {d_model}")
    print(f"- Vocabulary size: {vocab_size}")
    print(f"- Bytes per parameter: {bytes_per_param} (bfloat16)")
    
    # Calculate memory per token
    # Each token generation requires storing:
    # 1. Model forward pass activations (12 layers × hidden states)
    # 2. Attention key-value pairs
    # 3. Logits and token distributions
    
    # Simplified estimation
    activations_per_token = d_model * 12 * 2  # hidden states for 12 layers (forward + backward)
    attention_per_token = d_model * 12 * 4   # K, V for each layer × 2 for gradients
    logits_per_token = vocab_size * 2        # logits + gradients
    
    total_per_token = (activations_per_token + attention_per_token + logits_per_token) * bytes_per_param
    total_per_token_mb = total_per_token / 1024 / 1024
    
    print(f"\nMemory per token (without checkpointing):")
    print(f"- Activations: {activations_per_token * bytes_per_param / 1024 / 1024:.2f} MB")
    print(f"- Attention KV: {attention_per_token * bytes_per_param / 1024 / 1024:.2f} MB")
    print(f"- Logits: {logits_per_token * bytes_per_param / 1024 / 1024:.2f} MB")
    print(f"- Total per token: {total_per_token_mb:.2f} MB")
    
    # With checkpointing every 4 tokens
    checkpoint_interval = 4
    saved_fraction = (checkpoint_interval - 1) / checkpoint_interval
    
    print(f"\nWith checkpointing every {checkpoint_interval} tokens:")
    print(f"- Only store activations for 1/{checkpoint_interval} of tokens")
    print(f"- Memory saved: ~{saved_fraction * 100:.0f}% of activation memory")
    
    # Calculate for different sequence lengths
    print(f"\nEstimated memory usage for different sequence lengths:")
    print(f"{'Seq Length':<12} {'Without CP (MB)':<15} {'With CP (MB)':<15} {'Saved (MB)':<12} {'Saved %':<10}")
    print("-" * 65)
    
    for seq_len in [8, 16, 32, 64, 128]:
        without_cp = seq_len * total_per_token_mb
        # With checkpointing, we only store every 4th token's activations
        with_cp = (seq_len / checkpoint_interval) * total_per_token_mb + (seq_len * logits_per_token * bytes_per_param / 1024 / 1024)
        saved = without_cp - with_cp
        saved_pct = (saved / without_cp) * 100
        
        print(f"{seq_len:<12} {without_cp:<15.1f} {with_cp:<15.1f} {saved:<12.1f} {saved_pct:<10.1f}")
    
    # Batch size scaling
    print(f"\nFor batch size 8, sequence length 32:")
    batch_size = 8
    seq_len = 32
    total_without = batch_size * seq_len * total_per_token_mb
    total_with = batch_size * ((seq_len / checkpoint_interval) * total_per_token_mb + (seq_len * logits_per_token * bytes_per_param / 1024 / 1024))
    print(f"- Without checkpointing: {total_without:.1f} MB")
    print(f"- With checkpointing: {total_with:.1f} MB")
    print(f"- Memory saved: {total_without - total_with:.1f} MB ({(total_without - total_with)/total_without*100:.1f}%)")


if __name__ == "__main__":
    theoretical_analysis()
    
    if torch.cuda.is_available():
        print("\n" + "=" * 70)
        print("Empirical Memory Analysis")
        print("=" * 70)
        analyze_memory_scaling()
    else:
        print("\nNote: Empirical memory analysis requires CUDA. Running on CPU will not show actual memory savings.")