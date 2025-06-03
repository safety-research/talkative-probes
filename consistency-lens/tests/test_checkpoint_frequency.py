"""Test memory savings with different checkpoint frequencies."""

import torch
import gc
from lens.models.decoder import Decoder, DecoderConfig
from transformers import AutoTokenizer


def test_checkpoint_frequencies():
    """Test memory usage with different checkpoint frequencies."""
    
    if not torch.cuda.is_available():
        print("CUDA required for memory testing")
        return
    
    device = torch.device("cuda")
    model_name = "gpt2"
    batch_size = 8
    max_length = 16
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Memory Savings with Different Checkpoint Frequencies")
    print("=" * 70)
    print(f"Model: {model_name}, Batch: {batch_size}, Seq Length: {max_length}")
    print("-" * 70)
    
    # Test different checkpoint frequencies
    frequencies = [
        (None, "No checkpointing (original)"),
        (4, "Every 4 tokens (current default)"),
        (2, "Every 2 tokens"),
        (1, "Every token (maximum savings)")
    ]
    
    baseline_memory = None
    
    for checkpoint_freq, description in frequencies:
        torch.cuda.empty_cache()
        gc.collect()
        
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
            use_checkpointing=(checkpoint_freq is not None),
            checkpoint_every_n_tokens=checkpoint_freq if checkpoint_freq else 4
        )
        
        decoder = Decoder(config).to(device)
        decoder.set_prompt("The meaning of <embed> is:", tokenizer)
        
        d_model = decoder.base.config.hidden_size
        test_activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1024 / 1024
        
        # Generate
        if checkpoint_freq is None:
            gen = decoder.generate_soft(
                test_activation,
                max_length=max_length,
                gumbel_tau=1.0
            )
        else:
            gen = decoder.generate_soft_chkpt(
                test_activation,
                max_length=max_length,
                gumbel_tau=1.0,
                checkpoint_every_n_tokens=checkpoint_freq
            )
        
        # Force computation graph
        loss = gen.generated_text_embeddings.sum()
        
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated() / 1024 / 1024
        mem_used = mem_after - mem_before
        
        if baseline_memory is None:
            baseline_memory = mem_used
            
        savings = baseline_memory - mem_used if checkpoint_freq else 0
        savings_pct = (savings / baseline_memory * 100) if checkpoint_freq else 0
        
        # Calculate theoretical savings
        if checkpoint_freq:
            if checkpoint_freq == 1:
                # Checkpoint all but first token
                checkpointed_ratio = (max_length - 1) / max_length
            else:
                # Checkpoint every N tokens
                num_checkpoints = (max_length // checkpoint_freq) - 1  # Exclude first checkpoint at 0
                checkpointed_ratio = num_checkpoints / max_length
            theoretical_savings_pct = checkpointed_ratio * 100 * 0.8  # 0.8 factor for non-checkpointable overhead
        else:
            theoretical_savings_pct = 0
            
        print(f"\n{description}:")
        print(f"  Memory used: {mem_used:.1f} MB")
        if checkpoint_freq:
            print(f"  Memory saved: {savings:.1f} MB ({savings_pct:.1f}%)")
            print(f"  Theoretical max savings: ~{theoretical_savings_pct:.1f}%")
            if checkpoint_freq == 1:
                print(f"  Checkpointing {max_length-1}/{max_length} tokens")
            else:
                num_checkpointed = (max_length // checkpoint_freq) - 1
                print(f"  Checkpointing {num_checkpointed}/{max_length} tokens")
        
        # Cleanup
        loss.backward()  # Complete the computation
        del decoder, gen, loss, test_activation
    
    print("\n" + "-" * 70)
    print("Summary:")
    print("- Checkpointing every token gives maximum memory savings")
    print("- Trade-off: More frequent checkpointing = more recomputation during backward")
    print("- Recommendation: Use checkpoint_every_n_tokens=1 for memory-constrained situations")
    print("                  Use checkpoint_every_n_tokens=2 for balanced memory/speed")


if __name__ == "__main__":
    test_checkpoint_frequencies()