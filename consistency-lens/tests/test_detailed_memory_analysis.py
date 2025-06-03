"""Detailed memory analysis to understand why savings are minimal."""

import torch
import gc
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from transformers import AutoTokenizer
import numpy as np


def profile_generation_memory():
    """Profile memory usage at each step of generation."""
    
    if not torch.cuda.is_available():
        print("CUDA required for memory profiling")
        return
    
    torch.cuda.empty_cache()
    gc.collect()
    
    device = torch.device("cuda")
    model_name = "gpt2"
    batch_size = 8
    max_length = 16
    
    # Create models
    decoder_config = DecoderConfig(
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
    
    decoder = Decoder(decoder_config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    decoder.set_prompt("The meaning of <embed> is:", tokenizer)
    
    d_model = decoder.base.config.hidden_size
    
    print("Memory Profile for Generation")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {max_length}")
    print(f"Hidden dim: {d_model}")
    print(f"Vocab size: {decoder.base.config.vocab_size}")
    print()
    
    # Initial state
    torch.cuda.synchronize()
    mem_start = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"Initial memory: {mem_start:.1f} MB")
    
    # Create activation
    test_activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    torch.cuda.synchronize()
    mem_after_activation = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"After creating activation: {mem_after_activation:.1f} MB (+{mem_after_activation - mem_start:.1f} MB)")
    
    # Analyze memory during generation
    print("\nMemory during generation (original method):")
    print("-" * 50)
    
    # Manually step through generation to profile
    activation_input = test_activation.to(decoder.proj.weight.dtype)
    B, d_model = activation_input.shape
    
    # Get embedding tables
    input_emb_table = decoder.base.get_input_embeddings().weight
    output_emb_table = decoder.base.get_output_embeddings().weight
    
    # Build initial sequence
    if decoder.prompt_left_emb is not None:
        decoder.prompt_left_emb = decoder.prompt_left_emb.to(device)
    if decoder.prompt_right_emb is not None:
        decoder.prompt_right_emb = decoder.prompt_right_emb.to(device)
        
    parts = []
    if decoder.prompt_left_emb is not None:
        parts.append(decoder.prompt_left_emb.expand(B, -1, -1))
    if decoder.prompt_right_emb is not None:
        parts.append(decoder.prompt_right_emb.expand(B, -1, -1))
    a_proj = decoder.proj(activation_input).unsqueeze(1)
    parts.append(a_proj)
    seq_embs = torch.cat(parts, dim=1)
    
    torch.cuda.synchronize()
    mem_after_init = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"After initial setup: {mem_after_init:.1f} MB (+{mem_after_init - mem_after_activation:.1f} MB)")
    
    # Track memory per token
    mem_per_token = []
    intermediate_tensors = []  # Keep references to prevent garbage collection
    
    for i in range(max_length):
        # Forward pass through model
        out = decoder.base(inputs_embeds=seq_embs, output_hidden_states=True)
        h_last = out.last_hidden_state if hasattr(out, "last_hidden_state") else out.hidden_states[-1]
        logits_t = decoder.out(h_last[:, -1])
        
        # Gumbel-softmax
        ste_token_dist = torch.nn.functional.gumbel_softmax(
            logits_t / 1.0,
            tau=1.0,
            hard=True
        )
        
        # Get embeddings
        emb_t_input = ste_token_dist @ input_emb_table
        emb_t_output = ste_token_dist @ output_emb_table
        
        # Update sequence
        seq_embs = torch.cat([seq_embs, emb_t_input.unsqueeze(1)], dim=1)
        
        # Store intermediates to prevent GC
        intermediate_tensors.append((out, h_last, logits_t, ste_token_dist, emb_t_output))
        
        torch.cuda.synchronize()
        mem_current = torch.cuda.memory_allocated() / 1024 / 1024
        mem_delta = mem_current - (mem_per_token[-1] if mem_per_token else mem_after_init)
        mem_per_token.append(mem_current)
        
        if i < 5 or i == max_length - 1:  # Print first few and last
            print(f"Token {i+1}: {mem_current:.1f} MB (+{mem_delta:.1f} MB)")
    
    print(f"\nTotal memory after generation: {mem_per_token[-1]:.1f} MB")
    print(f"Average memory per token: {(mem_per_token[-1] - mem_after_init) / max_length:.1f} MB")
    
    # Create loss and backward
    all_embeds = torch.stack([t[4] for t in intermediate_tensors], dim=1)
    loss = all_embeds.sum()
    
    torch.cuda.synchronize()
    mem_before_backward = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"\nBefore backward: {mem_before_backward:.1f} MB")
    
    loss.backward()
    
    torch.cuda.synchronize()
    mem_after_backward = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"After backward: {mem_after_backward:.1f} MB")
    
    # Analyze what's taking memory
    print("\nMemory breakdown analysis:")
    print("-" * 50)
    
    # Model parameters
    model_params = sum(p.numel() for p in decoder.parameters()) * 2 / 1024 / 1024  # bf16
    print(f"Model parameters: {model_params:.1f} MB")
    
    # Analyze tensor sizes
    print("\nKey tensor sizes per token:")
    vocab_size = decoder.base.config.vocab_size
    n_layers = decoder.base.config.n_layer
    n_heads = decoder.base.config.n_head
    
    # Per-token memory
    hidden_states_size = batch_size * d_model * 2 / 1024 / 1024  # bf16
    attention_size = batch_size * n_layers * n_heads * d_model * 2 / 1024 / 1024
    logits_size = batch_size * vocab_size * 2 / 1024 / 1024
    
    print(f"- Hidden states: {hidden_states_size:.2f} MB")
    print(f"- Attention (est): {attention_size:.2f} MB") 
    print(f"- Logits: {logits_size:.2f} MB")
    print(f"- Total (theoretical): {hidden_states_size + attention_size + logits_size:.2f} MB")
    
    # Compare with encoder
    print("\n" + "=" * 60)
    print("Comparing with full training step memory usage")
    
    # Clean up
    del decoder, intermediate_tensors, all_embeds, loss
    torch.cuda.empty_cache()
    gc.collect()


def analyze_full_training_step():
    """Analyze memory in a full training step."""
    
    if not torch.cuda.is_available():
        print("CUDA required")
        return
        
    from lens.models.encoder import Encoder, EncoderConfig
    from lens.training.loop import train_step
    
    torch.cuda.empty_cache()
    gc.collect()
    
    device = torch.device("cuda")
    model_name = "gpt2"
    batch_size = 8
    t_text = 16
    
    # Create models
    decoder_config = DecoderConfig(
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
    
    decoder = Decoder(decoder_config).to(device)
    encoder = Encoder(encoder_config).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    decoder.set_prompt("The meaning of <embed> is:", tokenizer)
    
    d_model = decoder.base.config.hidden_size
    
    print("\nFull Training Step Memory Analysis")
    print("=" * 60)
    
    torch.cuda.synchronize()
    mem_start = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"After model creation: {mem_start:.1f} MB")
    
    # Create batch
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
        "orig": None
    }
    
    loss_fns = {
        "T_text": t_text,
        "tau": 1.0,
        "alpha": 0.1,
        "lm_base_weight": 1.0,
        "kl_base_weight": 0.0,
        "entropy_weight": 0.0,
        "mse_weight": 1.0
    }
    
    torch.cuda.synchronize()
    mem_before_step = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"Before train step: {mem_before_step:.1f} MB")
    
    # Run training step
    losses = train_step(batch, models, loss_fns)
    
    torch.cuda.synchronize()
    mem_after_forward = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"After forward pass: {mem_after_forward:.1f} MB (+{mem_after_forward - mem_before_step:.1f} MB)")
    
    # Backward
    losses['total'].backward()
    
    torch.cuda.synchronize()
    mem_after_backward = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"After backward pass: {mem_after_backward:.1f} MB (+{mem_after_backward - mem_after_forward:.1f} MB)")
    
    # Analyze where memory goes
    print("\nMemory distribution:")
    print("-" * 50)
    
    # Model sizes
    decoder_params = sum(p.numel() for p in decoder.parameters()) * 2 / 1024 / 1024
    encoder_params = sum(p.numel() for p in encoder.parameters()) * 2 / 1024 / 1024
    print(f"Decoder parameters: {decoder_params:.1f} MB")
    print(f"Encoder parameters: {encoder_params:.1f} MB")
    print(f"Total model params: {decoder_params + encoder_params:.1f} MB")
    
    # Generation overhead
    generation_overhead = mem_after_forward - mem_before_step
    print(f"\nGeneration overhead: {generation_overhead:.1f} MB")
    print(f"Per token: {generation_overhead / t_text:.1f} MB")
    
    # Compare to total memory
    total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
    print(f"\nTotal GPU memory: {total_gpu_mem:.1f} GB")
    print(f"Current usage: {mem_after_backward / 1024:.1f} GB ({mem_after_backward / (total_gpu_mem * 1024) * 100:.1f}%)")


if __name__ == "__main__":
    profile_generation_memory()
    print("\n" * 2)
    analyze_full_training_step()