"""Investigate why gradient checkpointing isn't saving 75% of memory."""

import torch
import gc
from lens.models.decoder import Decoder, DecoderConfig
from transformers import AutoTokenizer


def trace_checkpoint_memory():
    """Trace exactly what's being saved/discarded with checkpointing."""
    
    if not torch.cuda.is_available():
        print("CUDA required")
        return
    
    torch.cuda.empty_cache()
    gc.collect()
    
    device = torch.device("cuda")
    model_name = "gpt2"
    batch_size = 8
    max_length = 16
    checkpoint_interval = 4
    
    # Create decoder
    decoder_config = DecoderConfig(
        model_name=model_name,
        n_prompt_tokens=0,
        base_model=False,
        projection_layer=True,
        output_head=True,
        embedding_head=False,
        eye_init=True,
        trainable_prompts=True,
        use_checkpointing=True  # Use checkpointed version
    )
    
    decoder = Decoder(decoder_config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    decoder.set_prompt("The meaning of <embed> is:", tokenizer)
    
    d_model = decoder.base.config.hidden_size
    
    print("Gradient Checkpointing Memory Analysis")
    print("=" * 60)
    print(f"Checkpointing every {checkpoint_interval} tokens")
    print()
    
    # Create activation
    test_activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    
    # Manual generation to trace memory
    print("Tracing memory during checkpointed generation:")
    print("-" * 50)
    
    from torch.utils.checkpoint import checkpoint
    
    # Setup initial sequence
    activation_input = test_activation.to(decoder.proj.weight.dtype)
    parts = []
    if decoder.prompt_left_emb is not None:
        decoder.prompt_left_emb = decoder.prompt_left_emb.to(device)
        parts.append(decoder.prompt_left_emb.expand(batch_size, -1, -1))
    if decoder.prompt_right_emb is not None:
        decoder.prompt_right_emb = decoder.prompt_right_emb.to(device)
        parts.append(decoder.prompt_right_emb.expand(batch_size, -1, -1))
    a_proj = decoder.proj(activation_input).unsqueeze(1)
    parts.append(a_proj)
    seq_embs = torch.cat(parts, dim=1)
    
    input_emb_table = decoder.base.get_input_embeddings().weight
    output_emb_table = decoder.base.get_output_embeddings().weight
    
    # Track what's kept vs discarded
    saved_tensors = []
    checkpointed_ranges = []
    
    torch.cuda.synchronize()
    mem_start = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"Memory at start: {mem_start:.1f} MB")
    
    # Generation step function
    def generation_step(seq_embs_input, step_idx):
        # Model forward pass
        out = decoder.base(inputs_embeds=seq_embs_input, output_hidden_states=True)
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
        hard_ids = ste_token_dist.argmax(dim=-1)
        
        return logits_t, emb_t_input, emb_t_output, hard_ids, out
    
    # Generate tokens
    all_outputs = []
    for step in range(max_length):
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1024 / 1024
        
        if step % checkpoint_interval == 0 and step > 0:
            # Checkpointed execution
            print(f"\n[CHECKPOINT] Token {step}:")
            outputs = checkpoint(generation_step, seq_embs, step, use_reentrant=False)
            logits_t, emb_t_input, emb_t_output, hard_ids, model_out = outputs
            checkpointed_ranges.append(step)
        else:
            # Regular execution
            outputs = generation_step(seq_embs, step)
            logits_t, emb_t_input, emb_t_output, hard_ids, model_out = outputs
            # Save reference to prevent GC
            saved_tensors.append((logits_t, model_out))
        
        seq_embs = torch.cat([seq_embs, emb_t_input.unsqueeze(1)], dim=1)
        all_outputs.append(emb_t_output)
        
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated() / 1024 / 1024
        mem_delta = mem_after - mem_before
        
        status = "[CKPT]" if step in checkpointed_ranges else "[SAVE]"
        print(f"Token {step} {status}: {mem_after:.1f} MB (+{mem_delta:.1f} MB)")
    
    # Stack outputs
    text_embs = torch.stack(all_outputs, dim=1)
    
    torch.cuda.synchronize()
    mem_before_backward = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"\nBefore backward: {mem_before_backward:.1f} MB")
    print(f"Total forward memory: {mem_before_backward - mem_start:.1f} MB")
    
    # Backward pass
    loss = text_embs.sum()
    loss.backward()
    
    torch.cuda.synchronize()
    mem_after_backward = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"After backward: {mem_after_backward:.1f} MB")
    
    # Analysis
    print("\nCheckpointing Analysis:")
    print("-" * 50)
    print(f"Tokens checkpointed: {len(checkpointed_ranges)} / {max_length}")
    print(f"Tokens saved in memory: {max_length - len(checkpointed_ranges)} / {max_length}")
    print(f"Expected memory ratio: {(max_length - len(checkpointed_ranges)) / max_length:.1%}")
    print(f"But actual savings are only ~22%")
    
    print("\nWhy the discrepancy?")
    print("-" * 50)
    print("1. Checkpointing only affects intermediate activations within generation_step")
    print("2. The sequence embeddings (seq_embs) grow at EVERY step and must be kept")
    print("3. Output embeddings (text_embs) must be kept for the encoder")
    print("4. The transformer's internal checkpointing is not active")
    
    # Calculate growing sequence memory
    print("\nGrowing sequence memory:")
    for i in [1, 4, 8, 12, 16]:
        seq_len = 3 + i  # prompt + generated tokens
        seq_mem = batch_size * seq_len * d_model * 2 / 1024 / 1024  # bf16
        print(f"  After token {i}: seq_len={seq_len}, memory={seq_mem:.1f} MB")
    
    # Total sequence memory
    total_seq_mem = sum(batch_size * (3 + i) * d_model * 2 / 1024 / 1024 for i in range(1, max_length + 1))
    print(f"\nTotal cumulative sequence memory: {total_seq_mem:.1f} MB")
    print("This CANNOT be checkpointed because each step depends on the previous sequence!")


def compare_actual_implementations():
    """Compare the actual implementations side by side."""
    
    if not torch.cuda.is_available():
        print("\nCUDA required")
        return
        
    print("\n\n" + "=" * 60)
    print("Comparing Actual Memory Usage")
    print("=" * 60)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    device = torch.device("cuda")
    model_name = "gpt2"
    batch_size = 8
    max_length = 16
    
    for use_checkpoint in [False, True]:
        torch.cuda.empty_cache()
        gc.collect()
        
        decoder_config = DecoderConfig(
            model_name=model_name,
            n_prompt_tokens=0,
            base_model=False,
            projection_layer=True,
            output_head=True,
            embedding_head=False,
            eye_init=True,
            trainable_prompts=True,
            use_checkpointing=use_checkpoint
        )
        
        decoder = Decoder(decoder_config).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        decoder.set_prompt("The meaning of <embed> is:", tokenizer)
        
        d_model = decoder.base.config.hidden_size
        test_activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1024 / 1024
        
        # Generate
        if use_checkpoint:
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
        
        # Force computation graph
        loss = gen.generated_text_embeddings.sum()
        
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated() / 1024 / 1024
        
        method = "Checkpointed" if use_checkpoint else "Original"
        print(f"\n{method} method:")
        print(f"  Memory used: {mem_after - mem_before:.1f} MB")
        
        if use_checkpoint:
            # Calculate what CAN be saved
            # Only the transformer outputs for non-checkpointed steps
            saved_steps = max_length - (max_length // 4)  # steps not checkpointed
            # Each step saves transformer output tensors
            # But NOT the growing sequence or output embeddings
            potential_savings = saved_steps * (batch_size * d_model * 12 * 2) / 1024 / 1024  # 12 layers
            print(f"  Theoretical savings from checkpointing: ~{potential_savings:.1f} MB")
            print(f"  But growing sequences and outputs still required!")
        
        del decoder, gen, loss


if __name__ == "__main__":
    trace_checkpoint_memory()
    compare_actual_implementations()