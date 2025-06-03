#!/usr/bin/env python3
"""Debug KV cache gradient issues."""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def compare_generation_methods():
    """Compare outputs and gradients between generate_soft and generate_soft_kv_cached."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    batch_size = 1
    d_model = 768
    seq_length = 4
    
    print("Debugging KV Cache Gradients")
    print("=" * 60)
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42) if torch.cuda.is_available() else None
    
    # Create two identical decoders
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=False,  # Start with simple case
    )
    
    decoder1 = Decoder(config).to(device)
    decoder1.set_prompt("explain <embed>:", tokenizer)
    
    decoder2 = Decoder(config).to(device)
    decoder2.set_prompt("explain <embed>:", tokenizer)
    
    # Ensure they have identical weights
    decoder2.load_state_dict(decoder1.state_dict())
    
    # Create identical activations
    torch.manual_seed(123)
    activation1 = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    activation2 = activation1.detach().clone().requires_grad_(True)
    
    # Generate with both methods
    print("\nGenerating with both methods...")
    gen1 = decoder1.generate_soft(activation1, max_length=seq_length, gumbel_tau=1.0)
    gen2 = decoder2.generate_soft_kv_cached(activation2, max_length=seq_length, gumbel_tau=1.0)
    
    # Compare outputs
    print("\nComparing outputs:")
    emb_diff = (gen1.generated_text_embeddings - gen2.generated_text_embeddings).abs().max().item()
    logits_diff = (gen1.raw_lm_logits - gen2.raw_lm_logits).abs().max().item()
    ids_same = torch.equal(gen1.hard_token_ids, gen2.hard_token_ids)
    
    print(f"  Embeddings max diff: {emb_diff:.2e}")
    print(f"  Logits max diff: {logits_diff:.2e}")
    print(f"  Hard IDs identical: {ids_same}")
    
    # Detailed comparison of logits
    if logits_diff > 1e-6:
        print("\n  Detailed logits comparison:")
        for t in range(seq_length):
            t_diff = (gen1.raw_lm_logits[:, t] - gen2.raw_lm_logits[:, t]).abs().max().item()
            print(f"    Time step {t}: max diff = {t_diff:.2e}")
            if t_diff > 1e-6:
                # Find which vocab items differ most
                diffs = (gen1.raw_lm_logits[0, t] - gen2.raw_lm_logits[0, t]).abs()
                top_diffs = diffs.topk(5)
                print(f"      Top 5 differing vocab items:")
                for i, (diff, idx) in enumerate(zip(top_diffs.values, top_diffs.indices)):
                    print(f"        {idx.item()}: diff = {diff.item():.2e}")
    
    # Compute gradients
    print("\nComputing gradients...")
    loss1 = gen1.generated_text_embeddings.sum()
    loss2 = gen2.generated_text_embeddings.sum()
    
    loss1.backward()
    loss2.backward()
    
    # Compare gradients
    print("\nComparing gradients:")
    act_grad_diff = (activation1.grad - activation2.grad).abs().max().item()
    proj_weight_grad_diff = (decoder1.proj.weight.grad - decoder2.proj.weight.grad).abs().max().item()
    proj_bias_grad_diff = (decoder1.proj.bias.grad - decoder2.proj.bias.grad).abs().max().item()
    
    print(f"  Activation grad diff: {act_grad_diff:.2e}")
    print(f"  Projection weight grad diff: {proj_weight_grad_diff:.2e}")
    print(f"  Projection bias grad diff: {proj_bias_grad_diff:.2e}")
    
    # Check if the issue is in the forward or backward pass
    print("\n\nChecking loss values:")
    print(f"  generate_soft loss: {loss1.item():.6f}")
    print(f"  generate_soft_kv_cached loss: {loss2.item():.6f}")
    print(f"  Loss difference: {abs(loss1.item() - loss2.item()):.2e}")


def trace_kv_cache_computation():
    """Trace through KV cache computation to find where differences arise."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\n\nTracing KV Cache Computation")
    print("=" * 60)
    
    # Simple setup
    batch_size = 1
    d_model = 768
    seq_length = 2  # Very short for debugging
    
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
    )
    
    decoder = Decoder(config).to(device)
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    # Create activation
    torch.manual_seed(42)
    activation = torch.randn(batch_size, d_model, device=device)
    
    # Manually trace through both methods
    print("\nManual generation with naive method:")
    
    # Get initial embeddings
    prompt_left_emb = decoder.prompt_left_emb
    prompt_right_emb = decoder.prompt_right_emb
    
    # Build initial sequence
    parts = []
    if prompt_left_emb is not None:
        parts.append(prompt_left_emb.expand(batch_size, -1, -1))
    a_proj = decoder.proj(activation).unsqueeze(1)
    parts.append(a_proj)
    if prompt_right_emb is not None:
        parts.append(prompt_right_emb.expand(batch_size, -1, -1))
    
    seq_embs = torch.cat(parts, dim=1)
    print(f"  Initial sequence shape: {seq_embs.shape}")
    
    # First forward pass
    out = decoder.base(inputs_embeds=seq_embs, output_hidden_states=True)
    logits = decoder.out(out.last_hidden_state[:, -1])
    print(f"  First logits sum: {logits.sum().item():.6f}")
    
    # Check numerical stability in Gumbel-Softmax
    print("\nChecking Gumbel-Softmax stability:")
    with torch.amp.autocast('cuda', enabled=False):
        logits_f32 = logits.float()
        logits_stable = logits_f32 - logits_f32.max(dim=-1, keepdim=True)[0].detach()
        gumbel_out = F.gumbel_softmax(logits_stable, tau=1.0, hard=True)
    
    print(f"  Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
    print(f"  Stable logits range: [{logits_stable.min().item():.2f}, {logits_stable.max().item():.2f}]")
    print(f"  Gumbel output sum: {gumbel_out.sum(dim=-1).item()}")


if __name__ == "__main__":
    compare_generation_methods()
    trace_kv_cache_computation()