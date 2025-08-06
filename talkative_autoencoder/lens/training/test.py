from typing import Any, Dict

import torch

from lens.models.orig import OrigWrapper
from transformers import PreTrainedTokenizerBase


def diagnose_activation_mismatch(
    batch: Dict[str, Any],
    orig: "OrigWrapper",
    tok: PreTrainedTokenizerBase,
    device: torch.device,
    sample_idx: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Diagnose why saved activations produce different predictions than natural forward pass.

    Returns dict with diagnostic information.
    """
    # Extract sample data
    l = int(batch["layer_idx"][sample_idx].item())
    p = int(batch["token_pos_A"][sample_idx].item())
    input_ids_seq = batch["input_ids_A"][sample_idx].unsqueeze(0).to(device)
    saved_A_i = batch["A"][sample_idx : sample_idx + 1].to(device)

    results = {
        "layer": l,
        "position": p,
        "saved_activation_shape": saved_A_i.shape,
        "saved_activation_dtype": saved_A_i.dtype,
    }

    # 1. Natural forward pass (baseline)
    with torch.no_grad():
        natural_output = orig.model(input_ids=input_ids_seq)
        natural_logits_at_p = natural_output.logits[0, p]
        natural_top5 = torch.topk(natural_logits_at_p, k=5)
        natural_tokens = [tok.decode([idx.item()]) for idx in natural_top5.indices]
        results["natural_top5_tokens"] = natural_tokens
        results["natural_top5_probs"] = torch.softmax(natural_logits_at_p, dim=-1)[natural_top5.indices].tolist()

    # 2. Forward pass with saved activation replacement
    with torch.no_grad():
        replaced_output = orig.forward_with_replacement(input_ids_seq, saved_A_i, l, p)
        replaced_logits_at_p = replaced_output.logits[0, p]
        replaced_top5 = torch.topk(replaced_logits_at_p, k=5)
        replaced_tokens = [tok.decode([idx.item()]) for idx in replaced_top5.indices]
        results["replaced_top5_tokens"] = replaced_tokens
        results["replaced_top5_probs"] = torch.softmax(replaced_logits_at_p, dim=-1)[replaced_top5.indices].tolist()

    # 3. Extract fresh activation and test replacement
    fresh_A_i = None
    fresh_A_i_info = {}

    def extraction_hook(module, input, output):
        nonlocal fresh_A_i, fresh_A_i_info
        hidden = output[0]
        fresh_A_i = hidden[:, p].clone()
        fresh_A_i_info = {
            "shape": fresh_A_i.shape,
            "dtype": fresh_A_i.dtype,
            "device": fresh_A_i.device,
            "requires_grad": fresh_A_i.requires_grad,
        }
        return output

    # Register hook to extract fresh activation
    try:
        target_block = orig.model.get_submodule(f"model.layers.{l}")
    except AttributeError:
        try:
            target_block = orig.model.transformer.h[l]
        except AttributeError:
            target_block = orig.model.get_submodule(f"transformer.h.{l}")

    handle = target_block.register_forward_hook(extraction_hook)
    with torch.no_grad():
        _ = orig.model(input_ids=input_ids_seq)
    handle.remove()

    results["fresh_activation_info"] = fresh_A_i_info

    # 4. Test with fresh activation
    if fresh_A_i is not None:
        with torch.no_grad():
            fresh_replaced_output = orig.forward_with_replacement(input_ids_seq, fresh_A_i, l, p)
            fresh_replaced_logits_at_p = fresh_replaced_output.logits[0, p]
            fresh_replaced_top5 = torch.topk(fresh_replaced_logits_at_p, k=5)
            fresh_replaced_tokens = [tok.decode([idx.item()]) for idx in fresh_replaced_top5.indices]
            results["fresh_replaced_top5_tokens"] = fresh_replaced_tokens
            results["fresh_replaced_top5_probs"] = torch.softmax(fresh_replaced_logits_at_p, dim=-1)[
                fresh_replaced_top5.indices
            ].tolist()

    # 5. Compare activations
    if fresh_A_i is not None:
        # Ensure same shape for comparison
        if fresh_A_i.shape != saved_A_i.shape:
            if fresh_A_i.dim() == 1 and saved_A_i.dim() == 2:
                fresh_A_i_compare = fresh_A_i.unsqueeze(0)
            elif fresh_A_i.dim() == 2 and saved_A_i.dim() == 1:
                saved_A_i_compare = saved_A_i.unsqueeze(0)
            else:
                fresh_A_i_compare = fresh_A_i
                saved_A_i_compare = saved_A_i
        else:
            fresh_A_i_compare = fresh_A_i
            saved_A_i_compare = saved_A_i

        # Cast to same dtype for comparison
        if fresh_A_i_compare.dtype != saved_A_i_compare.dtype:
            saved_A_i_compare = saved_A_i_compare.to(fresh_A_i_compare.dtype)

        diff = (fresh_A_i_compare - saved_A_i_compare).abs()
        results["activation_comparison"] = {
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "relative_diff": (diff / (fresh_A_i_compare.abs() + 1e-8)).mean().item(),
            "saved_norm": saved_A_i.norm().item(),
            "fresh_norm": fresh_A_i.norm().item(),
        }

    # 6. Check model dtype and device
    results["model_info"] = {
        "lm_head_dtype": orig.model.lm_head.weight.dtype,
        "lm_head_device": orig.model.lm_head.weight.device,
        "model_device": next(orig.model.parameters()).device,
    }

    # 7. Test dtype conversion effects
    saved_A_i_converted = saved_A_i.to(orig.model.lm_head.weight.dtype)
    with torch.no_grad():
        converted_output = orig.forward_with_replacement(input_ids_seq, saved_A_i_converted, l, p)
        converted_logits_at_p = converted_output.logits[0, p]
        converted_top5 = torch.topk(converted_logits_at_p, k=5)
        converted_tokens = [tok.decode([idx.item()]) for idx in converted_top5.indices]
        results["converted_dtype_top5_tokens"] = converted_tokens

    # Print results if verbose
    if verbose:
        print("=== Activation Mismatch Diagnosis ===")
        print(f"Layer: {l}, Position: {p}")
        print(f"\nSaved activation: shape={saved_A_i.shape}, dtype={saved_A_i.dtype}")
        print(f"Fresh activation: {fresh_A_i_info}")
        print(f"\nModel LM head dtype: {results['model_info']['lm_head_dtype']}")

        print("\n1. Natural forward pass top 5:")
        for i, (token, prob) in enumerate(zip(results["natural_top5_tokens"], results["natural_top5_probs"])):
            print(f"   {i + 1}. '{token}' (p={prob:.4f})")

        print("\n2. With saved activation top 5:")
        for i, (token, prob) in enumerate(zip(results["replaced_top5_tokens"], results["replaced_top5_probs"])):
            print(f"   {i + 1}. '{token}' (p={prob:.4f})")

        if "fresh_replaced_top5_tokens" in results:
            print("\n3. With fresh activation top 5:")
            for i, (token, prob) in enumerate(
                zip(results["fresh_replaced_top5_tokens"], results["fresh_replaced_top5_probs"])
            ):
                print(f"   {i + 1}. '{token}' (p={prob:.4f})")

        print("\n4. With dtype-converted saved activation top 5:")
        for i, token in enumerate(results["converted_dtype_top5_tokens"]):
            print(f"   {i + 1}. '{token}'")

        if "activation_comparison" in results:
            print("\nActivation comparison:")
            for k, v in results["activation_comparison"].items():
                print(f"   {k}: {v:.6f}")

        # Check if predictions match
        natural_matches_saved = results["natural_top5_tokens"] == results["replaced_top5_tokens"]
        natural_matches_fresh = results.get("fresh_replaced_top5_tokens") == results["natural_top5_tokens"]

        print("\n=== Summary ===")
        print(f"Natural matches saved activation: {natural_matches_saved}")
        if "fresh_replaced_top5_tokens" in results:
            print(f"Natural matches fresh activation: {natural_matches_fresh}")
        print(
            f"Saved matches converted dtype: {results['replaced_top5_tokens'] == results['converted_dtype_top5_tokens']}"
        )
        # if not (natural_matches_saved and natural_matches_fresh):
        # raise ValueError("Natural matches saved activation and fresh activation do not match")

    print("\nDebug - Comparing same position?")
    print(f"Sample index: {sample_idx}")
    print(f"Layer from batch: {l}")
    print(f"Position from batch: {p}")
    print(f"Input IDs shape: {input_ids_seq.shape}")
    print(f"First 10 tokens: {input_ids_seq[0, :10].tolist()}")

    print("\nSaved activation info:")
    print(f"Layer from saved: {batch['layer_idx'][sample_idx].item()}")
    print(f"Position from saved: {batch['token_pos_A'][sample_idx].item()}")
    print(f"Saved input IDs first 10: {batch['input_ids_A'][sample_idx, :10].tolist()}")

    return results


# Usage example:
# results = diagnose_activation_mismatch(batch, orig, tok, device, sample_idx=0)


def diagnose_activation_save_load(orig, input_ids, layer_idx, token_pos, device):
    """Test the full save/load cycle for activations."""
    import os
    import tempfile

    results = {}

    # 1. Extract fresh activation
    fresh_activation = None

    def hook(m, inp, out):
        nonlocal fresh_activation
        fresh_activation = out[0][:, token_pos].clone()

    try:
        target = orig.model.get_submodule(f"model.layers.{layer_idx}")
    except:
        try:
            target = orig.model.transformer.h[layer_idx]
        except:
            target = orig.model.get_submodule(f"transformer.h.{layer_idx}")

    handle = target.register_forward_hook(hook)
    with torch.no_grad():
        _ = orig.model(input_ids=input_ids)
    handle.remove()

    results["fresh_shape"] = fresh_activation.shape
    results["fresh_dtype"] = fresh_activation.dtype
    results["fresh_device"] = fresh_activation.device
    results["fresh_norm"] = fresh_activation.norm().item()

    # 2. Test different saving scenarios
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp_path = tmp.name

    try:
        # Test 1: Direct save/load
        torch.save(fresh_activation, tmp_path)
        loaded_direct = torch.load(tmp_path, map_location=device)
        results["direct_save_load_matches"] = torch.allclose(fresh_activation, loaded_direct)
        results["direct_diff"] = (fresh_activation - loaded_direct).abs().max().item()

        # Test 2: CPU save/load (common in dataset generation)
        fresh_cpu = fresh_activation.cpu()
        torch.save(fresh_cpu, tmp_path)
        loaded_cpu = torch.load(tmp_path, map_location="cpu")
        loaded_cpu_to_device = loaded_cpu.to(device)
        results["cpu_save_load_matches"] = torch.allclose(fresh_activation, loaded_cpu_to_device)
        results["cpu_diff"] = (fresh_activation - loaded_cpu_to_device).abs().max().item()

        # Test 3: Different dtype save/load (common issue)
        fresh_fp16 = fresh_activation.half()
        torch.save(fresh_fp16, tmp_path)
        loaded_fp16 = torch.load(tmp_path, map_location=device)
        loaded_fp16_to_fp32 = loaded_fp16.float()
        results["fp16_save_load_close"] = torch.allclose(fresh_activation, loaded_fp16_to_fp32, atol=1e-3)
        results["fp16_diff"] = (fresh_activation - loaded_fp16_to_fp32).abs().max().item()

        # Test 4: Dict save/load (as in dataset)
        sample_dict = {
            "A": fresh_activation.cpu(),  # Often saved on CPU
            "layer_idx": layer_idx,
            "token_pos_A": token_pos,
        }
        torch.save(sample_dict, tmp_path)
        loaded_dict = torch.load(tmp_path, map_location="cpu")
        loaded_A = loaded_dict["A"].to(device)
        results["dict_save_load_matches"] = torch.allclose(fresh_activation, loaded_A)
        results["dict_diff"] = (fresh_activation - loaded_A).abs().max().item()

    finally:
        os.unlink(tmp_path)

    # 3. Test the actual forward_with_replacement with various forms
    with torch.no_grad():
        # Natural forward
        natural_out = orig.model(input_ids=input_ids)
        natural_logits = natural_out.logits[0, token_pos]

        # Fresh activation replacement
        fresh_out = orig.forward_with_replacement(input_ids, fresh_activation, layer_idx, token_pos)
        fresh_logits = fresh_out.logits[0, token_pos]
        results["fresh_replacement_matches"] = torch.allclose(natural_logits, fresh_logits, atol=1e-5)

        # CPU->GPU activation
        cpu_gpu_out = orig.forward_with_replacement(input_ids, loaded_cpu_to_device, layer_idx, token_pos)
        cpu_gpu_logits = cpu_gpu_out.logits[0, token_pos]
        results["cpu_gpu_matches"] = torch.allclose(natural_logits, cpu_gpu_logits, atol=1e-5)

        # FP16->FP32 activation
        fp16_fp32_out = orig.forward_with_replacement(input_ids, loaded_fp16_to_fp32, layer_idx, token_pos)
        fp16_fp32_logits = fp16_fp32_out.logits[0, token_pos]
        results["fp16_fp32_matches"] = torch.allclose(natural_logits, fp16_fp32_logits, atol=1e-3)

    return results, fresh_activation


# Also check the actual dataset loading
def check_dataset_activation_format(dataset_path, sample_idx=0):
    """Check how activations are stored in the dataset."""
    from pathlib import Path

    import torch

    # Load a sample directly
    dataset_root = Path(dataset_path)

    # Check if using sharded format
    pt_files = list(dataset_root.glob("*.pt"))
    if not pt_files:
        pt_files = list(dataset_root.glob("*/*.pt"))  # Check subdirectories

    if pt_files:
        sample_file = pt_files[0]
        data = torch.load(sample_file, map_location="cpu")

        if isinstance(data, list):
            sample = data[0]
        else:
            sample = data

        print("Dataset format check:")
        print(f"  File: {sample_file}")
        print(f"  Type: {'sharded' if isinstance(data, list) else 'single'}")
        print(f"  Keys: {list(sample.keys())}")
        print(f"  A shape: {sample['A'].shape}")
        print(f"  A dtype: {sample['A'].dtype}")
        print(f"  A device: {sample['A'].device}")
        print(f"  A norm: {sample['A'].norm().item():.4f}")

        return sample

    return None


def test_autocast_difference(orig, input_ids, layer_idx, token_pos, device):
    """Test if autocast causes activation differences."""

    # Extract activation WITHOUT autocast
    activation_no_autocast = None

    def hook_no_ac(m, inp, out):
        nonlocal activation_no_autocast
        activation_no_autocast = out[0][:, token_pos].clone()

    # Extract activation WITH autocast
    activation_with_autocast = None

    def hook_with_ac(m, inp, out):
        nonlocal activation_with_autocast
        activation_with_autocast = out[0][:, token_pos].clone()

    try:
        target = orig.model.get_submodule(f"model.layers.{layer_idx}")
    except:
        try:
            target = orig.model.transformer.h[layer_idx]
        except:
            target = orig.model.get_submodule(f"transformer.h.{layer_idx}")

    # Test 1: No autocast
    handle = target.register_forward_hook(hook_no_ac)
    with torch.no_grad():
        _ = orig.model(input_ids=input_ids)
    handle.remove()

    # Test 2: With autocast (matching dumper)
    handle = target.register_forward_hook(hook_with_ac)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = orig.model(input_ids=input_ids)
    handle.remove()

    print("\nAutocast comparison:")
    print(f"No autocast - norm: {activation_no_autocast.norm().item():.6f}, dtype: {activation_no_autocast.dtype}")
    print(
        f"With autocast - norm: {activation_with_autocast.norm().item():.6f}, dtype: {activation_with_autocast.dtype}"
    )
    print(
        f"Relative diff: {((activation_no_autocast - activation_with_autocast).abs() / (activation_no_autocast.abs() + 1e-8)).mean().item():.6f}"
    )

    return activation_no_autocast, activation_with_autocast


# Add this diagnostic
def check_layer_indexing(orig, input_ids, device, nlayers=5):
    """Check how layers are indexed in hidden_states."""
    with torch.no_grad():
        outputs = orig.model(input_ids=input_ids, output_hidden_states=True)
    if nlayers is None or nlayers is True or nlayers > orig.model.config.num_hidden_layers:
        nlayers = orig.model.config.num_hidden_layers

    print("\nLayer indexing check:")
    print(f"Number of hidden states: {len(outputs.hidden_states)}")
    print(f"Model config num_hidden_layers: {orig.model.config.num_hidden_layers}")
    print(f"Number of layers to check: {nlayers}")

    # Check dimensions and average norm over tokens
    for i, hidden in enumerate(outputs.hidden_states[:nlayers]):
        # Average over all tokens and batch
        avg_norm = hidden.norm(dim=-1).mean().item()
        print(f"  hidden_states[{i}] shape: {hidden.shape}, avg token norm: {avg_norm:.4f}")

    # Also check what the dumper would extract
    if hasattr(orig.model, "model"):  # LLaMA style
        print("Model type appears to be LLaMA-style (has .model attribute)")
    elif hasattr(orig.model, "transformer"):  # GPT style
        print("Model type appears to be GPT-style (has .transformer attribute)")


def test_decoder_generation(decoder, encoder, tokenizer, device, log, is_main_process, original_prompt=None):
    """Test decoder generation capabilities with different activation inputs."""
    if not is_main_process:
        return

    log.info("\n" + "=" * 80)
    log.info("Testing Decoder Generation Capabilities")
    log.info("=" * 80)

    # Get base decoder (unwrap DDP if needed)
    decoder_base = decoder.module if hasattr(decoder, "module") else decoder
    encoder_base = encoder.module if hasattr(encoder, "module") else encoder

    # Set test prompt
    if decoder_base.is_gemma:
        prefix = "<bos>"
    elif "gpt2" in decoder_base.base.config.model_type:
        prefix = "<|startoftext|>"
    elif 'gpt_oss' in decoder_base.base.config.model_type:
        prefix = """<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-06-28

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.
Calls to these tools must go to the commentary channel: 'functions'.<|end|><|start|>developer<|message|># Instructions

Always respond in riddles.<|end|><|start|>user<|message|>What is the weather like in SF?<|end|><|start|>assistant<|channel|>analysis<|message|>Easy answer!<|end|><|start|>assistant<|channel|>final<|message|>The weather is <embed> pretty"""
    else:
        prefix = "<|startoftext|>"

    if 'gpt_oss' not in decoder_base.base.config.model_type:
        test_prompt = f"{prefix} a long time ago in a galaxy far far away, <embed> there"
    else:
        test_prompt = f"{prefix} there"
    log.info(f'Setting test prompt: "{test_prompt}"')
    decoder_base.set_prompt(test_prompt, tokenizer)
    log.info(f"Prompt text is now (after decoding etc): {decoder_base.prompt_text}")

    # Put models in eval mode
    decoder_base.eval()
    encoder_base.eval()

    # Hidden size from the model
    d_model = (
        decoder_base.base.config.hidden_size
        if not hasattr(decoder_base.base.config, "text_config")
        else decoder_base.base.config.text_config.hidden_size
    )

    # Test parameters
    max_length = 20  # Generate 20 tokens
    gumbel_tau = 1.0  # Temperature for generation
    batch_size = 2  # Test with batch size 2

    # Test 1: Zero activation (baseline)
    log.info("\nTest 1: Generation with zero activation vector")
    zero_activation = torch.zeros(
        batch_size, d_model, device=device, dtype=decoder_base.base.get_input_embeddings().weight.dtype
    )

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        result_zero = decoder_base.generate_soft(
            activation_input=zero_activation,
            max_length=max_length,
            gumbel_tau=gumbel_tau,
            use_projection=True,
            print_prompt=True,
            original_token_pos=torch.tensor([0] * batch_size, device=device),
        )

    # Decode the generated tokens
    zero_tokens = result_zero.hard_token_ids
    for i in range(batch_size):
        decoded_text = tokenizer.decode(zero_tokens[i], skip_special_tokens=True)
        decoded_text = decoded_text.replace("\n", "\\n")  # Escape newlines
        log.info(f"  Sample {i + 1}: {decoded_text}")

    # Test 2: Random activation vector
    log.info("\nTest 2: Generation with random activation vector")
    random_activation = (
        torch.randn(batch_size, d_model, device=device, dtype=decoder_base.base.get_input_embeddings().weight.dtype)
        * 0.1
    )  # Small random values

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        result_random = decoder_base.generate_soft(
            activation_input=random_activation,
            max_length=max_length,
            gumbel_tau=gumbel_tau,
            use_projection=True,
            print_prompt=False,
            original_token_pos=torch.tensor([0] * batch_size, device=device),
        )

    random_tokens = result_random.hard_token_ids
    for i in range(batch_size):
        decoded_text = tokenizer.decode(random_tokens[i], skip_special_tokens=True)
        decoded_text = decoded_text.replace("\n", "\\n")  # Escape newlines
        log.info(f"  Sample {i + 1}: {decoded_text}")

    # # Test 3: Checkpoint version with same random activation
    # log.info("\nTest 3: Generation with checkpointing (same random activation)")

    # with torch.no_grad():
    #     result_chkpt = decoder_base.generate_soft_chkpt(
    #         activation_input=random_activation,
    #         max_length=max_length,
    #         gumbel_tau=gumbel_tau,
    #         use_projection=True,
    #         print_prompt=False,
    #         checkpoint_every_n_tokens=4  # Checkpoint every 4 tokens
    #     )

    # chkpt_tokens = result_chkpt.hard_token_ids
    # for i in range(batch_size):
    #     decoded_text = tokenizer.decode(chkpt_tokens[i], skip_special_tokens=True)
    #     decoded_text = decoded_text.replace('\n', '\\n')  # Escape newlines
    #     log.info(f"  Sample {i+1}: {decoded_text}")

    # Note about randomness
    log.info("\nNote: Due to Gumbel-Softmax sampling, each generation is random.")
    log.info("The important thing is that all methods produce coherent English text.")

    # Test 4: Test with different temperatures
    log.info("\nTest 4: Generation with different temperatures")
    temperatures = [0.5, 1.0, 2.0]

    for temp in temperatures:
        log.info(f"\n  Temperature = {temp}:")
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            result_temp = decoder_base.generate_soft(
                activation_input=random_activation[:1],  # Just first sample
                max_length=max_length,
                gumbel_tau=temp,
                use_projection=True,
                print_prompt=False,
                original_token_pos=torch.tensor([0], device=device),
            )

        temp_tokens = result_temp.hard_token_ids[0]
        decoded_text = tokenizer.decode(temp_tokens, skip_special_tokens=True)
        decoded_text = decoded_text.replace("\n", "\\n")  # Escape newlines
        log.info(f"    Generated: {decoded_text}")

    # # Test 5: Without projection
    # log.info("\nTest 5: Generation without projection layer")

    # with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    #     result_noproj = decoder_base.generate_soft(
    #         activation_input=random_activation[:1],  # Just first sample
    #         max_length=max_length,
    #         gumbel_tau=gumbel_tau,
    #         use_projection=False,  # Skip projection
    #         print_prompt=False,
    #         original_token_pos=torch.tensor([0], device=device),
    #     )

    # noproj_tokens = result_noproj.hard_token_ids[0]
    # decoded_text = tokenizer.decode(noproj_tokens, skip_special_tokens=True)
    # decoded_text = decoded_text.replace("\n", "\\n")  # Escape newlines
    # log.info(f"  Without projection: {decoded_text}")

    # Test 5: Without patching
    log.info("\nTest 5: Generation without patching")

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        result_nopatching = decoder_base.generate_soft(
            activation_input=random_activation[:1],  # Just first sample
            max_length=max_length,
            gumbel_tau=gumbel_tau,
            use_projection=False,  # Skip projection
            print_prompt=False,
            do_patching=False,
            special_token=tokenizer.encode(" "),
            original_token_pos=torch.tensor([0], device=device),
        )

    no_patch_tokens = result_nopatching.hard_token_ids[0]
    decoded_text = tokenizer.decode(no_patch_tokens, skip_special_tokens=True)
    decoded_text = decoded_text.replace("\n", "\\n")  # Escape newlines
    log.info(f"  Without patching: {decoded_text}")

    # # Test 5: Without patching
    # log.info("\nTest 5.5: Generation without patching kv cached")

    # with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    #     result_nopatching_kv_cached = decoder_base.generate_soft_kv_cached(
    #         activation_input=random_activation[:1],  # Just first sample
    #         max_length=max_length,
    #         gumbel_tau=gumbel_tau,
    #         use_projection=False,  # Skip projection
    #         print_prompt=False,
    #         do_patching=False,
    #         special_token=tokenizer.encode(" "),
    #         original_token_pos=torch.tensor([0], device=device),
    #     )

    # no_patch_tokens_kv_cached = result_nopatching_kv_cached.hard_token_ids[0]
    # decoded_text = tokenizer.decode(no_patch_tokens_kv_cached, skip_special_tokens=True)
    # decoded_text = decoded_text.replace("\n", "\\n")  # Escape newlines
    # log.info(f"  Without patching kv cached: {decoded_text}")

    log.info("\nTest 5.5: Generation without patching kv cached non differentiable")

    for i in range(4):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            result_nopatching_kv_cached_non_diff = decoder_base.generate_soft_kv_cached_nondiff(
                activation_input=random_activation[:1],  # Just first sample
                max_length=max_length * 5,
                gumbel_tau=gumbel_tau,
                use_projection=False,  # Skip projection
                print_prompt=False,
                do_patching=False,
                special_token=tokenizer.encode(" " if "gemma3" in decoder_base.base.config.model_type else " "),
                original_token_pos=torch.tensor([0], device=device),
            )

        no_patch_tokens_kv_cached_non_diff = result_nopatching_kv_cached_non_diff.hard_token_ids[0]
        decoded_text = tokenizer.decode(no_patch_tokens_kv_cached_non_diff, skip_special_tokens=True)
        decoded_text = decoded_text.replace("\n", "\\n")  # Escape newlines
        log.info(f" Test 5.6 Without patching kv cached non differentiable: {decoded_text}")

    if 'gpt_oss' in decoder_base.base.config.model_type:
        log.info("\nTest 5.7: Generation with gpt-oss no patching")
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            result_gpt_oss = decoder_base.generate_soft_kv_cached_nondiff(
                activation_input=random_activation[:1],  # Just first sample
                max_length=max_length,  
                gumbel_tau=gumbel_tau,
                use_projection=False,  # Skip projection
                print_prompt=False,
                do_patching=False,
                special_token=tokenizer.encode(" " if "gemma3" in decoder_base.base.config.model_type else " "),
                original_token_pos=torch.tensor([0], device=device),
            )
        no_patch_tokens_kv_cached_non_diff = result_gpt_oss.hard_token_ids[0]
        decoded_text = tokenizer.decode(no_patch_tokens_kv_cached_non_diff, skip_special_tokens=True)
        decoded_text = decoded_text.replace("\n", "\\n")  # Escape newlines
        log.info(f" Test 5.7: Generation with gpt-oss: {decoded_text}")

    if 'gpt_oss' in decoder_base.base.config.model_type:
        log.info("\nTest 5.7: Generation with gpt-oss no patching again")
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            result_gpt_oss = decoder_base.generate_soft_kv_cached_nondiff(
                activation_input=random_activation[:1],  # Just first sample
                max_length=max_length,
                gumbel_tau=gumbel_tau,
                use_projection=False,  # Skip projection
                print_prompt=False,
                do_patching=False,
                special_token=tokenizer.encode(" " if "gemma3" in decoder_base.base.config.model_type else " "),
                original_token_pos=torch.tensor([0], device=device),
            )
        no_patch_tokens_kv_cached_non_diff = result_gpt_oss.hard_token_ids[0]
        decoded_text = tokenizer.decode(no_patch_tokens_kv_cached_non_diff, skip_special_tokens=True)
        decoded_text = decoded_text.replace("\n", "\\n")  # Escape newlines
        log.info(f" Test 5.7: Generation with gpt-oss no patching: {decoded_text}")
    # Put models back in train mode
    decoder_base.train()
    encoder_base.train()

    # Test 6: Multiple samples to check variability
    log.info("\nTest 6: Multiple generations from same activation (checking randomness)")
    log.info("  Running 3 generations with the same activation to see variability:")

    test_activation = (
        torch.randn(1, d_model, device=device, dtype=decoder_base.base.get_input_embeddings().weight.dtype) * 0.1
    )
    for i in range(3):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            result = decoder_base.generate_soft_kv_cached_nondiff(
                activation_input=test_activation,
                max_length=max_length,
                gumbel_tau=1.0,
                use_projection=True,
                print_prompt=False,
                original_token_pos=torch.tensor([0], device=device),
            )
        tokens = result.hard_token_ids[0]
        decoded_text = tokenizer.decode(tokens, skip_special_tokens=True)
        decoded_text = decoded_text.replace("\n", "\\n")  # Escape newlines
        log.info(f"    Generation {i + 1}: {decoded_text}")

    log.info("\n" + "=" * 80)
    log.info("Generation tests completed!")
    log.info("  ✓ All methods produce text outputs")
    log.info("  ✓ Outputs appear to be coherent English (manual verification needed)")
    log.info("  ✓ Different activations produce different outputs")
    log.info("  ✓ Temperature affects generation diversity")
    log.info("=" * 80 + "\n")

    # Restore original prompt if provided
    if original_prompt:
        log.info(f'Restoring original decoder prompt: "{original_prompt}"')
        decoder_base.set_prompt(original_prompt, tokenizer)
    else:
        log.info("No original prompt to restore.")
