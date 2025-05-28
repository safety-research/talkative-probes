import torch
from typing import Dict, Any, List
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
            results["fresh_replaced_top5_probs"] = torch.softmax(fresh_replaced_logits_at_p, dim=-1)[fresh_replaced_top5.indices].tolist()
    
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
        
        print(f"\n1. Natural forward pass top 5:")
        for i, (token, prob) in enumerate(zip(results["natural_top5_tokens"], results["natural_top5_probs"])):
            print(f"   {i+1}. '{token}' (p={prob:.4f})")
        
        print(f"\n2. With saved activation top 5:")
        for i, (token, prob) in enumerate(zip(results["replaced_top5_tokens"], results["replaced_top5_probs"])):
            print(f"   {i+1}. '{token}' (p={prob:.4f})")
        
        if "fresh_replaced_top5_tokens" in results:
            print(f"\n3. With fresh activation top 5:")
            for i, (token, prob) in enumerate(zip(results["fresh_replaced_top5_tokens"], results["fresh_replaced_top5_probs"])):
                print(f"   {i+1}. '{token}' (p={prob:.4f})")
        
        print(f"\n4. With dtype-converted saved activation top 5:")
        for i, token in enumerate(results["converted_dtype_top5_tokens"]):
            print(f"   {i+1}. '{token}'")
        
        if "activation_comparison" in results:
            print(f"\nActivation comparison:")
            for k, v in results["activation_comparison"].items():
                print(f"   {k}: {v:.6f}")
        
        # Check if predictions match
        natural_matches_saved = results["natural_top5_tokens"] == results["replaced_top5_tokens"]
        natural_matches_fresh = results.get("fresh_replaced_top5_tokens") == results["natural_top5_tokens"]
        
        print(f"\n=== Summary ===")
        print(f"Natural matches saved activation: {natural_matches_saved}")
        if "fresh_replaced_top5_tokens" in results:
            print(f"Natural matches fresh activation: {natural_matches_fresh}")
        print(f"Saved matches converted dtype: {results['replaced_top5_tokens'] == results['converted_dtype_top5_tokens']}")
        #if not (natural_matches_saved and natural_matches_fresh):
            #raise ValueError("Natural matches saved activation and fresh activation do not match")
    
    print(f"\nDebug - Comparing same position?")
    print(f"Sample index: {sample_idx}")
    print(f"Layer from batch: {l}")
    print(f"Position from batch: {p}")
    print(f"Input IDs shape: {input_ids_seq.shape}")
    print(f"First 10 tokens: {input_ids_seq[0, :10].tolist()}")
    
    print(f"\nSaved activation info:")
    print(f"Layer from saved: {batch['layer_idx'][sample_idx].item()}")
    print(f"Position from saved: {batch['token_pos_A'][sample_idx].item()}")
    print(f"Saved input IDs first 10: {batch['input_ids_A'][sample_idx, :10].tolist()}")
    
    return results


# Usage example:
# results = diagnose_activation_mismatch(batch, orig, tok, device, sample_idx=0)

def diagnose_activation_save_load(orig, input_ids, layer_idx, token_pos, device):
    """Test the full save/load cycle for activations."""
    import tempfile
    import os
    
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
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
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
        loaded_cpu = torch.load(tmp_path, map_location='cpu')
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
        loaded_dict = torch.load(tmp_path, map_location='cpu')
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
    import torch
    from pathlib import Path
    
    # Load a sample directly
    dataset_root = Path(dataset_path)
    
    # Check if using sharded format
    pt_files = list(dataset_root.glob("*.pt"))
    if not pt_files:
        pt_files = list(dataset_root.glob("*/*.pt"))  # Check subdirectories
    
    if pt_files:
        sample_file = pt_files[0]
        data = torch.load(sample_file, map_location='cpu')
        
        if isinstance(data, list):
            sample = data[0]
        else:
            sample = data
        
        print(f"Dataset format check:")
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
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _ = orig.model(input_ids=input_ids)
    handle.remove()
    
    print(f"\nAutocast comparison:")
    print(f"No autocast - norm: {activation_no_autocast.norm().item():.6f}, dtype: {activation_no_autocast.dtype}")
    print(f"With autocast - norm: {activation_with_autocast.norm().item():.6f}, dtype: {activation_with_autocast.dtype}")
    print(f"Relative diff: {((activation_no_autocast - activation_with_autocast).abs() / (activation_no_autocast.abs() + 1e-8)).mean().item():.6f}")
    
    return activation_no_autocast, activation_with_autocast

# Add this diagnostic
def check_layer_indexing(orig, input_ids, device):
    """Check how layers are indexed in hidden_states."""
    with torch.no_grad():
        outputs = orig.model(input_ids=input_ids, output_hidden_states=True)
    
    print(f"\nLayer indexing check:")
    print(f"Number of hidden states: {len(outputs.hidden_states)}")
    print(f"Model config num_hidden_layers: {orig.model.config.num_hidden_layers}")
    
    # Check dimensions
    for i, hidden in enumerate(outputs.hidden_states[:5]):  # First 5 layers
        print(f"  hidden_states[{i}] shape: {hidden.shape}, norm: {hidden.norm().item():.4f}")
    
    # Also check what the dumper would extract
    if hasattr(orig.model, 'model'):  # LLaMA style
        print(f"Model type appears to be LLaMA-style (has .model attribute)")
    elif hasattr(orig.model, 'transformer'):  # GPT style
        print(f"Model type appears to be GPT-style (has .transformer attribute)")
