import torch
import nnsight
from nnsight import LanguageModel
from nnsight import CONFIG, util
from nnsight.tracing.graph import Proxy # For 
USE_REMOTE_EXECUTION = False



def get_layer_output_path(layer_idx, model):
    if model.config.model_type == "gpt2":
        return f"transformer.h.{layer_idx}"
    elif model.config.model_type == "llama":
        return f"layers.{layer_idx}"
    elif model.config.model_type == "gemma3":
        return f"language_model.model.layers.{layer_idx}"
    elif model.config.model_type == "gemma":
        return f"model.layers.{layer_idx}"
    else:
        return f"model.layers.{layer_idx}"

def get_model_info(model):
    if model.config.model_type == "gpt2":
        model_lmhead = model.lm_head
        num_layers = len(model.transformer.h)
    elif model.config.model_type == "llama":
        model_lmhead = model.lm_head
        num_layers = len(model.layers)
    elif model.config.model_type == "gemma3":
        model_lmhead = model.language_model.lm_head
        num_layers = model.config.text_config.num_hidden_layers
    elif model.config.model_type == "gemma":
        model_lmhead = model.lm_head
        num_layers = model.config.num_hidden_layers
    else:
        model_lmhead = model.lm_head
        num_layers = model.config.num_hidden_layers
    #print(f"{model.config.model_type} has {num_layers} layers")
    return num_layers, model_lmhead

def extract_activations_for_example(context, model, target_token_idx, layers_to_patch=None):
    activations = {}
    num_layers, _ = get_model_info(model)
    with model.trace(context, remote=USE_REMOTE_EXECUTION, scan=False, validate=False) as tracer:
        for layer_idx in range(num_layers):
            if layers_to_patch is not None and layer_idx not in layers_to_patch:
                continue
            layer_path = get_layer_output_path(layer_idx, model)
            module = util.fetch_attr(model, layer_path)
            proxy = module.output
            if isinstance(proxy, Proxy):
                act = proxy[0][0, target_token_idx, :].save()
            else:
                act = proxy[0][0, target_token_idx, :].save()
            activations[layer_path] = act
    activations = {k:v.detach().cpu() for k,v in activations.items()}
    return activations

def extract_activations_batch(contexts, model, target_token_idx, layers_to_patch=None):
    activations_batch = [{} for _ in range(len(contexts))]  
    num_layers, _ = get_model_info(model)
    with model.trace(remote=USE_REMOTE_EXECUTION, scan=False, validate=False) as tracer:
        for b, context in enumerate(contexts):
            with tracer.invoke(context):
                for layer_idx in range(num_layers):
                    if layers_to_patch is not None and layer_idx not in layers_to_patch:
                        continue
                    layer_path = get_layer_output_path(layer_idx, model)
                    module = util.fetch_attr(model, layer_path)
                    proxy = module.output
                    if isinstance(proxy, Proxy):
                        act = proxy[0][0, target_token_idx, :].save()
                    else:
                        act = proxy[0][0, target_token_idx, :].save()
                    activations_batch[b][layer_path] = act
    activations_batch = [{k:v.detach().cpu() for k,v in a.items()} for a in activations_batch]
    return activations_batch

def run_patching(patching_input, activations, model, filler_token_idx, topkn=100, layers_to_patch=None):
    num_layers, model_lmhead = get_model_info(model)
    with model.trace(patching_input, remote=USE_REMOTE_EXECUTION) as tracer:
        for layer_path, act in activations.items():
            if layers_to_patch is not None:
                try:
                    layer_idx = int(layer_path.split('.')[-1])
                except ValueError:
                    continue
                if layer_idx not in layers_to_patch:
                    continue
            module = util.fetch_attr(model, layer_path)
            module.output[0][0, filler_token_idx, :] = act
        logits = model_lmhead.output[0, -1, :]
        patched_pred = torch.argmax(logits, dim=-1).save()
        sorted_idx = torch.argsort(logits, descending=True)
        topk_idx = sorted_idx[:topkn]
        topk_logits = logits[topk_idx]
        topk_logprobs = torch.log_softmax(logits, dim=-1)[topk_idx]
        patched_result = (patched_pred, topk_idx.save(), topk_logits.save(), topk_logprobs.save())
    patched_result = tuple(p.detach().cpu() for p in patched_result)
    return patched_result

def run_patching_batch(patching_inputs, activations_batch, model, filler_token_idx, topkn=100, layers_to_patch=None):
    num_layers, model_lmhead = get_model_info(model)
    batch_size = len(patching_inputs)
    with model.trace(remote=USE_REMOTE_EXECUTION) as tracer:
        for b in range(batch_size):
            with tracer.invoke(patching_inputs[b]):
                for layer_path, act in activations_batch[b].items():
                    if layers_to_patch is not None:
                        try:
                            layer_idx = int(layer_path.split('.')[-1])
                        except ValueError:
                            continue
                        if layer_idx not in layers_to_patch:
                            continue
                    module = util.fetch_attr(model, layer_path)
                    module.output[0][0, filler_token_idx, :] = act
                logits = model_lmhead.output[0, -1, :]
                pred = torch.argmax(logits, dim=-1).save()
                sorted_idx = torch.argsort(logits, descending=True)
                topk_idx = sorted_idx[:topkn]
                topk_logits = logits[topk_idx]
                topk_logprobs = torch.log_softmax(logits, dim=-1)[topk_idx]
                if b == 0:
                    preds, topk_idxs, topk_logitss, topk_logprobss = [], [], [], []
                preds.append(pred)
                topk_idxs.append(topk_idx.save())
                topk_logitss.append(topk_logits.save())
                topk_logprobss.append(topk_logprobs.save())
    preds = [p.detach().cpu() for p in preds]
    topk_idxs = [p.detach().cpu() for p in topk_idxs]
    topk_logitss = [p.detach().cpu() for p in topk_logitss]
    topk_logprobss = [p.detach().cpu() for p in topk_logprobss]
    return preds, topk_idxs, topk_logitss, topk_logprobss

def run_baseline(baseline_input, model, topkn=100):
    num_layers, model_lmhead = get_model_info(model)
    with model.trace(baseline_input, remote=USE_REMOTE_EXECUTION) as tracer:
        logits = model_lmhead.output[0, -1, :]
        baseline_pred = torch.argmax(logits, dim=-1).save()
        sorted_idx = torch.argsort(logits, descending=True)
        topk_idx = sorted_idx[:topkn]
        topk_logits = logits[topk_idx]
        topk_logprobs = torch.log_softmax(logits, dim=-1)[topk_idx]
        baseline_result = (baseline_pred, topk_idx.save(), topk_logits.save(), topk_logprobs.save())
    baseline_result = tuple(p.detach().cpu() for p in baseline_result)
    return baseline_result

def run_baseline_batch(baseline_inputs, model, topkn=100):
    batch_size = len(baseline_inputs)
    _, model_lmhead = get_model_info(model)
    with model.trace(remote=USE_REMOTE_EXECUTION) as tracer:
        for b in range(batch_size):
            with tracer.invoke(baseline_inputs[b]):
                logits = model_lmhead.output[0, -1, :]
                pred = torch.argmax(logits, dim=-1).save()
                sorted_idx = torch.argsort(logits, descending=True)
                topk_idx = sorted_idx[:topkn]
                topk_logits = logits[topk_idx]
                topk_logprobs = torch.log_softmax(logits, dim=-1)[topk_idx]
                if b == 0:
                    preds, topk_idxs, topk_logitss, topk_logprobss = [], [], [], []
                preds.append(pred)
                topk_idxs.append(topk_idx.save())
                topk_logitss.append(topk_logits.save())
                topk_logprobss.append(topk_logprobs.save())
    preds = [p.detach().cpu() for p in preds]
    topk_idxs = [p.detach().cpu() for p in topk_idxs]
    topk_logitss = [p.detach().cpu() for p in topk_logitss]
    topk_logprobss = [p.detach().cpu() for p in topk_logprobss]
    return preds, topk_idxs, topk_logitss, topk_logprobss

# New functions for multi-token prediction

def run_multi_token_patching(patching_input, activations, model, filler_token_idx, max_new_tokens=1, topkn=100, layers_to_patch=None, temperature = None):
    """
    Run patching with multi-token prediction using model.generate
    
    Args:
        patching_input: Input text for patching
        activations: Dictionary of activations to patch
        model: Model to patch
        filler_token_idx: Index of token to replace with activation
        max_new_tokens: Number of new tokens to generate
        topkn: Number of top predictions to return
        layers_to_patch: Optional list of layer indices to patch
        
    Returns:
        Generated tokens, predictions, and logprobs for each new token
    """
    num_layers, model_lmhead = get_model_info(model)
    
    # Create a dictionary to map layer path to module
    layer_modules = {}
    for layer_idx in range(num_layers):
        if layers_to_patch is not None and layer_idx not in layers_to_patch:
            continue
        layer_path = get_layer_output_path(layer_idx, model)
        layer_modules[layer_path] = util.fetch_attr(model, layer_path)
    

    
    with model.generate(patching_input, max_new_tokens=max_new_tokens, remote=USE_REMOTE_EXECUTION, temperature=temperature) as tracer:
        # Create empty lists to store results
        all_tokens = []
        all_preds = nnsight.list().save()
        all_topk_idx = nnsight.list().save()
        all_topk_logits = nnsight.list().save()
        all_topk_logprobs = nnsight.list().save()
        # First apply patching at the initial position
        for layer_path, act in activations.items():
            if layers_to_patch is not None:
                try:
                    layer_idx = int(layer_path.split('.')[-1])
                except ValueError:
                    continue
                if layer_idx not in layers_to_patch:
                    continue
            module = layer_modules.get(layer_path)
            #if module:
            module.output[0][0, filler_token_idx, :] = act
        
        # Get generated tokens
        all_tokens = model.generator.output.save()
        
        # Use .all() to apply to all new token predictions
        with model_lmhead.all():
            logits = model_lmhead.output[0, -1, :]
            pred = torch.argmax(logits, dim=-1)
            sorted_idx = torch.argsort(logits, descending=True)
            topk_idx = sorted_idx[:topkn]
            topk_logits = logits[topk_idx]
            topk_logprobs = torch.log_softmax(logits, dim=-1)[topk_idx]
            
            all_preds.append(pred)
            all_topk_idx.append(topk_idx)
            all_topk_logits.append(topk_logits)
            all_topk_logprobs.append(topk_logprobs)
    
    # Convert to CPU tensors
    all_tokens = all_tokens.detach().cpu()
    all_preds = [p.detach().cpu() for p in all_preds]
    all_topk_idx = [idx.detach().cpu() for idx in all_topk_idx]
    all_topk_logits = [logit.detach().cpu() for logit in all_topk_logits]
    all_topk_logprobs = [logprob.detach().cpu() for logprob in all_topk_logprobs]
    
    return all_tokens, all_preds, all_topk_idx, all_topk_logits, all_topk_logprobs

# def run_multi_token_patching_with_all(patching_input, activations, model, filler_token_idx, max_new_tokens=1, topkn=100, layers_to_patch=None):
#     """
#     Run patching with multi-token prediction using model.generate and .all() method
#     to apply patching to all generated tokens
    
#     Args:
#         patching_input: Input text for patching
#         activations: Dictionary of activations to patch
#         model: Model to patch
#         filler_token_idx: Index of token to replace with activation
#         max_new_tokens: Number of new tokens to generate
#         topkn: Number of top predictions to return
#         layers_to_patch: Optional list of layer indices to patch
        
#     Returns:
#         Generated tokens, predictions, and logprobs for each new token
#     """
#     num_layers, model_lmhead = get_model_info(model)
    
#     # Get layer modules
#     layer_modules = {}
#     for layer_idx in range(num_layers):
#         if layers_to_patch is not None and layer_idx not in layers_to_patch:
#             continue
#         layer_path = get_layer_output_path(layer_idx, model)
#         layer_modules[layer_path] = util.fetch_attr(model, layer_path)
    

#     with model.generate(patching_input, max_new_tokens=max_new_tokens, remote=USE_REMOTE_EXECUTION) as tracer:
#             # Create empty lists to store results
#         all_preds = nnsight.list().save()
#         all_topk_idx = nnsight.list().save()
#         all_topk_logits = nnsight.list().save()
#         all_topk_logprobs = nnsight.list().save()

#         # Get all layer modules to apply .all() to
#         modules_to_patch = []
#         for layer_path, module in layer_modules.items():
#             if layers_to_patch is not None:
#                 try:
#                     layer_idx = int(layer_path.split('.')[-1])
#                 except ValueError:
#                     continue
#                 if layer_idx not in layers_to_patch:
#                     continue
#             modules_to_patch.append(module)
        
#         # Apply patching for each layer with .all()
#         for module in modules_to_patch:
#             with module.all():
#                 module.output[0][0, filler_token_idx, :] = activations[layer_path]
        
#         # Get generated tokens
#         all_tokens = model.generator.output.save()
        
#         # Get predictions for each token
#         with model_lmhead.all():
#             logits = model_lmhead.output[0, -1, :]
#             pred = torch.argmax(logits, dim=-1)
#             sorted_idx = torch.argsort(logits, descending=True)
#             topk_idx = sorted_idx[:topkn]
#             topk_logits = logits[topk_idx]
#             topk_logprobs = torch.log_softmax(logits, dim=-1)[topk_idx]
            
#             all_preds.append(pred)
#             all_topk_idx.append(topk_idx)
#             all_topk_logits.append(topk_logits)
#             all_topk_logprobs.append(topk_logprobs)
    
#     # Convert to CPU tensors
#     all_tokens = all_tokens.detach().cpu()
#     all_preds = [p.detach().cpu() for p in all_preds]
#     all_topk_idx = [idx.detach().cpu() for idx in all_topk_idx]
#     all_topk_logits = [logit.detach().cpu() for logit in all_topk_logits]
#     all_topk_logprobs = [logprob.detach().cpu() for logprob in all_topk_logprobs]
    
#     return all_tokens, all_preds, all_topk_idx, all_topk_logits, all_topk_logprobs

def run_multi_token_baseline(baseline_input, model, max_new_tokens=1, topkn=100, temperature = None):
    """
    Run baseline with multi-token prediction using model.generate
    
    Args:
        baseline_input: Input text for baseline
        model: Model to run
        max_new_tokens: Number of new tokens to generate
        topkn: Number of top predictions to return
        
    Returns:
        Generated tokens, predictions, and logprobs for each new token
    """
    _, model_lmhead = get_model_info(model)

    
    with model.generate(baseline_input, max_new_tokens=max_new_tokens, remote=USE_REMOTE_EXECUTION, temperature=temperature) as tracer:
            
        # Create empty lists to store results
        all_preds = nnsight.list().save()
        all_topk_idx = nnsight.list().save()
        all_topk_logits = nnsight.list().save()
        all_topk_logprobs = nnsight.list().save()
        # Get generated tokens
        all_tokens = model.generator.output.save()
        
        # Get predictions for each token
        with model_lmhead.all():
            logits = model_lmhead.output[0, -1, :]
            pred = torch.argmax(logits, dim=-1)
            sorted_idx = torch.argsort(logits, descending=True)
            topk_idx = sorted_idx[:topkn]
            topk_logits = logits[topk_idx]
            topk_logprobs = torch.log_softmax(logits, dim=-1)[topk_idx]
            
            all_preds.append(pred)
            all_topk_idx.append(topk_idx)
            all_topk_logits.append(topk_logits)
            all_topk_logprobs.append(topk_logprobs)
    
    # Convert to CPU tensors
    all_tokens = all_tokens.detach().cpu()
    all_preds = [p.detach().cpu() for p in all_preds]
    all_topk_idx = [idx.detach().cpu() for idx in all_topk_idx]
    all_topk_logits = [logit.detach().cpu() for logit in all_topk_logits]
    all_topk_logprobs = [logprob.detach().cpu() for logprob in all_topk_logprobs]
    
    return all_tokens, all_preds, all_topk_idx, all_topk_logits, all_topk_logprobs