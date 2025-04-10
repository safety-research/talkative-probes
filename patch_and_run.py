import torch
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
    #print(f"{model.config.model_type} has {num_layers} layers")
    return num_layers, model_lmhead

def extract_activations_for_example(context, model, target_token_idx):
    activations = {}
    num_layers, _ = get_model_info(model)
    with model.trace(context, remote=USE_REMOTE_EXECUTION, scan=False, validate=False) as tracer:
        for layer_idx in range(num_layers):
            layer_path = get_layer_output_path(layer_idx, model)
            module = util.fetch_attr(model, layer_path)
            proxy = module.output
            if isinstance(proxy, Proxy):
                act = proxy[0][0, target_token_idx, :].save()
            else:
                act = proxy[0][0, target_token_idx, :].save()
            activations[layer_path] = act
    return activations

def extract_activations_batch(contexts, model, target_token_idx):
    activations_batch = [{} for _ in range(len(contexts))]  
    num_layers, _ = get_model_info(model)
    with model.trace(remote=USE_REMOTE_EXECUTION, scan=False, validate=False) as tracer:
        for b, context in enumerate(contexts):
            with tracer.invoke(context):
                for layer_idx in range(num_layers):
                    layer_path = get_layer_output_path(layer_idx, model)
                    module = util.fetch_attr(model, layer_path)
                    proxy = module.output
                    if isinstance(proxy, Proxy):
                        act = proxy[0][0, target_token_idx, :].save()
                    else:
                        act = proxy[0][0, target_token_idx, :].save()
                    activations_batch[b][layer_path] = act
    return activations_batch

def run_patching(patching_input, activations, model, filler_token_idx, topkn=100):
    num_layers, model_lmhead = get_model_info(model)
    with model.trace(patching_input, remote=USE_REMOTE_EXECUTION) as tracer:
        for layer_path, act in activations.items():
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

def run_patching_batch(patching_inputs, activations_batch, model, filler_token_idx, topkn=100):
    num_layers, model_lmhead = get_model_info(model)
    batch_size = len(patching_inputs)
    with model.trace(remote=USE_REMOTE_EXECUTION) as tracer:
        for b in range(batch_size):
            with tracer.invoke(patching_inputs[b]):
                for layer_path, act in activations_batch[b].items():
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