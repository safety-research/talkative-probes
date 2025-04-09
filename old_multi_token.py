
# # %%
# print(unpatched_results[0])
# # %%
# for i in tqdm(range(len(dataset['original_contexts'])), desc="Running Patching/Baseline"):
#     free_unused_cuda_memory()
#     # Get single example data
#     patching_input = patching_inputs[i]
#     baseline_input = baseline_inputs[i]
#     if i == 0:
#         print("patching_input: '", patching_input, "'")
#         print("baseline_input: '", baseline_input, "'")
#         print("check patching: ", tokenizer.decode(tokenizer.encode(patching_input)))
#         print("check baseline: ", tokenizer.decode(tokenizer.encode(baseline_input)))
#         print(f"donor token: ' {tokenizer.decode(tokenizer.encode(dataset['original_contexts'][i])[TARGET_TOKEN_IDX])}' at position {TARGET_TOKEN_IDX} in the original context")
#         print(f"recipient token: '{tokenizer.decode(filler_token_id)}' at position {filler_token_idx_in_patched_input} in the patching input")
    
#     #--- Patched Run ---
#     # Use separate trace calls for patched and baseline to avoid interleaving issues
#     with model.trace(patching_input, remote=USE_REMOTE_EXECUTION) as patched_tracer:
#         # Patch activations from all layers
#         for layer_path, activations in extracted_activations.items():
#             # Access the target module's output where patching should occur
#             module = util.fetch_attr(model, layer_path)
#             module_to_patch = module.output
            
#             # Perform the patch: Set the activation at the filler token position
#             module_to_patch[0][0, filler_token_idx_in_patched_input, :] = 1*activations[i]#*(-1000000)
        
#         # For single token generation (MAX_NEW_TOKENS = 1)
#         if MAX_NEW_TOKENS == 1:
#             # Get the final logits from the patched run
#             patched_logits = model_lmhead.output[0, -1, :]  # Logits for the last token
#             # Get the predicted token ID (argmax)
#             patched_prediction = torch.argmax(patched_logits, dim=-1).save()
#             patched_outputs.append([patched_prediction])
#             # Save indices, logits, and logprobs of the top 100 tokens
#             sorted_indices = torch.argsort(patched_logits, descending=True)
#             topk_indices = sorted_indices[:100]
#             topk_logits = patched_logits[topk_indices]
#             topk_logprobs = torch.log_softmax(patched_logits, dim=-1)[topk_indices]
#             patched_logits_all.append((topk_indices.save(), topk_logits.save(), topk_logprobs.save()))
#         else:
#             # For multi-token generation, we need to save the initial logits
#             # then continue generating additional tokens
            
#             # Get logits for last position
#             initial_logits = model_lmhead.output[0, -1, :].detach()
            
#             # Sample first token (using temperature and top_p)
#             if TEMPERATURE > 0:
#                 # Apply temperature
#                 initial_logits = initial_logits / TEMPERATURE
#                 # Apply top_p sampling
#                 sorted_logits, sorted_indices = torch.sort(initial_logits, descending=True)
#                 cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=0), dim=0)
#                 sorted_indices_to_remove = cumulative_probs > TOP_P
#                 sorted_indices_to_remove[0] = False  # Keep at least the highest prob token
#                 indices_to_remove = sorted_indices_to_remove.scatter(
#                     0, sorted_indices, sorted_indices_to_remove
#                 )
#                 initial_logits[indices_to_remove] = -float('inf')
#                 # Sample from the filtered distribution
#                 probs = torch.softmax(initial_logits, dim=0)
#                 first_token = torch.multinomial(probs, 1).item()
#             else:
#                 # Greedy sampling
#                 first_token = torch.argmax(initial_logits, dim=0).item()
            
#             # Initialize the sequence with the first token
#             generated_tokens = [first_token]
            
#             # Continue generating tokens one by one (autoregressive)
#             current_input = tokenizer.decode(tokenizer.encode(patching_input) + [first_token], skip_special_tokens=False)
            
#             # Generate additional tokens (MAX_NEW_TOKENS - 1)
#             for _ in range(MAX_NEW_TOKENS - 1):
#                 # We need to run a separate trace for each additional token
#                 with model.trace(current_input, remote=USE_REMOTE_EXECUTION) as next_token_tracer:
#                     # Get logits for the last token position
#                     next_logits = model_lmhead.output[0, -1, :]
                
#                 # Apply temperature and top_p sampling
#                 next_logits = next_logits.detach()
#                 if TEMPERATURE > 0:
#                     next_logits = next_logits / TEMPERATURE
#                     # Apply top_p sampling (same as above)
#                     sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
#                     cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=0), dim=0)
#                     sorted_indices_to_remove = cumulative_probs > TOP_P
#                     sorted_indices_to_remove[0] = False
#                     indices_to_remove = sorted_indices_to_remove.scatter(
#                         0, sorted_indices, sorted_indices_to_remove
#                     )
#                     next_logits[indices_to_remove] = -float('inf')
#                     probs = torch.softmax(next_logits, dim=0)
#                     next_token = torch.multinomial(probs, 1).item()
#                 else:
#                     next_token = torch.argmax(next_logits, dim=0).item()
                
#                 # Add the new token to the sequence
#                 generated_tokens.append(next_token)
                
#                 # Update input for next iteration
#                 current_input = tokenizer.decode(
#                     tokenizer.encode(patching_input) + generated_tokens, 
#                     skip_special_tokens=False
#                 )
            
#             patched_outputs.append(generated_tokens.save())
#         # --- Baseline Run ---
#     with model.trace(baseline_input, remote=USE_REMOTE_EXECUTION) as baseline_tracer:
#         # For single token generation
#         if MAX_NEW_TOKENS == 1:
#             # Get the final logits from the baseline run
#             baseline_logits = model_lmhead.output[0, -1, :]  # Logits for the last token
#             baseline_prediction = torch.argmax(baseline_logits, dim=-1).save()
#             baseline_outputs.append([baseline_prediction])
#             # Save indices, logits, and logprobs of the top 100 tokens
#             sorted_indices = torch.argsort(baseline_logits, descending=True)
#             topk_indices = sorted_indices[:100]
#             topk_logits = baseline_logits[topk_indices]
#             topk_logprobs = torch.log_softmax(baseline_logits, dim=-1)[topk_indices]
#             baseline_logits_all.append((topk_indices.save(), topk_logits.save(), topk_logprobs.save()))
#         else:
#             # For multi-token generation
#             # Using similar approach as with patched input
            
#             # Get logits for last position 
#             initial_logits = model_lmhead.output[0, -1, :].detach()
            
#             # Sample first token with temperature and top_p
#             if TEMPERATURE > 0:
#                 initial_logits = initial_logits / TEMPERATURE
#                 # Apply top_p sampling
#                 sorted_logits, sorted_indices = torch.sort(initial_logits, descending=True)
#                 cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=0), dim=0)
#                 sorted_indices_to_remove = cumulative_probs > TOP_P
#                 sorted_indices_to_remove[0] = False
#                 indices_to_remove = sorted_indices_to_remove.scatter(
#                     0, sorted_indices, sorted_indices_to_remove
#                 )
#                 initial_logits[indices_to_remove] = -float('inf')
#                 probs = torch.softmax(initial_logits, dim=0)
#                 first_token = torch.multinomial(probs, 1)
#             else:
#                 first_token = torch.argmax(initial_logits, dim=0)
            
#             # Initialize sequence with first token
#             generated_tokens = [first_token]
            
#             # Continue generating tokens
#             current_input = tokenizer.decode(tokenizer.encode(baseline_input) + [first_token], skip_special_tokens=False)
            
#             # Generate additional tokens
#             for _ in range(MAX_NEW_TOKENS - 1):
#                 with model.trace(current_input, remote=USE_REMOTE_EXECUTION) as next_token_tracer:
#                     next_logits = model_lmhead.output[0, -1, :]
                
#                 # Apply temperature and top_p sampling
#                 next_logits = next_logits.detach()
#                 if TEMPERATURE > 0:
#                     next_logits = next_logits / TEMPERATURE
#                     # Apply top_p sampling
#                     sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
#                     cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=0), dim=0)
#                     sorted_indices_to_remove = cumulative_probs > TOP_P
#                     sorted_indices_to_remove[0] = False
#                     indices_to_remove = sorted_indices_to_remove.scatter(
#                         0, sorted_indices, sorted_indices_to_remove
#                     )
#                     next_logits[indices_to_remove] = -float('inf')
#                     probs = torch.softmax(next_logits, dim=0)
#                     next_token = torch.multinomial(probs, 1)
#                 else:
#                     next_token = torch.argmax(next_logits, dim=0)
                
#                 # Add new token to sequence
#                 generated_tokens.append(next_token)
                
#                 # Update input for next iteration
#                 current_input = tokenizer.decode(
#                     tokenizer.encode(baseline_input) + generated_tokens, 
#                     skip_special_tokens=False
#                 )
            
#             baseline_outputs.append(generated_tokens)
            
#     # --- Unpatched Run (same input as patching but without patching) ---
#     with model.trace(unpatched_inputs[i], remote=USE_REMOTE_EXECUTION) as unpatched_tracer:
#         if MAX_NEW_TOKENS == 1:
#             unpatched_logits = model_lmhead.output[0, -1, :]
#             unpatched_prediction = torch.argmax(unpatched_logits, dim=-1)
#             unpatched_outputs.append([unpatched_prediction])
#             # Save indices, logits, and logprobs of the top 100 tokens
#             sorted_indices = torch.argsort(unpatched_logits, descending=True)
#             topk_indices = sorted_indices[:100]
#             topk_logits = unpatched_logits[topk_indices]
#             topk_logprobs = torch.log_softmax(unpatched_logits, dim=-1)[topk_indices]
#             unpatched_logits_all.append((topk_indices.save(), topk_logits.save(), topk_logprobs.save()))

# print("Patching and baseline runs complete.")
# print(f"Collected {len(patched_outputs)} patched outputs and {len(baseline_outputs)} baseline outputs.")

# %%
# len(patched_results)
# Start of Selection