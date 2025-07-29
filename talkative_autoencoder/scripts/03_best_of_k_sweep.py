#!/usr/bin/env python3
"""
Best-of-K sweep analysis for Talkative Autoencoders.
Evaluates how variance recovery improves with different K values in best-of-K sampling.
"""

import os
import json
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import scipy.stats
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer
import hashlib
import pickle

import hydra
from omegaconf import DictConfig, OmegaConf
# Note: DDP not needed for this script as we're using the analyzer directly

# Core lens imports
from lens.analysis.analyzer_class import LensAnalyzer
from lens.data.collate import collate
from lens.training.train_aux import _prepare_dataloaders
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.orig import OrigWrapper
from lens.training.distributed import (
    init_distributed,
    is_main,
    setup_for_distributed,
    set_seed
)
from lens.training.fast_distributed_sampler import FastDistributedSampler
# Note: We don't use wandb logging for evaluation scripts
import logging

def escape_newlines(text: str) -> str:
    return text.replace("\n", "\\n")


def compute_cache_key(model_name: str, layer: int, dataset_cfg: Dict[str, Any], num_positions: int) -> str:
    """Compute a deterministic cache key based on model and dataset configuration.
    
    Note: We don't include max_val_samples in the key, allowing us to reuse cached vectors
    even when requesting fewer samples than were cached.
    """
    # Extract relevant dataset parameters that affect vector extraction
    key_components = {
        'model_name': model_name,
        'layer': layer,
        'activation_dir': dataset_cfg.get('activation_dir', ''),
        'val_activation_dir': dataset_cfg.get('val_activation_dir', ''),
        'on_the_fly_enabled': dataset_cfg.get('on_the_fly', {}).get('enabled', False),
        'min_pos': dataset_cfg.get('on_the_fly', {}).get('min_pos', 5),
        'position_selection_strategy': dataset_cfg.get('on_the_fly', {}).get('position_selection_strategy', 'random'),
        'num_positions': num_positions,
    }
    
    # Create a stable string representation
    key_str = json.dumps(key_components, sort_keys=True)
    
    # Generate hash
    return hashlib.md5(key_str.encode()).hexdigest()


def get_cache_path(cache_dir: Path, cache_key: str, rank: int) -> Path:
    """Get the cache file path for a specific rank."""
    cache_path = cache_dir / cache_key / f"rank_{rank}_vectors.pkl"
    return cache_path


def save_cached_vectors(
    cache_path: Path,
    all_A: torch.Tensor,
    all_positions: torch.Tensor,
    all_token_ids: torch.Tensor,
    metadata: Dict[str, Any]
) -> None:
    """Save extracted vectors and metadata to cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    cache_data = {
        'all_A': all_A.cpu(),
        'all_positions': all_positions.cpu(),
        'all_token_ids': all_token_ids.cpu(),
        'metadata': metadata
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"Saved cache to {cache_path}")


def load_cached_vectors(cache_path: Path) -> Optional[Dict[str, Any]]:
    """Load cached vectors if they exist."""
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        print(f"Loaded cache from {cache_path}")
        return cache_data
    except Exception as e:
        print(f"Failed to load cache from {cache_path}: {e}")
        return None


class BestOfKEvaluator:
    """Evaluates best-of-K performance with variance recovery calculation."""
    
    def __init__(self, checkpoint_path: str, device: torch.device, use_bf16: bool = True, orig_model_name=None, orig_model_wrapper=None):
        self.device = device
        self.use_bf16 = use_bf16
        
        # Load analyzer with optional original model name for different activations
        if orig_model_name is not None:
            print('loading with shared and diff')
            # Pass the model name to avoid duplicate loading
            self.analyzer = LensAnalyzer(
                checkpoint_path=checkpoint_path,
                device=device,
                no_orig=True,
                use_bf16=use_bf16,
                different_activations_orig=orig_model_wrapper.model,
                shared_base_model=orig_model_wrapper,
                comparison_tl_checkpoint=False,
                
            )
        else:
            self.analyzer = LensAnalyzer(
                checkpoint_path=checkpoint_path,
                device=device,
                use_bf16=use_bf16,
                shared_base_model=orig_model_wrapper.model,
                comparison_tl_checkpoint=False,
            )
        
        # Extract components we need
        self.encoder = self.analyzer.encoder
        self.decoder = self.analyzer.decoder
        self.tokenizer = self.analyzer.tokenizer
        self.orig_model = self.analyzer.orig_model
        self.layer = self.analyzer.layer
        self.t_text = self.analyzer.t_text
        self.tau = self.analyzer.tau
        
        # Position selection parameters (defaults from on_the_fly_datasets.py)
        self.min_pos = 5  # Default min position
        self.position_selection_strategy = 'random'  # Default strategy
    
    def generate_best_of_k(
        self,
        A_targets: torch.Tensor,
        positions: torch.Tensor,
        k: int,
        temperature: float = 1.0,
        max_batch_size: int = 32,  # Maximum batch size for generation
        position_batch_size: int = 32,  # Batch size for position processing
        current_token_ids_all: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        debug_high_mse: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate best-of-K explanations and return reconstructions.
        
        Args:
            max_batch_size: Maximum batch size for processing k-expanded samples
            position_batch_size: Batch size for processing positions
        
        Returns:
            best_A_hat: Best reconstructed activations (batch_size, hidden_dim)
            best_mses: MSE for best reconstructions (batch_size,)
            best_tokens: Best explanation tokens (batch_size, t_text)
        """
        total_positions = A_targets.shape[0]
        all_best_A_hat = []
        all_best_mses = []
        all_best_tokens = []
        
        # Process positions in batches
        for start_idx in tqdm(range(0, total_positions, position_batch_size), desc=f"Generating best-of-{k}", leave=False):
            end_idx = min(start_idx + position_batch_size, total_positions)
            batch_A = A_targets[start_idx:end_idx]
            batch_positions = positions[start_idx:end_idx]
            current_batch_size = batch_A.shape[0]
            current_token_ids = current_token_ids_all[start_idx:end_idx]
            
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                # Expand for K samples per position
                A_expanded = batch_A.repeat_interleave(k, dim=0)
                positions_expanded = batch_positions.repeat_interleave(k)
                current_token_ids_expanded = current_token_ids.repeat_interleave(k)
                
                # Process the k-expanded samples in sub-batches
                expanded_size = A_expanded.shape[0]
                all_A_hat_for_batch = []
                all_tokens_for_batch = []
                all_decoded_texts = []
                
                for sub_start in range(0, expanded_size, max_batch_size):
                    sub_end = min(sub_start + max_batch_size, expanded_size)
                    
                    # Generate K explanations for this sub-batch
                    gen_result = self.decoder.generate_soft_kv_cached_nondiff(
                        A_expanded[sub_start:sub_end],
                        max_length=self.t_text,
                        gumbel_tau=self.tau,
                        original_token_pos=positions_expanded[sub_start:sub_end],
                        temperature=temperature,
                    )
                    
                    # Encode the generated explanations
                    A_hat_sub = self.encoder(
                        gen_result.generated_text_embeddings,
                        original_token_pos=positions_expanded[sub_start:sub_end],
                        current_token_ids=current_token_ids_expanded[sub_start:sub_end] if self.encoder.config.add_current_token else None
                    )
                    
                    all_A_hat_for_batch.append(A_hat_sub)
                    all_tokens_for_batch.append(gen_result.hard_token_ids)
                    
                    # Decode text for debugging
                    decoded_text = self.tokenizer.batch_decode(gen_result.hard_token_ids, skip_special_tokens=False)
                    all_decoded_texts.extend(decoded_text)
                
                # Concatenate all sub-batches
                A_hat_expanded = torch.cat(all_A_hat_for_batch, dim=0)
                explanation_tokens = torch.cat(all_tokens_for_batch, dim=0)
                
                # Calculate MSEs
                mses = torch.nn.functional.mse_loss(
                    A_expanded, A_hat_expanded, reduction='none'
                ).mean(dim=1)
                
                # Reshape to (batch_size, k)
                mses_per_pos = mses.view(current_batch_size, k)
                A_hat_per_pos = A_hat_expanded.view(current_batch_size, k, -1)
                tokens_per_pos = explanation_tokens.view(current_batch_size, k, -1)
                
                # Select best for each position
                best_indices = mses_per_pos.argmin(dim=1)
                
                # Gather best results
                batch_best_A_hat = A_hat_per_pos[torch.arange(current_batch_size), best_indices]
                batch_best_mses = mses_per_pos[torch.arange(current_batch_size), best_indices]
                batch_best_tokens = tokens_per_pos[torch.arange(current_batch_size), best_indices]
                
                # Debug high MSE values
                high_mse_mask = batch_best_mses > 200000
                if high_mse_mask.any() and debug_high_mse:
                    print(f"\n⚠️  Found {high_mse_mask.sum().item()} positions with MSE > 200000")
                    high_mse_indices = torch.where(high_mse_mask)[0]
                    
                    for idx in high_mse_indices[:5]:  # Show first 5 high MSE cases
                        pos_in_batch = idx.item()
                        actual_position = batch_positions[pos_in_batch].item()
                        mse_value = batch_best_mses[pos_in_batch].item()
                        
                        # Get the decoded explanation for this position
                        best_k_idx = best_indices[pos_in_batch].item()
                        explanation_idx = pos_in_batch * k + best_k_idx
                        explanation_text = all_decoded_texts[explanation_idx]
                        
                        print(f"\nHigh MSE at position {actual_position}, MSE: {mse_value:.2f}, norm: {batch_A[pos_in_batch].norm()}, norm_hat: {batch_best_A_hat[pos_in_batch].norm()}")
                        print(f"Generated explanation: '{explanation_text}'")
                        
                        # Get current token
                        current_token_id = current_token_ids[pos_in_batch].item()
                        current_token_text = self.tokenizer.decode([current_token_id], skip_special_tokens=False)
                        print(f"last 10 tokens: {escape_newlines(self.tokenizer.batch_decode(input_ids[start_idx+pos_in_batch:start_idx+pos_in_batch+1,max(0,actual_position-100):min(actual_position+1,input_ids.shape[1])], skip_special_tokens=False))[0]} on token '{current_token_text}' (id: {current_token_id})")
                        
                        # If we have access to full input_ids, show context
                        if input_ids is not None:
                            print(f"Token position in sequence: {actual_position}")
            
            all_best_A_hat.append(batch_best_A_hat)
            all_best_mses.append(batch_best_mses)
            all_best_tokens.append(batch_best_tokens)
    
        # Concatenate all batches
        return (
            torch.cat(all_best_A_hat, dim=0),
            torch.cat(all_best_mses, dim=0),
            torch.cat(all_best_tokens, dim=0)
        )
    
    def evaluate_with_k(
        self,
        val_loader,
        k: int,
        max_batches: Optional[int] = None,
        temperature: float = 1.0,
        max_generation_batch_size: int = 32,  # Renamed from generation_batch_size
        sample_positions: bool = True,
        num_positions: int = 1,
        min_pos: Optional[int] = None,
        position_selection_strategy: Optional[str] = None,
        vector_extraction_batch_size: int = 8,
        do_bootstrap: bool = True,
        load_store: bool = False,
        cache_dir: Optional[Path] = None,
        model_name: Optional[str] = None,
        dataset_cfg: Optional[Dict[str, Any]] = None,
        rank: int = 0,
        max_val_samples: int = 1000,
    ) -> Dict[str, Any]:
        """Evaluate model with best-of-K sampling and calculate variance recovery."""
        
        # Use provided values or defaults
        if min_pos is not None:
            self.min_pos = min_pos
        if position_selection_strategy is not None:
            self.position_selection_strategy = position_selection_strategy
            
        all_A = []
        all_A_hat = []
        all_mses = []
        all_positions_list = []
        all_token_ids_list = []
        total_positions = 0
        
        # Check if we should use caching
        cache_key = None
        cache_path = None
        cached_data = None
        if load_store and cache_dir and model_name and dataset_cfg:
            cache_key = compute_cache_key(model_name, self.layer, dataset_cfg, num_positions)
            cache_path = get_cache_path(cache_dir, cache_key, rank)
            
            # Try to load cached vectors
            cached_data = load_cached_vectors(cache_path)
            if cached_data is not None:
                print(f"Using cached vectors from {cache_path}")
                cached_all_A = cached_data['all_A'].to(self.device)
                cached_positions = cached_data['all_positions'].to(self.device)
                cached_token_ids = cached_data['all_token_ids'].to(self.device)
                
                # Check if we need to use a subset of cached vectors
                cached_samples = cached_all_A.shape[0] // num_positions  # Number of cached samples
                requested_samples = max_batches if max_batches else float('inf')
                
                if cached_samples > requested_samples:
                    # Use only the requested number of samples
                    vectors_to_use = requested_samples * num_positions
                    print(f"Cache has {cached_all_A.shape[0]} vectors ({cached_samples} samples), using first {vectors_to_use} vectors ({requested_samples} samples)")
                    cached_all_A = cached_all_A[:vectors_to_use]
                    cached_positions = cached_positions[:vectors_to_use]
                    cached_token_ids = cached_token_ids[:vectors_to_use]
                else:
                    print(f"Using all {cached_all_A.shape[0]} cached vectors ({cached_samples} samples)")
                
                # Generate best-of-K using cached vectors
                print(f"Generating best-of-{k} explanations using {cached_all_A.shape[0]} vectors...")
                A_hat_flat, mses_flat, _ = self.generate_best_of_k(
                    cached_all_A, cached_positions, k, temperature,
                    max_batch_size=max_generation_batch_size,
                    position_batch_size=max_generation_batch_size,
                    current_token_ids_all=cached_token_ids,
                    input_ids=None  # We don't have the full input_ids from cache
                )
                
                # Store results and skip vector extraction
                all_A.append(cached_all_A.cpu())
                all_A_hat.append(A_hat_flat.cpu())
                all_mses.append(mses_flat.cpu())
                total_positions = cached_all_A.shape[0]
                
                # Skip the normal extraction loop
                # Go directly to concatenation and statistics calculation
            else:
                print(f"No cached vectors found at {cache_path}, will extract and cache them")
        
        # For position sampling
        if sample_positions:
            import random
            random.seed(42)  # For reproducibility

        # Only extract vectors if we didn't load from cache
        if not (load_store and cached_data is not None):
            # We need to actually iterate through the dataloader
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Best-of-{k} Eval")):
                    if max_batches and batch_idx >= max_batches:
                        break

                    input_ids_A = batch['input_ids_A'].to(self.device)
                    batch_size, seq_len = input_ids_A.shape

                    # If num_positions==1 and 'A' in batch, use precomputed activations and positions directly
                    if sample_positions and seq_len > 5 and num_positions == 1 and 'A' in batch:
                        A_flat = batch['A'].to(self.device)
                        positions = batch['token_pos_A'].to(self.device)
                        current_token_ids = batch['input_ids_A'].to(self.device)[torch.arange(batch_size, device=self.device), positions]
                        hidden_dim = A_flat.shape[-1]
                    else:
                        # Otherwise, compute activations as needed
                        all_A_batch = []
                        for start in range(0, batch_size, vector_extraction_batch_size):
                            end = min(start + vector_extraction_batch_size, batch_size)
                            input_ids_chunk = input_ids_A[start:end]
                            A_chunk = self.orig_model.get_all_activations_at_layer(
                                input_ids_chunk, self.layer, no_grad=True
                            )
                            all_A_batch.append(A_chunk)
                        A_batch = torch.cat(all_A_batch, dim=0)
                        batch_size, seq_len, hidden_dim = A_batch.shape

                        # Get attention mask
                        attention_mask = batch.get('attention_mask', None)
                        if attention_mask is None and self.tokenizer.pad_token_id is not None:
                            attention_mask = batch['input_ids_A'].ne(self.tokenizer.pad_token_id).long()

                        if sample_positions and seq_len > 5:
                            all_sampled_A = []
                            all_sampled_positions = []
                            all_sampled_token_ids = []
                            for b in range(batch_size):
                                input_ids_b = batch['input_ids_A'][b].to(self.device)

                                if attention_mask is not None:
                                    attn_mask_b = attention_mask[b].to(self.device)
                                    start_idx = torch.argmax(attn_mask_b.int())
                                    end_idx = seq_len - 1 - torch.argmax(torch.flip(attn_mask_b, dims=[0]).int())
                                    if not torch.any(attn_mask_b):
                                        continue
                                else:
                                    start_idx = torch.tensor(0, device=self.device)
                                    end_idx = torch.tensor(seq_len - 1, device=self.device)

                                lower_bound = start_idx + self.min_pos
                                effective_lower_bound = torch.minimum(lower_bound, end_idx)
                                upper_bound = end_idx

                                if effective_lower_bound > upper_bound:
                                    continue

                                # Sample positions, deduplicate, then ensure num_positions by resampling if needed
                                sampled_positions = []
                                attempts = 0
                                max_attempts = 10 * num_positions
                                while len(set(sampled_positions)) < num_positions and attempts < max_attempts:
                                    if self.position_selection_strategy == "midpoint":
                                        pos = (effective_lower_bound + upper_bound) // 2
                                    elif self.position_selection_strategy == "random":
                                        rand_float = random.random()
                                        valid_range = (upper_bound - effective_lower_bound + 1).clamp(min=1)
                                        pos = effective_lower_bound + int(rand_float * valid_range.item())
                                    else:
                                        raise ValueError(f"Unknown position_selection_strategy: '{self.position_selection_strategy}'")
                                    pos = torch.clamp(torch.tensor(pos, device=self.device), min=0, max=seq_len-1)
                                    sampled_positions.append(pos.item())
                                    attempts += 1

                                # Remove duplicates and sort
                                sampled_positions = sorted(list(set(sampled_positions)))

                                # If still not enough, pad by random sampling (with replacement)
                                while len(sampled_positions) < num_positions:
                                    if self.position_selection_strategy == "midpoint":
                                        pos = (effective_lower_bound + upper_bound) // 2
                                    elif self.position_selection_strategy == "random":
                                        rand_float = random.random()
                                        valid_range = (upper_bound - effective_lower_bound + 1).clamp(min=1)
                                        pos = effective_lower_bound + int(rand_float * valid_range.item())
                                    else:
                                        raise ValueError(f"Unknown position_selection_strategy: '{self.position_selection_strategy}'")
                                    pos = torch.clamp(torch.tensor(pos, device=self.device), min=0, max=seq_len-1)
                                    sampled_positions.append(pos.item())
                                    sampled_positions = sorted(list(set(sampled_positions)))
                                # If we overshot, truncate
                                sampled_positions = sampled_positions[:num_positions]

                                if len(sampled_positions) == 0:
                                    continue

                                sampled_positions_tensor = torch.tensor(sampled_positions, device=self.device)

                                A_sampled = A_batch[b, sampled_positions, :]
                                all_sampled_A.append(A_sampled)
                                all_sampled_positions.append(sampled_positions_tensor)
                                token_ids_sampled = input_ids_b[sampled_positions]
                                all_sampled_token_ids.append(token_ids_sampled)

                            if len(all_sampled_A) == 0:
                                continue

                            A_flat = torch.cat(all_sampled_A, dim=0)
                            positions = torch.cat(all_sampled_positions, dim=0)
                            current_token_ids = torch.cat(all_sampled_token_ids, dim=0)
                        else:
                            # Use all positions
                            A_flat = A_batch.view(-1, hidden_dim)
                            positions = torch.arange(seq_len, device=self.device).repeat(batch_size)
                            batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, seq_len).reshape(-1)
                            position_indices = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1).reshape(-1)
                            current_token_ids = batch['input_ids_A'].to(self.device)[batch_indices, position_indices]

                    # Generate best-of-K
                    A_hat_flat, mses_flat, _ = self.generate_best_of_k(
                        A_flat, positions, k, temperature, 
                        max_batch_size=max_generation_batch_size,
                        position_batch_size=max_generation_batch_size,  # Keep original batch size for positions
                        current_token_ids_all=current_token_ids,
                        input_ids=batch['input_ids_A'].to(self.device)
                    )

                    # Store results
                    all_A.append(A_flat.cpu())
                    all_A_hat.append(A_hat_flat.cpu())
                    all_mses.append(mses_flat.cpu())
                    all_positions_list.append(positions.cpu())
                    all_token_ids_list.append(current_token_ids.cpu())
                    total_positions += A_flat.shape[0]
        
        # Concatenate all results (handle empty case)
        if len(all_A) > 0:
            all_A = torch.cat(all_A, dim=0)
            all_A_hat = torch.cat(all_A_hat, dim=0)
            all_mses = torch.cat(all_mses, dim=0)
        else:
            # Handle case where no vectors were extracted
            print(f"Warning: No vectors extracted on rank {rank}")
            all_A = torch.tensor([])
            all_A_hat = torch.tensor([])
            all_mses = torch.tensor([])
        
        # Save to cache if caching is enabled and we extracted vectors
        if load_store and cache_path and not (load_store and cached_data is not None):
            # Concatenate positions and token IDs
            all_positions = torch.cat(all_positions_list, dim=0) if all_positions_list else torch.tensor([])
            all_token_ids = torch.cat(all_token_ids_list, dim=0) if all_token_ids_list else torch.tensor([])
            
            # Save the extracted vectors
            metadata = {
                'num_positions': num_positions,
                'min_pos': self.min_pos,
                'position_selection_strategy': self.position_selection_strategy,
                'total_positions': total_positions,
                'sample_positions': sample_positions,
                'num_samples': total_positions // num_positions if num_positions > 0 else 0,
            }
            save_cached_vectors(cache_path, all_A, all_positions, all_token_ids, metadata)
        
        # In distributed mode, we need to gather all activations to compute global statistics
        if dist.is_initialized() and dist.get_world_size() > 1:
            # Gather all activations from all processes
            world_size = dist.get_world_size()
            
            # Get sizes from all processes
            local_size = torch.tensor(all_A.shape[0], device=self.device)
            all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
            dist.all_gather(all_sizes, local_size)
            
            # Prepare gathered tensors
            max_size = max(s.item() for s in all_sizes)
            padded_A = torch.zeros(max_size, all_A.shape[1], device=self.device)
            padded_A_hat = torch.zeros(max_size, all_A_hat.shape[1], device=self.device)
            padded_mses = torch.zeros(max_size, device=self.device)
            
            # Copy local data
            padded_A[:all_A.shape[0]] = all_A.to(self.device)
            padded_A_hat[:all_A_hat.shape[0]] = all_A_hat.to(self.device)
            padded_mses[:all_mses.shape[0]] = all_mses.to(self.device)
            
            # Gather from all processes
            gathered_A = [torch.zeros_like(padded_A) for _ in range(world_size)]
            gathered_A_hat = [torch.zeros_like(padded_A_hat) for _ in range(world_size)]
            gathered_mses = [torch.zeros_like(padded_mses) for _ in range(world_size)]
            
            dist.all_gather(gathered_A, padded_A)
            dist.all_gather(gathered_A_hat, padded_A_hat)
            dist.all_gather(gathered_mses, padded_mses)
            
            # Concatenate only valid data
            all_A_list = []
            all_A_hat_list = []
            all_mses_list = []
            for i, size in enumerate(all_sizes):
                all_A_list.append(gathered_A[i][:size.item()].cpu())
                all_A_hat_list.append(gathered_A_hat[i][:size.item()].cpu())
                all_mses_list.append(gathered_mses[i][:size.item()].cpu())
            
            all_A = torch.cat(all_A_list, dim=0).float().numpy()
            all_A_hat = torch.cat(all_A_hat_list, dim=0).float().numpy()
            all_mses = torch.cat(all_mses_list, dim=0).float().numpy()
        else:
            all_A = all_A.float().cpu().numpy()
            all_A_hat = all_A_hat.float().cpu().numpy()
            all_mses = all_mses.float().cpu().numpy()
        
        # Now calculate variance recovery on the full dataset
        print(f"Calculating variance recovered with {all_A.shape[0]} vectors from {all_A.shape[0]//num_positions} sequences.")
        
        # Check sample size and warn if small
        n_samples = all_A.shape[0]
        if n_samples < 1000:
            print(f"⚠️  Warning: Small sample size ({n_samples}). Variance recovery estimates may be biased.")
        
        # Compute variance explained (Equation 10 from https://arxiv.org/abs/2404.16014)
        # Using ddof=1 for unbiased variance estimation
        total_variance = np.var(all_A, axis=0, ddof=1).sum()
        residual_variance = np.var(all_A - all_A_hat, axis=0, ddof=1).sum()
        variance_recovery = 1 - (residual_variance / total_variance) if total_variance > 0 else 0.0
        
        if do_bootstrap:
            # Bootstrap for error estimation and bias correction
            n_bootstrap = 1000
            bootstrap_var_recoveries = []
            bootstrap_mses = []
            bootstrap_r_squared = []
        
            print(f"Running bootstrap with {n_bootstrap} samples for error estimation and bias correction...")
            np.random.seed(42)  # For reproducibility
        
            for _ in tqdm(range(n_bootstrap), desc="Bootstrap", leave=False):
                # Sample with replacement
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)

                A_boot = all_A[bootstrap_indices]
                A_hat_boot = all_A_hat[bootstrap_indices]

                # Calculate variance recovery for bootstrap sample (with ddof=1)
                total_var_boot = np.var(A_boot, axis=0, ddof=1).sum()
                residual_var_boot = np.var(A_boot - A_hat_boot, axis=0, ddof=1).sum()
                var_recovery_boot = 1 - (residual_var_boot / total_var_boot) if total_var_boot > 0 else 0.0
                bootstrap_var_recoveries.append(var_recovery_boot)

                # MSE for bootstrap sample
                mse_boot = np.mean((A_boot - A_hat_boot) ** 2)
                bootstrap_mses.append(mse_boot)

                # R-squared for bootstrap sample
                mean_A_boot = np.mean(A_boot)
                ss_res_boot = np.sum((A_boot - A_hat_boot) ** 2)
                ss_tot_boot = np.sum((A_boot - mean_A_boot) ** 2)
                r_squared_boot = 1 - (ss_res_boot / ss_tot_boot) if ss_tot_boot > 0 else 0.0
                bootstrap_r_squared.append(r_squared_boot)
        
            # Calculate bias and bias-corrected estimates
            bootstrap_var_recoveries = np.array(bootstrap_var_recoveries)
            var_recovery_bias = np.mean(bootstrap_var_recoveries) - variance_recovery
            var_recovery_bias_corrected = 2 * variance_recovery - np.mean(bootstrap_var_recoveries)
        
            # Alternative: BCa (bias-corrected and accelerated) confidence intervals
            # First calculate the bias correction factor z0
            z0 = scipy.stats.norm.ppf((bootstrap_var_recoveries < variance_recovery).mean())
        
            # Jackknife for acceleration factor
            jackknife_var_recoveries = []
            for i in range(n_samples):
                # Leave one out
                mask = np.ones(n_samples, dtype=bool)
                mask[i] = False
                A_jack = all_A[mask]
                A_hat_jack = all_A_hat[mask]

                total_var_jack = np.var(A_jack, axis=0, ddof=1).sum()
                residual_var_jack = np.var(A_jack - A_hat_jack, axis=0, ddof=1).sum()
                var_recovery_jack = 1 - (residual_var_jack / total_var_jack) if total_var_jack > 0 else 0.0
                jackknife_var_recoveries.append(var_recovery_jack)
        
            jackknife_var_recoveries = np.array(jackknife_var_recoveries)
            jack_mean = np.mean(jackknife_var_recoveries)
        
            # Acceleration factor
            numerator = np.sum((jack_mean - jackknife_var_recoveries) ** 3)
            denominator = 6 * (np.sum((jack_mean - jackknife_var_recoveries) ** 2) ** 1.5)
            a = numerator / denominator if denominator > 0 else 0.0
        
            # BCa confidence intervals
            alpha = 0.05  # For 95% CI
            z_alpha = scipy.stats.norm.ppf([alpha/2, 1-alpha/2])
        
            # Adjusted percentiles
            alpha1 = scipy.stats.norm.cdf(z0 + (z0 + z_alpha[0]) / (1 - a * (z0 + z_alpha[0])))
            alpha2 = scipy.stats.norm.cdf(z0 + (z0 + z_alpha[1]) / (1 - a * (z0 + z_alpha[1])))
        
            # BCa confidence intervals
            var_recovery_bca_ci = np.percentile(bootstrap_var_recoveries, [alpha1 * 100, alpha2 * 100])
        
            # Standard confidence intervals (percentile method)
            var_recovery_std = np.std(bootstrap_var_recoveries)
            var_recovery_ci = np.percentile(bootstrap_var_recoveries, [2.5, 97.5])
        
            # Do the same for MSE and R-squared
            mse_std = np.std(bootstrap_mses)
            mse_ci = np.percentile(bootstrap_mses, [2.5, 97.5])
        
            r_squared_std = np.std(bootstrap_r_squared)
            r_squared_ci = np.percentile(bootstrap_r_squared, [2.5, 97.5])
        else:
            var_recovery_bias = 0
            var_recovery_bias_corrected = 0
            var_recovery_std = 0
            var_recovery_ci = [0, 0]
            var_recovery_bca_ci = [0, 0]
            mse_ci = [0, 0]
            r_squared_ci = [0, 0]
            mse_std = 0
            r_squared_std = 0
        
            # Also compute mean and MSE for additional metrics
        mean_A = np.mean(all_A)
        var_A_direct = np.var(all_A, ddof=1)
        mse = np.mean((all_A - all_A_hat) ** 2)
        
        # R-squared
        ss_res = np.sum((all_A - all_A_hat) ** 2)
        ss_tot = np.sum((all_A - mean_A) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            'k': k,
            'avg_mse': float(np.mean(all_mses)),
            'variance_recovery': float(variance_recovery),
            'variance_recovery_bias': float(var_recovery_bias),
            'variance_recovery_bias_corrected': float(var_recovery_bias_corrected),
            'variance_recovery_std': float(var_recovery_std),
            'variance_recovery_ci_lower': float(var_recovery_ci[0]),
            'variance_recovery_ci_upper': float(var_recovery_ci[1]),
            'variance_recovery_bca_ci_lower': float(var_recovery_bca_ci[0]),
            'variance_recovery_bca_ci_upper': float(var_recovery_bca_ci[1]),
            'mse_std': float(mse_std),
            'mse_ci_lower': float(mse_ci[0]),
            'mse_ci_upper': float(mse_ci[1]),
            'r_squared': float(r_squared),
            'r_squared_std': float(r_squared_std),
            'r_squared_ci_lower': float(r_squared_ci[0]),
            'r_squared_ci_upper': float(r_squared_ci[1]),
            'total_positions': int(total_positions),
            'original_variance': float(var_A_direct),
            'original_mean': float(mean_A),
            'mean_variance_mean_over_hidden': np.var(all_A-all_A_hat, axis=0, ddof=1).mean(),
            'residual_variance_mean_over_hidden': np.var(all_A-all_A_hat, axis=0, ddof=1).mean()
        }


def setup_distributed_eval(cfg: DictConfig):
    """Initialize distributed training environment."""
    rank, world_size, local_rank = init_distributed()
    setup_for_distributed(rank == 0)
    
    # Set device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    if cfg.get('seed') is not None:
        set_seed(cfg.seed + rank)
    
    return rank, world_size, device


def prepare_val_dataloader(cfg: DictConfig, rank: int, world_size: int, device: torch.device, log=None, analyzer=None, max_val_samples_req=1000, dataloader_batch_size=32, do_not_extract_activations_val=True):
    """Prepare validation dataloader."""
    # Extract dataset paths from config with safe defaults
    dataset_cfg = cfg.get('dataset', {})
    
    activation_dir = dataset_cfg.get('activation_dir', './data/SimpleStories_train')
    val_activation_dir = dataset_cfg.get('val_activation_dir', activation_dir)
    
    # Check if on-the-fly generation is enabled
    on_the_fly_cfg = dataset_cfg.get('on_the_fly', {})
    on_the_fly_enabled = on_the_fly_cfg.get('enabled', False)
    
    # For evaluation, we typically don't need on-the-fly generation if we have pre-computed activations
    # Check if we have a valid validation activation directory
    if on_the_fly_enabled and val_activation_dir and Path(val_activation_dir).exists():
        # If we have pre-computed validation activations, disable on-the-fly for evaluation
        if log and is_main():
            log.info(f"Found pre-computed validation activations at {val_activation_dir}. Disabling on-the-fly generation for evaluation.")
        on_the_fly_enabled = False
    
    # If on-the-fly generation is enabled, we need to load the original model
    orig_model_for_gen = None
    tokenizer_for_gen = None
    generation_device = None
    rank_for_gen = None
    world_size_for_gen = None
    
    if on_the_fly_enabled:
        if log and is_main():
            log.info("On-the-fly generation is enabled. Loading original model for generation...")
        
        # If analyzer is provided and has the original model, use it
        if analyzer and hasattr(analyzer, 'orig_model') and analyzer.orig_model is not None:
            orig_model_for_gen = analyzer.orig_model
            tokenizer_for_gen = analyzer.tokenizer
            generation_device = device
            rank_for_gen = rank
            world_size_for_gen = world_size
            if log and is_main():
                log.info("Using original model from analyzer for on-the-fly generation")
        else:
            # Load the original model and tokenizer for generation
            orig_model_name = cfg.get('orig_model_name', cfg.get('model_name'))
            
            if orig_model_name:
                # Import necessary classes
                from transformers import AutoTokenizer
                
                # Load tokenizer
                tokenizer_for_gen = AutoTokenizer.from_pretrained(orig_model_name)
                if tokenizer_for_gen.pad_token is None:
                    tokenizer_for_gen.pad_token = tokenizer_for_gen.eos_token
                
                # Create OrigWrapper with model name
                orig_model_for_gen = OrigWrapper(
                    orig_model_name,
                    torch_dtype=torch.bfloat16 if cfg.get('use_bf16', True) else torch.float32,
                    load_in_8bit=False,
                    base_to_use=None  # Will load its own model
                )
                orig_model_for_gen.to(device)
                # Note: OrigWrapper already sets the model to eval() mode in __init__
                
                # Set generation parameters
                generation_device = device
                rank_for_gen = rank
                world_size_for_gen = world_size
            else:
                raise ValueError("orig_model_name must be specified when on_the_fly generation is enabled")
    
    # Use _prepare_dataloaders to get datasets
    _, val_dataset = _prepare_dataloaders(
        config=OmegaConf.to_container(cfg, resolve=True),  # Convert to dict
        activation_dir=activation_dir,
        effective_val_activation_dir=val_activation_dir,
        max_train_samples_req=0,
        max_val_samples_req= max_val_samples_req,
        log=log,
        orig_model_for_gen=orig_model_for_gen,
        tokenizer_for_gen=tokenizer_for_gen,
        generation_device=generation_device,
        rank=rank_for_gen,
        world_size=world_size_for_gen,
        samples_per_regeneration_cycle=None,  # Not needed for validation
        do_not_extract_activations_val=do_not_extract_activations_val
    )
    
    if val_dataset is None or len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty")
    
    val_sampler = FastDistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=cfg.get('seed', 42),
        drop_last=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=dataloader_batch_size,
        sampler=val_sampler,
        num_workers=cfg.get('data', {}).get('num_workers', 0),
        pin_memory=True,
        drop_last=False,
        collate_fn=collate
    )
    
    return val_loader, orig_model_for_gen


@hydra.main(version_base=None, config_path="../conf", config_name="gemma3_CHAT_27b_frozen_nopostfix")
def main(cfg: DictConfig) -> None:
    """Run best-of-K sweep analysis."""
    
    # Check if we're using an eval-specific config
    if 'eval' in cfg and isinstance(cfg.eval, DictConfig):
        # If eval config is loaded, use it directly
        eval_cfg = cfg.eval
        print(f"Using eval config: {eval_cfg}")
    else:
        # Otherwise, create a default eval config
        eval_cfg = OmegaConf.create({
            'checkpoint_path': cfg.get('checkpoint_path', ''),
            'k_values': [1, 4, 16],
            'batch_size': 32,
            'max_batches': None,
            'temperature': 1.0,
            'max_generation_batch_size': 32,  # Renamed from generation_batch_size
            'use_bf16': True,
            'output_dir': 'eval_results',
            'output_file': 'k_sweep_results.json',
            'num_positions': 1
        })
    
    # Setup distributed
    rank, world_size, device = setup_distributed_eval(cfg)
    is_main_process = is_main()
    
    # Initialize logging
    if is_main_process:
        print("Starting best-of-K sweep analysis...")
        print(f"Checkpoint: {eval_cfg.checkpoint_path}")
        print(f"K values: {eval_cfg.k_values}")
    
    # Load checkpoint and merge configs
    checkpoint_path = Path(eval_cfg.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Create a new config by merging checkpoint config with current config
    if 'config' in ckpt:
        # Start with checkpoint config as base
        merged_cfg = OmegaConf.create(ckpt['config'])
        
        # Override with eval-specific settings
        OmegaConf.set_struct(merged_cfg, False)
        if 'eval' not in merged_cfg:
            merged_cfg.eval = OmegaConf.create({})
        
        # Copy all eval settings
        for key, value in eval_cfg.items():
            merged_cfg.eval[key] = value
        
        # Ensure required keys exist with defaults
        if 'dataset' not in merged_cfg:
            merged_cfg.dataset = OmegaConf.create({
                'activation_dir': './data/SimpleStories_train',
                'val_activation_dir': './data/SimpleStories_test'
            })
        
        if 'data' not in merged_cfg:
            merged_cfg.data = OmegaConf.create({'num_workers': 0})
        
        # Preserve seed if it exists
        if 'seed' in cfg:
            merged_cfg.seed = cfg.seed
            
        OmegaConf.set_struct(merged_cfg, True)
        cfg = merged_cfg
    else:
        # No config in checkpoint, just ensure eval config is accessible
        if 'eval' not in cfg:
            OmegaConf.set_struct(cfg, False)
            cfg.eval = eval_cfg
            OmegaConf.set_struct(cfg, True)
    
    # Initialize minimal logging for dataloader preparation
    # We don't need wandb for evaluation, just use a simple logger
    if is_main_process:
        import logging
        log = logging.getLogger(__name__)
        log.setLevel(logging.INFO)
        # Add console handler if not already present
        if not log.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            log.addHandler(handler)
    else:
        log = None
    max_val_samples_req = eval_cfg.get('max_val_samples') if eval_cfg.get('max_val_samples') is not None else 1000
    
    # Check if cache exists before creating dataloader
    should_skip_extraction = False
    if eval_cfg.get('load_store', False) and eval_cfg.get('num_positions', 1) == 1:
        # Check if cache would exist
        cache_dir = Path(eval_cfg.get('cache_dir', 'vector_cache'))
        model_name = cfg.get('orig_model_name', cfg.get('model_name'))
        
        # We need to know the layer to compute cache key
        # Load checkpoint briefly to get the layer
        ckpt_temp = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'encoder_config' in ckpt_temp:
            layer = ckpt_temp['encoder_config']['layer']
        elif 'layer' in ckpt_temp:
            layer = ckpt_temp['layer']
        else:
            layer = 5  # Default fallback
        
        if model_name and 'dataset' in cfg:
            # Compute cache key
            dataset_cfg_dict = OmegaConf.to_container(cfg.dataset, resolve=True)
            cache_key = compute_cache_key(model_name, layer, dataset_cfg_dict, eval_cfg.get('num_positions', 1))
            cache_path = get_cache_path(cache_dir, cache_key, rank)
            
            # Check if cache exists
            if cache_path.exists():
                print(f"Cache exists at {cache_path}, will skip vector extraction in dataloader")
                should_skip_extraction = True
    
    # Prepare dataloader - we'll pass None for analyzer since we'll create it after
    cfg["dataset"]["on_the_fly"]["generation_batch_size"] = eval_cfg.get('dataloader_fwd_pass_batch_size', 32)
    # When num_positions=1 AND cache exists, set do_not_extract_activations_val=True to skip extraction
    do_not_extract = eval_cfg.num_positions != 1 or should_skip_extraction
    val_loader, orig_model_for_gen = prepare_val_dataloader(cfg, rank, world_size, device, log, analyzer=None, max_val_samples_req=max_val_samples_req, dataloader_batch_size=eval_cfg.get('dataloader_batch_size', 32), do_not_extract_activations_val=do_not_extract)
    
    # Initialize evaluator
    # Check if we need to pass the original model name to the analyzer
    orig_model_name_for_analyzer = None
    if cfg.data.get('on_the_fly', {}).get('enabled', False):
        # Use the same model name that was used for dataloader generation
        orig_model_name_for_analyzer = cfg.data.on_the_fly.get('orig_model_name')
    
    evaluator = BestOfKEvaluator(
        str(checkpoint_path),
        device,
        use_bf16=cfg.eval.get('use_bf16', True),
        orig_model_name=orig_model_name_for_analyzer,
        orig_model_wrapper=orig_model_for_gen
    )
    
    # Run sweep with simplified access
    all_results = []
    k_values = cfg.eval.k_values

    def to_serializable(obj):
        # Recursively convert numpy types to native Python types
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    for k in k_values:
        if is_main_process:
            print(f"\nEvaluating with K={k}...")
        # Reset DataLoader iterator to the beginning
        val_loader_iter = iter(val_loader)
        
        results = evaluator.evaluate_with_k(
            val_loader_iter,
            k=k,
            max_batches=cfg.eval.get('max_batches'),
            temperature=cfg.eval.get('temperature', 1.0),
            max_generation_batch_size=cfg.eval.get('max_generation_batch_size', 32),  # Changed parameter name
            vector_extraction_batch_size=cfg.eval.get('vector_extraction_batch_size', 8),
            do_bootstrap=cfg.eval.get('do_bootstrap', True),
            load_store=cfg.eval.get('load_store', False),
            cache_dir=Path(cfg.eval.get('cache_dir', 'vector_cache')),
            model_name=cfg.get('orig_model_name', cfg.get('model_name')),
            dataset_cfg=OmegaConf.to_container(cfg.dataset, resolve=True) if 'dataset' in cfg else {},
            rank=rank,
            max_val_samples=max_val_samples_req,
        )
        
        # Gather results across processes if distributed
        if dist.is_initialized() and world_size > 1:
            # Convert results to tensors for gathering
            gathered_results = {}
            for key, value in results.items():
                if key == 'k':
                    gathered_results[key] = value
                else:
                    tensor_value = torch.tensor(value, device=device)
                    gathered_list = [torch.zeros_like(tensor_value) for _ in range(world_size)]
                    dist.all_gather(gathered_list, tensor_value)
                    
                    # For statistics that are already computed on the full gathered dataset,
                    # just take the value from rank 0 (all ranks should have the same value)
                    if key in ['avg_mse', 'variance_recovery', 'r_squared', 'original_variance', 'original_mean', 'total_positions']:
                        gathered_results[key] = float(gathered_list[0].item()) if key != 'total_positions' else int(gathered_list[0].item())
            
            results = gathered_results
        
        all_results.append(results)
        
        if is_main_process:
            print(f"K={k}: MSE={results['avg_mse']:.6f}±{results['mse_std']:.6f}, Variance Recovered={results['variance_recovery']:.4f}±{results['variance_recovery_std']:.4f} with 95% CI [{results['variance_recovery_ci_lower']:.4f}, {results['variance_recovery_ci_upper']:.4f}] and confidence interval for bias [{results['variance_recovery_bca_ci_lower']:.4f}, {results['variance_recovery_bca_ci_upper']:.4f}]")
    
    # Save results
    if is_main_process:
        # Get output config
        output_dir = Path(cfg.eval.get('output_dir', 'eval_results'))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / cfg.eval.get('output_file', 'k_sweep_results.json')
        
        final_results = {
            'checkpoint_path': str(checkpoint_path),
            'eval_config': OmegaConf.to_container(cfg.eval, resolve=True),
            'results_by_k': all_results,
            'summary': {
                'k_values': [float(k) for k in k_values],
                'mse_values': [float(r['avg_mse']) for r in all_results],
                'variance_recovery_values': [float(r['variance_recovery']) for r in all_results],
                'r_squared_values': [float(r['r_squared']) for r in all_results]
            }
        }
        with open(output_file, 'w') as f:
            json.dump(to_serializable(final_results), f, indent=2)
        print(f"\nResults saved to: {output_file}")
        print("\nSummary:")
        print("K\tVar Recovery (Raw)\tVar Recovery (Bias-Corrected)\tBias\t\tR²")
        print("-" * 100)
        for r in all_results:
            bias_corrected = r.get('variance_recovery_bias_corrected', r['variance_recovery'])
            bias = r.get('variance_recovery_bias', 0)
            print(f"{r['k']}\t{r['variance_recovery']:.4f}±{r.get('variance_recovery_std', 0):.4f}\t\t"
                  f"{bias_corrected:.4f}\t\t\t{bias:.4f}\t"
                  f"{r['r_squared']:.4f}±{r.get('r_squared_std', 0):.4f}")
        
        # Print confidence intervals
        print("\n95% Confidence Intervals:")
        print("K\tVar Recovery CI (Percentile)\tVar Recovery CI (BCa)")
        print("-" * 80)
        for r in all_results:
            if 'variance_recovery_ci_lower' in r:
                print(f"{r['k']}\t[{r['variance_recovery_ci_lower']:.4f}, {r['variance_recovery_ci_upper']:.4f}]\t\t"
                      f"[{r.get('variance_recovery_bca_ci_lower', r['variance_recovery_ci_lower']):.4f}, "
                      f"{r.get('variance_recovery_bca_ci_upper', r['variance_recovery_ci_upper']):.4f}]")
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()