#!/usr/bin/env python3
"""
Best-of-K sweep analysis for Talkative Autoencoders.
Evaluates how variance recovery improves with different K values in best-of-K sampling.
"""

import copy
import json

# Note: We don't use wandb logging for evaluation scripts
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
import scipy.stats
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

# Note: DDP not needed for this script as we're using the analyzer directly
# Core lens imports
from lens.analysis.analyzer_class import LensAnalyzer
from lens.data.collate import collate
from lens.models.orig import OrigWrapper
from lens.training.distributed import init_distributed, is_main, set_seed, setup_for_distributed
from lens.training.fast_distributed_sampler import FastDistributedSampler
from lens.training.train_aux import _prepare_dataloaders


def get_dataloader_for_distributed(dataset, batch_size, world_size, rank, shuffle=True, **kwargs):
    """Create a DataLoader with DistributedSampler if needed."""
    # Check if dataset is already sharded (like RankInMemoryTrainingCache)
    is_pre_sharded = hasattr(dataset, "rank") and hasattr(dataset, "world_size")

    if world_size > 1 and not is_pre_sharded:
        sampler = FastDistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        kwargs.pop("shuffle", None)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, **kwargs)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    return dataloader


def escape_newlines(text: str) -> str:
    return text.replace("\n", "\\n")


def get_cache_path(cache_dir: Path, model_name: str, layer: int, seq_len: int, rank: int = 0) -> Path:
    """Get the cache file path with descriptive naming: model_name/layer/seq_len/vectors.pkl"""
    # Sanitize model name for filesystem (replace slashes with underscores)
    safe_model_name = model_name.replace("/", "_")

    # Create hierarchical path: vector_cache/model_name/layer_X/seq_Y/vectors.pkl
    cache_path = cache_dir / safe_model_name / f"layer_{layer}" / f"seq_{seq_len}" / "vectors.pkl"
    return cache_path


def save_cached_vectors(
    cache_path: Path,
    all_A: torch.Tensor,
    all_positions: torch.Tensor,
    all_token_ids: torch.Tensor,
    metadata: Dict[str, Any],
    all_input_ids: torch.Tensor = None,
) -> None:
    """Save extracted vectors and metadata to cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache_data = {
        "all_A": all_A.cpu(),
        "all_positions": all_positions.cpu(),
        "all_token_ids": all_token_ids.cpu(),
        "metadata": metadata,
        "all_input_ids": all_input_ids.cpu() if all_input_ids is not None else None,
    }

    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)

    print(f"Saved cache to {cache_path}")


def load_cached_vectors(cache_path: Path) -> Optional[Dict[str, Any]]:
    """Load cached vectors if they exist."""
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)
        print(f"Loaded cache from {cache_path}")
        # Ensure all_input_ids is present, if not, set to None for backward compatibility
        if "all_input_ids" not in cache_data:
            cache_data["all_input_ids"] = None

        return cache_data
    except Exception as e:
        print(f"Failed to load cache from {cache_path}: {e}")
        return None


def get_token_role_classifier(tokenizer, log=None):
    """
    Create a token role classifier based on the tokenizer type.
    Returns a function that classifies token positions in a sequence.
    """
    tokenizer_class = tokenizer.__class__.__name__.lower()
    model_type = None

    # Get special token IDs directly
    special_token_ids = set()
    if hasattr(tokenizer, "all_special_ids"):
        special_token_ids = set(tokenizer.all_special_ids)

    # Gemma models
    if "gemma" in tokenizer_class or hasattr(tokenizer, "vocab"):
        model_type = "gemma"
        # For Gemma, we need to look for the actual token IDs
        # Common patterns: <start_of_turn>, <end_of_turn>, user, model
        start_of_turn_id = None
        end_of_turn_id = None
        user_id = None
        model_id = None

        # Try different methods to get special tokens
        # Method 1: Check special tokens dict
        if hasattr(tokenizer, "special_tokens_map"):
            if log:
                log.info(f"Special tokens map: {tokenizer.special_tokens_map}")

        # Method 2: Try to get from added_tokens_encoder
        if hasattr(tokenizer, "added_tokens_encoder"):
            start_of_turn_id = tokenizer.added_tokens_encoder.get("<start_of_turn>", None)
            end_of_turn_id = tokenizer.added_tokens_encoder.get("<end_of_turn>", None)
            if log:
                log.info(f"Added tokens encoder has start_of_turn: {start_of_turn_id}, end_of_turn: {end_of_turn_id}")

        # Method 3: Try encoding to get IDs
        try:
            # For Gemma, we might need to encode the full pattern
            start_user = tokenizer.encode("<start_of_turn>user", add_special_tokens=False)
            start_model = tokenizer.encode("<start_of_turn>model", add_special_tokens=False)
            end_turn = tokenizer.encode("<end_of_turn>", add_special_tokens=False)

            if log:
                log.info(f"Encoded '<start_of_turn>user': {start_user}")
                log.info(f"Encoded '<start_of_turn>model': {start_model}")
                log.info(f"Encoded '<end_of_turn>': {end_turn}")

            # Also try individual tokens
            user_id = (
                tokenizer.encode("user", add_special_tokens=False)[0]
                if tokenizer.encode("user", add_special_tokens=False)
                else None
            )
            model_id = (
                tokenizer.encode("model", add_special_tokens=False)[0]
                if tokenizer.encode("model", add_special_tokens=False)
                else None
            )
        except Exception as e:
            if log:
                log.warning(f"Error encoding Gemma tokens: {e}")

        if log:
            log.info(f"Gemma token IDs - user: {user_id}, model: {model_id}")

    # Get pad token ID
    pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") else None

    # Store the encoded patterns for Gemma
    start_user_pattern = None
    start_model_pattern = None
    end_turn_pattern = None

    if model_type == "gemma":
        try:
            start_user_pattern = tokenizer.encode("<start_of_turn>user", add_special_tokens=False)
            start_model_pattern = tokenizer.encode("<start_of_turn>model", add_special_tokens=False)
            end_turn_pattern = tokenizer.encode("<end_of_turn>", add_special_tokens=False)
        except:
            pass

    def classify_sequence(input_ids):
        """Classify each token position in the sequence."""
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.cpu().numpy()

        roles = ["other"] * len(input_ids)
        current_role = "other"

        i = 0
        while i < len(input_ids):
            token_id = input_ids[i]

            # Check if it's a pad token
            if pad_token_id is not None and token_id == pad_token_id:
                roles[i] = "pad"
                i += 1
                continue

            # For Gemma, check for role transitions using pattern matching
            if model_type == "gemma":
                pattern_found = False

                # Check for <start_of_turn>user pattern
                if start_user_pattern and i + len(start_user_pattern) <= len(input_ids):
                    if all(input_ids[i + j] == start_user_pattern[j] for j in range(len(start_user_pattern))):
                        current_role = "user"
                        for j in range(len(start_user_pattern)):
                            roles[i + j] = "special"
                        i += len(start_user_pattern)
                        pattern_found = True

                # Check for <start_of_turn>model pattern
                if not pattern_found and start_model_pattern and i + len(start_model_pattern) <= len(input_ids):
                    if all(input_ids[i + j] == start_model_pattern[j] for j in range(len(start_model_pattern))):
                        current_role = "assistant"
                        for j in range(len(start_model_pattern)):
                            roles[i + j] = "special"
                        i += len(start_model_pattern)
                        pattern_found = True

                # Check for <end_of_turn> pattern
                if not pattern_found and end_turn_pattern and i + len(end_turn_pattern) <= len(input_ids):
                    if all(input_ids[i + j] == end_turn_pattern[j] for j in range(len(end_turn_pattern))):
                        for j in range(len(end_turn_pattern)):
                            roles[i + j] = "special"
                        i += len(end_turn_pattern)
                        pattern_found = True

                if pattern_found:
                    continue

            # Check if it's a special token (but not pad)
            if token_id in special_token_ids:
                roles[i] = "special"
            else:
                # Regular token - assign current role
                roles[i] = current_role

            i += 1

        return roles

    return classify_sequence, model_type


def get_token_roles_for_positions(input_ids_batch, positions_batch, tokenizer):
    """
    Get token roles for specific positions in a batch of sequences.

    Args:
        input_ids_batch: Tensor of shape (batch_size, seq_len) or flattened positions
        positions_batch: Tensor of positions (may be flattened)
        tokenizer: The tokenizer

    Returns:
        List of token roles for each position
    """
    classify_sequence, _ = get_token_role_classifier(tokenizer)
    token_roles = []

    if isinstance(positions_batch, torch.Tensor):
        positions_batch = positions_batch.cpu().numpy()

    # Handle different input formats
    if isinstance(input_ids_batch, torch.Tensor) and len(input_ids_batch.shape) == 2:
        # Standard case: batch of sequences
        batch_size, seq_len = input_ids_batch.shape

        # If positions are flattened but we have multiple sequences
        if len(positions_batch.shape) == 1 and len(positions_batch) > batch_size:
            # Positions are sampled across multiple sequences
            # We need to map each position back to its sequence
            for i, pos in enumerate(positions_batch):
                seq_idx = i // (len(positions_batch) // batch_size) if batch_size > 0 else 0
                if seq_idx < batch_size:
                    input_ids = input_ids_batch[seq_idx].cpu().numpy()
                    roles = classify_sequence(input_ids)
                    if 0 <= pos < len(roles):
                        token_roles.append(roles[pos])
                    else:
                        token_roles.append("other")
                else:
                    token_roles.append("other")
        else:
            # One position per sequence
            for seq_idx in range(batch_size):
                input_ids = input_ids_batch[seq_idx].cpu().numpy()
                roles = classify_sequence(input_ids)
                if seq_idx < len(positions_batch):
                    pos = positions_batch[seq_idx]
                    if 0 <= pos < len(roles):
                        token_roles.append(roles[pos])
                    else:
                        token_roles.append("other")
    elif len(input_ids_batch.shape) == 2 and input_ids_batch.shape[0] == len(positions_batch):
        # Each position has its own full sequence (from cache)
        for input_ids, pos in zip(input_ids_batch, positions_batch):
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.cpu().numpy()
            roles = classify_sequence(input_ids)
            if 0 <= pos < len(roles):
                token_roles.append(roles[pos])
            else:
                token_roles.append("other")
    else:
        # Fallback: return "other" for all positions
        token_roles = ["other"] * len(positions_batch)

    return token_roles


def calculate_selected_token_stats(token_roles, log=None, is_main=True):
    """
    Calculate statistics for selected token positions.

    Args:
        token_roles: List of token roles for selected positions
        log: Optional logger for output
        is_main: Whether this is the main process (controls logging)

    Returns:
        Dict with token counts and proportions for selected positions
    """
    # Initialize counters
    role_counts = {"user": 0, "assistant": 0, "system": 0, "special": 0, "other": 0, "pad": 0, "total": 0}

    # Count tokens by role
    for role in token_roles:
        if role in role_counts:
            role_counts[role] += 1
        role_counts["total"] += 1

    # Calculate proportions
    total = role_counts["total"]
    proportions = {}
    for role in ["user", "assistant", "system", "special", "other", "pad"]:
        proportions[role] = role_counts[role] / total if total > 0 else 0.0

    stats = {"counts": role_counts, "proportions": proportions, "total_positions": total}

    if is_main and log:
        log.info("\n" + "=" * 50)
        log.info("Selected Token Position Statistics:")
        log.info("=" * 50)
        log.info(f"Total selected positions: {total}")
        log.info(f"User tokens: {role_counts['user']} ({proportions['user']:.1%})")
        log.info(f"Assistant tokens: {role_counts['assistant']} ({proportions['assistant']:.1%})")
        log.info(f"System tokens: {role_counts['system']} ({proportions['system']:.1%})")
        log.info(f"Special tokens: {role_counts['special']} ({proportions['special']:.1%})")
        log.info(f"Other tokens: {role_counts['other']} ({proportions['other']:.1%})")
        if role_counts["pad"] > 0:
            log.info(f"Pad tokens: {role_counts['pad']} ({proportions['pad']:.1%})")
        log.info("=" * 50 + "\n")

    return stats


def calculate_chat_token_stats(input_ids_list, tokenizer, log=None, is_main=True):
    """
    Calculate statistics on the proportion of user/assistant/system tokens in chat-formatted data.

    Args:
        input_ids_list: List of input_ids tensors or a single tensor of shape (batch, seq_len)
        tokenizer: The tokenizer with special tokens for chat formatting
        log: Optional logger for output
        is_main: Whether this is the main process (controls logging)

    Returns:
        Dict with token counts and proportions for each role
    """
    # Convert to list if single tensor
    if isinstance(input_ids_list, torch.Tensor):
        if len(input_ids_list.shape) == 2:
            input_ids_list = [input_ids_list[i] for i in range(input_ids_list.shape[0])]
        else:
            input_ids_list = [input_ids_list]

    # Initialize counters
    role_counts = {"user": 0, "assistant": 0, "system": 0, "special": 0, "other": 0, "pad": 0, "total": 0}

    # Get the classifier function
    classify_sequence, model_type = get_token_role_classifier(tokenizer, log if is_main else None)

    # Process each sequence
    for seq_idx, input_ids in enumerate(input_ids_list):
        roles = classify_sequence(input_ids)

        # Count tokens by role
        for role in roles:
            role_counts[role] += 1
            role_counts["total"] += 1

        # Debug first sequence
        if is_main and log and seq_idx == 0:
            # Show first 100 tokens with their roles
            debug_length = min(100, len(input_ids))
            if isinstance(input_ids, torch.Tensor):
                debug_ids = input_ids[:debug_length].cpu().tolist()
            else:
                debug_ids = input_ids[:debug_length].tolist()
            debug_tokens = tokenizer.convert_ids_to_tokens(debug_ids)
            debug_roles = roles[:debug_length]

            log.info("\nFirst 100 tokens with roles:")
            for i in range(0, debug_length, 10):
                tokens_slice = debug_tokens[i : i + 10]
                roles_slice = debug_roles[i : i + 10]
                log.info(f"Tokens: {tokens_slice}")
                log.info(f"Roles:  {roles_slice}")

    # Calculate proportions
    total = role_counts["total"]
    proportions = {}
    for role in ["user", "assistant", "system", "special", "other", "pad"]:
        proportions[role] = role_counts[role] / total if total > 0 else 0.0

    stats = {
        "counts": role_counts,
        "proportions": proportions,
        "model_type": model_type,
        "total_sequences": len(input_ids_list),
    }

    if is_main and log:
        log.info("\n" + "=" * 50)
        log.info("Chat Token Statistics:")
        log.info("=" * 50)
        log.info(f"Model type: {model_type}")
        log.info(f"Total sequences: {len(input_ids_list)}")
        log.info(f"Total tokens: {total}")
        log.info(f"User tokens: {role_counts['user']} ({proportions['user']:.1%})")
        log.info(f"Assistant tokens: {role_counts['assistant']} ({proportions['assistant']:.1%})")
        log.info(f"System tokens: {role_counts['system']} ({proportions['system']:.1%})")
        log.info(f"Special tokens: {role_counts['special']} ({proportions['special']:.1%})")
        log.info(f"Other tokens: {role_counts['other']} ({proportions['other']:.1%})")
        if role_counts["pad"] > 0:
            log.info(f"Pad tokens: {role_counts['pad']} ({proportions['pad']:.1%})")
        log.info("=" * 50 + "\n")

    return stats


class BestOfKEvaluator:
    """Evaluates best-of-K performance with variance recovery calculation."""

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        use_bf16: bool = True,
        orig_model_name=None,
        orig_model_wrapper=None,
        use_hard_encoder_string=False,
    ):
        self.device = device
        self.use_bf16 = use_bf16

        # Load analyzer with optional original model name for different activations
        if orig_model_name is not None:
            print("loading with shared and diff")
            # Pass the model name to avoid duplicate loading
            self.analyzer = LensAnalyzer(
                checkpoint_path=checkpoint_path,
                device=device,
                no_orig=True if orig_model_wrapper is not None else False,
                use_bf16=use_bf16,
                different_activations_orig=orig_model_wrapper if orig_model_wrapper is not None else None,
                shared_base_model=orig_model_wrapper.model if orig_model_wrapper is not None else None,
                comparison_tl_checkpoint=False,
            )
        else:
            self.analyzer = LensAnalyzer(
                checkpoint_path=checkpoint_path,
                device=device,
                use_bf16=use_bf16,
                shared_base_model=orig_model_wrapper.model if orig_model_wrapper is not None else None,
                comparison_tl_checkpoint=False,
            )
        self.analyzer.to(device)

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
        self.position_selection_strategy = "random"  # Default strategy

        if use_hard_encoder_string:
            print("using hard encoder string")
            self.encoder.set_soft_prompt_from_text(self.encoder.config.soft_prompt_init_text, self.tokenizer)

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
        log=None,
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
        for start_idx in tqdm(
            range(0, total_positions, position_batch_size), desc=f"Generating best-of-{k}", leave=False
        ):
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
                        current_token_ids=current_token_ids_expanded[sub_start:sub_end]
                        if self.encoder.config.add_current_token
                        else None,
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
                mses = torch.nn.functional.mse_loss(A_expanded, A_hat_expanded, reduction="none").mean(dim=1)

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
                batch_A_norms = batch_A.norm(dim=1)

                # Print only the 5 largest mses and the 5 largest norms (if present and not overlapping with mses)
                if debug_high_mse:
                    # Ensure top_mse_indices is always defined
                    top_mse_indices = torch.tensor([], dtype=torch.long, device=batch_best_mses.device)
                    # Find top 5 largest MSEs above threshold
                    high_mse_indices = torch.where(batch_best_mses > 1000000)[0]
                    if high_mse_indices.numel() > 0:
                        top_mse_vals, top_mse_idx_in_mask = torch.topk(
                            batch_best_mses[high_mse_indices], min(5, high_mse_indices.numel())
                        )
                        top_mse_indices = high_mse_indices[top_mse_idx_in_mask]
                        log.info(f"\n⚠️  Top {top_mse_indices.numel()} positions with largest MSE > 200000")
                        for number, idx in enumerate(top_mse_indices):
                            log.info(f"Top {number} mse: {top_mse_vals[number].item()}")
                            log.info("----------------")
                            pos_in_batch = idx.item()
                            actual_position = batch_positions[pos_in_batch].item()
                            mse_value = batch_best_mses[pos_in_batch].item()
                            norm_value = batch_A[pos_in_batch].norm().item()
                            norm_hat_value = batch_best_A_hat[pos_in_batch].norm().item()
                            best_k_idx = best_indices[pos_in_batch].item()
                            explanation_idx = pos_in_batch * k + best_k_idx
                            explanation_text = all_decoded_texts[explanation_idx]
                            log.info(
                                f"\nHigh MSE at position {actual_position}, MSE: {mse_value:.2f}, norm: {norm_value}, norm_hat: {norm_hat_value}"
                            )
                            log.info(f"Generated explanation: '{explanation_text}'")
                            current_token_id = current_token_ids[pos_in_batch].item()
                            current_token_text = self.tokenizer.decode([current_token_id], skip_special_tokens=False)
                            log.info(
                                f"last 10 tokens: {escape_newlines(self.tokenizer.batch_decode(input_ids[start_idx + pos_in_batch : start_idx + pos_in_batch + 1, max(0, actual_position - 10000) : min(actual_position + 1, input_ids.shape[1])], skip_special_tokens=False)[0])}\n\n--------------\n\n on token '{escape_newlines(current_token_text)}' (id: {current_token_id})"
                            )
                            if input_ids is not None:
                                log.info(f"Token position in sequence: {actual_position}")

                    # Find top 5 largest norms above threshold, excluding those already printed for MSE
                    high_norm_indices = torch.where(batch_A_norms > 1000000)[0]
                    # Exclude indices already in top_mse_indices
                    norm_only_indices = [
                        idx for idx in high_norm_indices.tolist() if idx not in set(top_mse_indices.tolist())
                    ]
                    if len(norm_only_indices) > 0:
                        norm_only_indices_tensor = torch.tensor(norm_only_indices, device=batch_A_norms.device)
                        top_norm_vals, top_norm_idx_in_mask = torch.topk(
                            batch_A_norms[norm_only_indices_tensor], min(5, len(norm_only_indices))
                        )
                        top_norm_indices = norm_only_indices_tensor[top_norm_idx_in_mask]
                        log.info(
                            f"\n⚠️  Top {top_norm_indices.numel()} positions with largest norm > 200000 (not already shown for MSE)"
                        )
                        for number, idx in enumerate(top_norm_indices):
                            log.info(f"Top {number} norm: {top_norm_vals[number].item()}")
                            log.info("----------------")
                            pos_in_batch = idx.item()
                            actual_position = batch_positions[pos_in_batch].item()
                            mse_value = batch_best_mses[pos_in_batch].item()
                            norm_value = batch_A[pos_in_batch].norm().item()
                            norm_hat_value = batch_best_A_hat[pos_in_batch].norm().item()
                            best_k_idx = best_indices[pos_in_batch].item()
                            explanation_idx = pos_in_batch * k + best_k_idx
                            explanation_text = all_decoded_texts[explanation_idx]
                            log.info(
                                f"\nHigh norm at position {actual_position}, MSE: {mse_value:.2f}, norm: {norm_value}, norm_hat: {norm_hat_value}"
                            )
                            log.info(f"Generated explanation: '{explanation_text}'")
                            current_token_id = current_token_ids[pos_in_batch].item()
                            current_token_text = self.tokenizer.decode([current_token_id], skip_special_tokens=False)
                            log.info(
                                f"last 10 tokens: {escape_newlines(self.tokenizer.batch_decode(input_ids[start_idx + pos_in_batch : start_idx + pos_in_batch + 1, max(0, actual_position - 10000) : min(actual_position + 1, input_ids.shape[1])], skip_special_tokens=False)[0])}\n\n--------------\n\n on token '{escape_newlines(current_token_text)}' (id: {current_token_id})"
                            )
                            if input_ids is not None:
                                log.info(f"Token position in sequence: {actual_position}")

            all_best_A_hat.append(batch_best_A_hat)
            all_best_mses.append(batch_best_mses)
            all_best_tokens.append(batch_best_tokens)

        # Concatenate all batches
        return (torch.cat(all_best_A_hat, dim=0), torch.cat(all_best_mses, dim=0), torch.cat(all_best_tokens, dim=0))

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
        log=None,
        compute_variance_by_role: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate model with best-of-K sampling and calculate variance recovery."""

        # Use provided values or defaults
        if min_pos is not None:
            self.min_pos = min_pos
        if position_selection_strategy is not None:
            self.position_selection_strategy = position_selection_strategy

        # --- 1. Try to load cache if requested ---
        use_cached = False
        cache_path = None
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        cached_data = None
        if load_store and cache_dir and model_name and dataset_cfg:
            # Get sequence length from dataset config
            seq_len = dataset_cfg.get("seq_len", 1024)  # Default to 1024 if not specified
            cache_path = get_cache_path(cache_dir, model_name, self.layer, seq_len, rank)
            cached_data = load_cached_vectors(cache_path)
            if cached_data is not None:
                # Safety check: only use cache if it has enough vectors
                cached_num_vectors = cached_data["all_A"].shape[0]

                # Always expect max_val_samples total vectors (across all ranks)
                expected_samples = max_val_samples

                if cached_num_vectors < expected_samples:
                    print(
                        f"⚠️  Cache at {cache_path} only has {cached_num_vectors} vectors, but {expected_samples} required. Ignoring cache and regenerating."
                    )
                else:
                    use_cached = True
            else:
                log.warning(f"No cache found at {cache_path} at rank {rank}")

        # --- 2. If cache is found, use it and skip extraction ---
        if use_cached:
            log.info(f"Using cached vectors from {cache_path} at rank {rank}")
            cached_all_A = cached_data["all_A"].to(self.device)
            cached_positions = cached_data["all_positions"].to(self.device)
            cached_token_ids = cached_data["all_token_ids"].to(self.device)
            cached_all_input_ids = cached_data["all_input_ids"].to(self.device)

            cached_samples = cached_all_A.shape[0] // num_positions
            if max_batches:
                requested_samples = max_batches
            else:
                requested_samples = max_val_samples // world_size if world_size > 1 else max_val_samples

            if cached_samples > requested_samples:
                vectors_to_use = requested_samples * num_positions
                start = rank * vectors_to_use
                end = start + vectors_to_use
                log.info(
                    f"Cache has {cached_all_A.shape[0]} vectors ({cached_samples} samples), using only rank {rank} slice [{start}:{end}] ({requested_samples} samples)"
                )
                cached_all_A = cached_all_A[start:end]
                cached_positions = cached_positions[start:end]
                cached_token_ids = cached_token_ids[start:end]
                cached_all_input_ids = cached_all_input_ids[start:end]
            elif cached_samples == requested_samples:
                start = rank * requested_samples * num_positions
                end = start + requested_samples * num_positions
                cached_all_A = cached_all_A[start:end]
                cached_positions = cached_positions[start:end]
                cached_token_ids = cached_token_ids[start:end]
                cached_all_input_ids = cached_all_input_ids[start:end]
                log.info(
                    f"Using all {cached_all_A.shape[0]} cached vectors ({requested_samples} samples) for rank {rank}"
                )
            else:
                raise ValueError(
                    f"Cache has {cached_all_A.shape[0]} vectors ({cached_samples} samples), but requested {requested_samples} samples"
                )

            # Generate best-of-K using cached vectors
            log.info(f"Generating best-of-{k} explanations using {cached_all_A.shape[0]} vectors...")
            if k == 1:
                # Decode the first 100 tokens, keeping special characters
                try:
                    first_token_ids = cached_all_input_ids[0]
                    decoded = self.tokenizer.decode(first_token_ids.flatten().tolist(), skip_special_tokens=False)
                    log.info(f"First decoded (up to 100 chars): '{decoded[:]}'")
                except Exception as e:
                    log.warning(f"Error decoding first tokens: {e}")

            # Calculate chat token statistics
            token_stats = None
            all_token_roles = []
            if cached_all_input_ids is not None and cached_all_input_ids.shape[0] > 0:
                # First calculate local stats without logging
                log.info("Calculating local chat token statistics from cached data...")
                local_token_stats = calculate_chat_token_stats(
                    cached_all_input_ids, self.tokenizer, log=None, is_main=False
                )

                # Get token roles for each position
                log.info("Getting token roles for each position...")
                all_token_roles = get_token_roles_for_positions(cached_all_input_ids, cached_positions, self.tokenizer)

                # Calculate stats for selected positions
                local_selected_stats = calculate_selected_token_stats(all_token_roles, log=None, is_main=False)

            A_hat_flat, mses_flat, _ = self.generate_best_of_k(
                cached_all_A,
                cached_positions,
                k,
                temperature,
                max_batch_size=max_generation_batch_size,
                position_batch_size=max_generation_batch_size,
                current_token_ids_all=cached_token_ids,
                input_ids=cached_all_input_ids,
                log=log,
            )

            # Store results and return early
            all_A = cached_all_A.cpu()
            all_A_hat = A_hat_flat.cpu()
            all_mses = mses_flat.cpu()
            total_positions = cached_all_A.shape[0]
            # In distributed mode, we need to gather all activations to compute global statistics
            log.info(f"Gathering {all_A.shape[0]} from process {rank}")
            if dist.is_initialized() and dist.get_world_size() > 1:
                # Gather all activations from all processes
                world_size = dist.get_world_size()
                log.info(f"Rank {rank} has {all_A.shape[0]} vectors")

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
                padded_A[: all_A.shape[0]] = all_A.to(self.device)
                padded_A_hat[: all_A_hat.shape[0]] = all_A_hat.to(self.device)
                padded_mses[: all_mses.shape[0]] = all_mses.to(self.device)

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
                    all_A_list.append(gathered_A[i][: size.item()].cpu())
                    all_A_hat_list.append(gathered_A_hat[i][: size.item()].cpu())
                    all_mses_list.append(gathered_mses[i][: size.item()].cpu())

                all_A = torch.cat(all_A_list, dim=0).float().numpy()
                all_A_hat = torch.cat(all_A_hat_list, dim=0).float().numpy()
                all_mses = torch.cat(all_mses_list, dim=0).float().numpy()
            else:
                all_A = all_A.float().cpu().numpy()
                all_A_hat = all_A_hat.float().cpu().numpy()
                all_mses = all_mses.float().cpu().numpy()

            # Aggregate token statistics across all GPUs
            selected_token_stats = None
            if "local_token_stats" in locals() and local_token_stats is not None:
                if dist.is_initialized() and dist.get_world_size() > 1:
                    # Gather token counts from all ranks
                    local_counts = local_token_stats["counts"]
                    gathered_counts = [None] * world_size
                    dist.all_gather_object(gathered_counts, local_counts)

                    # Sum up counts from all ranks
                    total_counts = {
                        "user": sum(counts["user"] for counts in gathered_counts),
                        "assistant": sum(counts["assistant"] for counts in gathered_counts),
                        "system": sum(counts["system"] for counts in gathered_counts),
                        "special": sum(counts["special"] for counts in gathered_counts),
                        "other": sum(counts["other"] for counts in gathered_counts),
                        "pad": sum(counts["pad"] for counts in gathered_counts),
                        "total": sum(counts["total"] for counts in gathered_counts),
                    }

                    # Calculate global proportions
                    total = total_counts["total"]
                    proportions = {}
                    for role in ["user", "assistant", "system", "special", "other", "pad"]:
                        proportions[role] = total_counts[role] / total if total > 0 else 0.0

                    # Update token_stats with aggregated data
                    token_stats = {
                        "counts": total_counts,
                        "proportions": proportions,
                        "model_type": local_token_stats["model_type"],
                        "total_sequences": sum(
                            counts.get("total_sequences", 0) for counts in gathered_counts if isinstance(counts, dict)
                        ),
                    }

                    # Log aggregated stats on rank 0
                    if rank == 0 and log:
                        log.info("\n" + "=" * 50)
                        log.info("Aggregated Chat Token Statistics (All GPUs):")
                        log.info("=" * 50)
                        log.info(f"Model type: {token_stats['model_type']}")
                        log.info(f"Total sequences: {token_stats['total_sequences']}")
                        log.info(f"Total tokens: {total}")
                        log.info(f"User tokens: {total_counts['user']} ({proportions['user']:.1%})")
                        log.info(f"Assistant tokens: {total_counts['assistant']} ({proportions['assistant']:.1%})")
                        log.info(f"System tokens: {total_counts['system']} ({proportions['system']:.1%})")
                        log.info(f"Special tokens: {total_counts['special']} ({proportions['special']:.1%})")
                        log.info(f"Other tokens: {total_counts['other']} ({proportions['other']:.1%})")
                        if total_counts["pad"] > 0:
                            log.info(f"Pad tokens: {total_counts['pad']} ({proportions['pad']:.1%})")
                        log.info("=" * 50 + "\n")
                else:
                    token_stats = local_token_stats

            # Aggregate selected token statistics across all GPUs
            if "local_selected_stats" in locals() and local_selected_stats is not None:
                if dist.is_initialized() and dist.get_world_size() > 1:
                    # Gather selected token counts from all ranks
                    local_selected_counts = local_selected_stats["counts"]
                    gathered_selected_counts = [None] * world_size
                    dist.all_gather_object(gathered_selected_counts, local_selected_counts)

                    # Sum up counts from all ranks
                    total_selected_counts = {
                        "user": sum(counts["user"] for counts in gathered_selected_counts),
                        "assistant": sum(counts["assistant"] for counts in gathered_selected_counts),
                        "system": sum(counts["system"] for counts in gathered_selected_counts),
                        "special": sum(counts["special"] for counts in gathered_selected_counts),
                        "other": sum(counts["other"] for counts in gathered_selected_counts),
                        "pad": sum(counts["pad"] for counts in gathered_selected_counts),
                        "total": sum(counts["total"] for counts in gathered_selected_counts),
                    }

                    # Calculate global proportions
                    total_selected = total_selected_counts["total"]
                    selected_proportions = {}
                    for role in ["user", "assistant", "system", "special", "other", "pad"]:
                        selected_proportions[role] = (
                            total_selected_counts[role] / total_selected if total_selected > 0 else 0.0
                        )

                    # Update selected_token_stats with aggregated data
                    selected_token_stats = {
                        "counts": total_selected_counts,
                        "proportions": selected_proportions,
                        "total_positions": total_selected,
                    }

                    # Log aggregated selected stats on rank 0
                    if rank == 0 and log:
                        log.info("\n" + "=" * 50)
                        log.info("Aggregated Selected Token Statistics (All GPUs):")
                        log.info("=" * 50)
                        log.info(f"Total selected positions: {total_selected}")
                        log.info(f"User tokens: {total_selected_counts['user']} ({selected_proportions['user']:.1%})")
                        log.info(
                            f"Assistant tokens: {total_selected_counts['assistant']} ({selected_proportions['assistant']:.1%})"
                        )
                        log.info(
                            f"System tokens: {total_selected_counts['system']} ({selected_proportions['system']:.1%})"
                        )
                        log.info(
                            f"Special tokens: {total_selected_counts['special']} ({selected_proportions['special']:.1%})"
                        )
                        log.info(
                            f"Other tokens: {total_selected_counts['other']} ({selected_proportions['other']:.1%})"
                        )
                        if total_selected_counts["pad"] > 0:
                            log.info(f"Pad tokens: {total_selected_counts['pad']} ({selected_proportions['pad']:.1%})")
                        log.info("=" * 50 + "\n")
                else:
                    selected_token_stats = local_selected_stats

            # Now calculate variance recovery on the full dataset
            if rank == 0:
                log.info(
                    f"Calculating variance recovered with {all_A.shape[0]} vectors from {all_A.shape[0] // num_positions} sequences."
                )

            # Check sample size and warn if small
            n_samples = all_A.shape[0]
            if n_samples < 1000:
                print(f"⚠️  Warning: Small sample size ({n_samples}). Variance recovery estimates may be biased.")

            # Compute variance explained (Equation 10 from https://arxiv.org/abs/2404.16014)
            # Using ddof=1 for unbiased variance estimation
            total_variance = np.var(all_A, axis=0, ddof=1).sum()
            residual_variance = np.var(all_A - all_A_hat, axis=0, ddof=1).sum()
            variance_recovery = 1 - (residual_variance / total_variance) if total_variance > 0 else 0.0
            mse_for_new_variance = np.mean((all_A - all_A_hat) ** 2, axis=0).sum()
            mse_variance_recovery = 1 - (mse_for_new_variance / total_variance) if total_variance > 0 else 0.0

            if do_bootstrap:
                # Bootstrap for error estimation and bias correction
                n_bootstrap = 1000
                bootstrap_var_recoveries = []
                bootstrap_mses = []
                bootstrap_r_squared = []

                log.info(f"Running bootstrap with {n_bootstrap} samples for error estimation and bias correction...")
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
                z_alpha = scipy.stats.norm.ppf([alpha / 2, 1 - alpha / 2])

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
            # Vector-wise norms for downstream histogram
            activation_norms = np.linalg.norm(all_A, axis=1)

            # R-squared
            ss_res = np.sum((all_A - all_A_hat) ** 2)
            ss_tot = np.sum((all_A - mean_A) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            # Calculate MSE breakdown by token role if available
            mse_by_role = {}
            variance_recovery_by_role = {}
            if "all_token_roles" in locals() and all_token_roles and compute_variance_by_role:
                # Group MSEs and activations by token role
                role_mses = {}
                role_activations = {}
                role_reconstructions = {}

                for i, (mse, role) in enumerate(zip(all_mses, all_token_roles)):
                    if role not in role_mses:
                        role_mses[role] = []
                        role_activations[role] = []
                        role_reconstructions[role] = []
                    role_mses[role].append(mse)
                    role_activations[role].append(all_A[i])
                    role_reconstructions[role].append(all_A_hat[i])

                # Calculate average MSE and variance recovery for each role
                for role, mses in role_mses.items():
                    mse_by_role[role] = {
                        "avg_mse": float(np.mean(mses)),
                        "count": len(mses),
                        "std": float(np.std(mses)),
                    }

                    # Calculate variance recovery for this role
                    if len(role_activations[role]) > 10:  # Only compute if we have enough samples
                        role_A = np.array(role_activations[role])
                        role_A_hat = np.array(role_reconstructions[role])

                        # Compute variance recovery for this role
                        role_total_variance = np.var(role_A, axis=0, ddof=1).sum()
                        role_residual_variance = np.var(role_A - role_A_hat, axis=0, ddof=1).sum()
                        role_variance_recovery = (
                            1 - (role_residual_variance / role_total_variance) if role_total_variance > 0 else 0.0
                        )

                        variance_recovery_by_role[role] = {
                            "variance_recovery": float(role_variance_recovery),
                            "total_variance": float(role_total_variance),
                            "residual_variance": float(role_residual_variance),
                            "count": len(role_activations[role]),
                        }

            # Include token stats if available
            result_dict = {
                "k": k,
                "avg_mse": float(np.mean(all_mses)),
                "variance_recovery": float(variance_recovery),
                "variance_recovery_bias": float(var_recovery_bias),
                "variance_recovery_bias_corrected": float(var_recovery_bias_corrected),
                "variance_recovery_std": float(var_recovery_std),
                "variance_recovery_ci_lower": float(var_recovery_ci[0]),
                "variance_recovery_ci_upper": float(var_recovery_ci[1]),
                "variance_recovery_bca_ci_lower": float(var_recovery_bca_ci[0]),
                "variance_recovery_bca_ci_upper": float(var_recovery_bca_ci[1]),
                "mse_std": float(mse_std),
                "mse_ci_lower": float(mse_ci[0]),
                "mse_ci_upper": float(mse_ci[1]),
                "r_squared": float(r_squared),
                "r_squared_std": float(r_squared_std),
                "r_squared_ci_lower": float(r_squared_ci[0]),
                "r_squared_ci_upper": float(r_squared_ci[1]),
                "total_positions": int(total_positions),
                "original_variance": float(var_A_direct),
                "original_mean": float(mean_A),
                "mean_variance_mean_over_hidden": np.var(all_A - all_A_hat, axis=0, ddof=1).mean(),
                "residual_variance_mean_over_hidden": np.var(all_A - all_A_hat, axis=0, ddof=1).mean(),
                # Extra data for histograms
                "mse_values": all_mses.tolist(),
                "activation_norms": activation_norms.tolist(),
                "mse_for_new_variance": float(mse_for_new_variance),
                "mse_variance_recovery": float(mse_variance_recovery),
            }

            # Add token stats if available
            if "token_stats" in locals() and token_stats is not None:
                result_dict["token_stats"] = token_stats

            # Add selected token stats if available
            if "selected_token_stats" in locals() and selected_token_stats is not None:
                result_dict["selected_token_stats"] = selected_token_stats

            # Add MSE by role if available
            if mse_by_role:
                result_dict["mse_by_role"] = mse_by_role

            # Add variance recovery by role if available
            if variance_recovery_by_role:
                result_dict["variance_recovery_by_role"] = variance_recovery_by_role

            return result_dict

        # --- 3. If no cache, use dataloader to extract vectors and fill cache ---
        print("No cached vectors found, extracting vectors using dataloader and will fill cache if enabled.")
        all_A = []
        all_A_hat = []
        all_mses = []
        all_positions_list = []
        all_token_ids_list = []
        all_input_ids_list = []
        all_token_roles = []
        total_positions = 0

        # We need to actually iterate through the dataloader
        with torch.no_grad():
            # Calculate per-rank sample limit
            # per_rank_sample_limit = max_val_samples // (dist.get_world_size() if dist.is_initialized() else 1)
            processed_samples = 0
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Best-of-{k} Eval on rank {rank}")):
                batch_size = batch["input_ids_A"].shape[0]
                # if processed_samples + batch_size > per_rank_sample_limit:
                #     # Only process up to the limit
                #     batch = {k: v[:per_rank_sample_limit - processed_samples] for k, v in batch.items()}
                #     batch_size = batch["input_ids_A"].shape[0]
                if max_batches and batch_idx >= max_batches:
                    log.info(f"Reached max_batches {max_batches} on rank {rank}")
                    break

                input_ids_A = batch["input_ids_A"].to(self.device)
                batch_size, seq_len = input_ids_A.shape

                # If num_positions==1 and 'A' in batch, use precomputed activations and positions directly
                if sample_positions and seq_len > 5 and num_positions == 1 and "A" in batch:
                    A_flat = batch["A"].to(self.device)
                    positions = batch["token_pos_A"].to(self.device)
                    current_token_ids = batch["input_ids_A"].to(self.device)[
                        torch.arange(batch_size, device=self.device), positions
                    ]
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
                    attention_mask = batch.get("attention_mask", None)
                    if attention_mask is None and self.tokenizer.pad_token_id is not None:
                        attention_mask = batch["input_ids_A"].ne(self.tokenizer.pad_token_id).long()

                    if sample_positions and seq_len > 5:
                        all_sampled_A = []
                        all_sampled_positions = []
                        all_sampled_token_ids = []
                        for b in range(batch_size):
                            input_ids_b = batch["input_ids_A"][b].to(self.device)

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
                                    raise ValueError(
                                        f"Unknown position_selection_strategy: '{self.position_selection_strategy}'"
                                    )
                                pos = torch.clamp(torch.tensor(pos, device=self.device), min=0, max=seq_len - 1)
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
                                    raise ValueError(
                                        f"Unknown position_selection_strategy: '{self.position_selection_strategy}'"
                                    )
                                pos = torch.clamp(torch.tensor(pos, device=self.device), min=0, max=seq_len - 1)
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
                        batch_indices = (
                            torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, seq_len).reshape(-1)
                        )
                        position_indices = (
                            torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1).reshape(-1)
                        )
                        current_token_ids = batch["input_ids_A"].to(self.device)[batch_indices, position_indices]

                # Generate best-of-K
                A_hat_flat, mses_flat, _ = self.generate_best_of_k(
                    A_flat,
                    positions,
                    k,
                    temperature,
                    max_batch_size=max_generation_batch_size,
                    position_batch_size=max_generation_batch_size,  # Keep original batch size for positions
                    current_token_ids_all=current_token_ids,
                    input_ids=input_ids_A,  # Pass the full input_ids_A for each sampled position
                    log=log,
                )

                # Store results
                all_A.append(A_flat.cpu())
                all_A_hat.append(A_hat_flat.cpu())
                all_mses.append(mses_flat.cpu())
                all_positions_list.append(positions.cpu())
                all_token_ids_list.append(current_token_ids.cpu())

                # Get token roles for these positions
                if "input_ids_A" in batch:
                    batch_token_roles = get_token_roles_for_positions(batch["input_ids_A"], positions, self.tokenizer)
                    all_token_roles.extend(batch_token_roles)
                # For each sampled position, store the full input sequence
                # Repeat the input sequence for each sampled position in this batch
                if "input_ids_A" in batch:
                    input_ids_A_cpu = batch["input_ids_A"].cpu()
                    # If positions is length N, and batch_size is B, we need to know which sequence each position came from
                    # For sample_positions, positions are sampled per sequence, so we can track this
                    if sample_positions and seq_len > 5 and num_positions == 1 and "A" in batch:
                        # Each position comes from a different sequence in the batch
                        for b in range(input_ids_A_cpu.shape[0]):
                            all_input_ids_list.append(input_ids_A_cpu[b])
                    else:
                        # For the general case, repeat the input sequence for each sampled position
                        # positions.shape[0] == number of sampled positions
                        # batch_indices tells us which sequence each position came from
                        if "batch_indices" in locals():
                            for idx in batch_indices.cpu().tolist():
                                all_input_ids_list.append(input_ids_A_cpu[idx])
                        else:
                            # Fallback: repeat the first sequence
                            for _ in range(positions.shape[0]):
                                all_input_ids_list.append(input_ids_A_cpu[0])
                total_positions += A_flat.shape[0]
                processed_samples += batch_size

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
        log.info(
            f"Finished evaluating on rank {rank}, we have {all_A.shape[0] if hasattr(all_A, 'shape') else 0} vectors"
        )

        # Save to cache if caching is enabled and we extracted vectors
        if load_store and cache_path:
            # In distributed mode, gather all vectors to rank 0 before saving
            if dist.is_initialized() and dist.get_world_size() > 1:
                world_size = dist.get_world_size()

                # Gather sizes first (all ranks participate, order is [0, 1, ..., world_size-1])
                local_size = torch.tensor(all_A.shape[0], device=self.device)
                all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
                dist.all_gather(all_sizes, local_size)

                # All ranks must synchronize here to ensure all_gather is complete
                dist.barrier()
                # The original send/recv logic was brittle and had a bug related to receiving tensors
                # with unknown dimensions. Using `dist.gather_object` is more robust as it handles
                # serialization of objects with varying sizes and shapes, simplifying the code.

                # Each rank prepares its data payload to be gathered on rank 0.
                # Tensors are moved to CPU to free up GPU memory and avoid issues with distributed communication.
                all_positions_concat = torch.cat(all_positions_list, dim=0) if all_positions_list else torch.tensor([])
                all_token_ids_concat = torch.cat(all_token_ids_list, dim=0) if all_token_ids_list else torch.tensor([])
                all_input_ids_concat = (
                    torch.stack(all_input_ids_list, dim=0) if all_input_ids_list else torch.tensor([])
                )

                payload = None
                if all_A.shape[0] > 0:
                    payload = {
                        "A": all_A.cpu(),
                        "positions": all_positions_concat.cpu(),
                        "token_ids": all_token_ids_concat.cpu(),
                        "input_ids": all_input_ids_concat.cpu(),
                    }

                # All ranks participate in the gather operation.
                # Rank 0 prepares a list to receive the payloads.
                gathered_payloads = [None] * world_size if rank == 0 else None
                dist.gather_object(payload, gathered_payloads, dst=0)

                # Rank 0 now has all the data and can process and save it.
                if rank == 0:
                    log.info(
                        f"Successfully gathered data on rank 0. Now processing and saving to cache at {cache_path}."
                    )

                    # Filter out empty payloads and extract tensors into lists
                    gathered_A_list = [p["A"] for p in gathered_payloads if p]
                    gathered_positions_list = [p["positions"] for p in gathered_payloads if p]
                    gathered_token_ids_list = [p["token_ids"] for p in gathered_payloads if p]
                    gathered_input_ids_list = [p["input_ids"] for p in gathered_payloads if p]

                    # Concatenate all gathered data into single tensors
                    all_A_gathered = torch.cat(gathered_A_list, dim=0) if gathered_A_list else torch.tensor([])
                    all_positions_gathered = (
                        torch.cat(gathered_positions_list, dim=0) if gathered_positions_list else torch.tensor([])
                    )
                    all_token_ids_gathered = (
                        torch.cat(gathered_token_ids_list, dim=0) if gathered_token_ids_list else torch.tensor([])
                    )
                    all_input_ids_gathered = (
                        torch.cat(gathered_input_ids_list, dim=0) if gathered_input_ids_list else torch.tensor([])
                    )

                    # Always overwrite cache if it exists
                    if cache_path.exists():
                        print(
                            f"⚠️  Overwriting existing cache at {cache_path} with new data ({all_A_gathered.shape[0]} vectors)."
                        )
                        cache_path.unlink()

                    # Save the centralized cache from rank 0
                    metadata = {
                        "num_positions": num_positions,
                        "min_pos": self.min_pos,
                        "position_selection_strategy": self.position_selection_strategy,
                        "total_positions": all_A_gathered.shape[0],
                        "sample_positions": sample_positions,
                        "num_samples": all_A_gathered.shape[0] // num_positions if num_positions > 0 else 0,
                        "original_world_size": world_size,
                    }
                    log.info(
                        f"Starting save process at {cache_path} from rank {rank} with {all_A_gathered.shape[0]} vectors"
                    )
                    save_cached_vectors(
                        cache_path,
                        all_A_gathered,
                        all_positions_gathered,
                        all_token_ids_gathered,
                        metadata,
                        all_input_ids_gathered,
                    )

                # Synchronize before continuing
                log.info(f"Synchronizing before continuing on rank {rank}")
                dist.barrier()
            else:
                # Single process - save directly
                all_positions = torch.cat(all_positions_list, dim=0) if all_positions_list else torch.tensor([])
                all_token_ids = torch.cat(all_token_ids_list, dim=0) if all_token_ids_list else torch.tensor([])
                all_input_ids = torch.stack(all_input_ids_list, dim=0) if all_input_ids_list else torch.tensor([])

                metadata = {
                    "num_positions": num_positions,
                    "min_pos": self.min_pos,
                    "position_selection_strategy": self.position_selection_strategy,
                    "total_positions": total_positions,
                    "sample_positions": sample_positions,
                    "num_samples": total_positions // num_positions if num_positions > 0 else 0,
                    "original_world_size": 1,
                }
                # Always overwrite cache if it exists
                if cache_path.exists():
                    print(f"⚠️  Overwriting existing cache at {cache_path} with new data ({all_A.shape[0]} vectors).")
                    cache_path.unlink()
                save_cached_vectors(cache_path, all_A, all_positions, all_token_ids, metadata, all_input_ids)

        # Calculate chat token statistics
        token_stats = None
        local_token_stats = None
        local_selected_stats = None
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            if "all_input_ids" in locals() and all_input_ids is not None and all_input_ids.shape[0] > 0:
                log.info("Calculating local chat token statistics...")
                local_token_stats = calculate_chat_token_stats(all_input_ids, self.tokenizer, log=None, is_main=False)
            elif "all_input_ids_list" in locals() and all_input_ids_list:
                log.info("Calculating local chat token statistics...")
                local_token_stats = calculate_chat_token_stats(
                    all_input_ids_list, self.tokenizer, log=None, is_main=False
                )

            # Calculate stats for selected positions
            if "all_token_roles" in locals() and all_token_roles:
                log.info("Calculating local selected token statistics...")
                local_selected_stats = calculate_selected_token_stats(all_token_roles, log=None, is_main=False)

        # In distributed mode, we need to gather all activations to compute global statistics
        log.info(f"Gathering {all_A.shape[0]} from process {rank}")
        if dist.is_initialized() and dist.get_world_size() > 1:
            # Gather all activations from all processes
            world_size = dist.get_world_size()
            print(f"Rank {rank} has {all_A.shape[0]} vectors")

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
            padded_A[: all_A.shape[0]] = all_A.to(self.device)
            padded_A_hat[: all_A_hat.shape[0]] = all_A_hat.to(self.device)
            padded_mses[: all_mses.shape[0]] = all_mses.to(self.device)

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
                all_A_list.append(gathered_A[i][: size.item()].cpu())
                all_A_hat_list.append(gathered_A_hat[i][: size.item()].cpu())
                all_mses_list.append(gathered_mses[i][: size.item()].cpu())

            all_A = torch.cat(all_A_list, dim=0).float().numpy()
            all_A_hat = torch.cat(all_A_hat_list, dim=0).float().numpy()
            all_mses = torch.cat(all_mses_list, dim=0).float().numpy()
        else:
            all_A = all_A.float().cpu().numpy()
            all_A_hat = all_A_hat.float().cpu().numpy()
            all_mses = all_mses.float().cpu().numpy()

        # Aggregate token statistics across all GPUs
        if "local_token_stats" in locals() and local_token_stats is not None:
            if dist.is_initialized() and dist.get_world_size() > 1:
                # Gather token counts from all ranks
                local_counts = local_token_stats["counts"]
                gathered_counts = [None] * world_size
                dist.all_gather_object(gathered_counts, local_counts)

                # Sum up counts from all ranks
                total_counts = {
                    "user": sum(counts["user"] for counts in gathered_counts),
                    "assistant": sum(counts["assistant"] for counts in gathered_counts),
                    "system": sum(counts["system"] for counts in gathered_counts),
                    "special": sum(counts["special"] for counts in gathered_counts),
                    "other": sum(counts["other"] for counts in gathered_counts),
                    "pad": sum(counts["pad"] for counts in gathered_counts),
                    "total": sum(counts["total"] for counts in gathered_counts),
                }

                # Calculate global proportions
                total = total_counts["total"]
                proportions = {}
                for role in ["user", "assistant", "system", "special", "other", "pad"]:
                    proportions[role] = total_counts[role] / total if total > 0 else 0.0

                # Update token_stats with aggregated data
                token_stats = {
                    "counts": total_counts,
                    "proportions": proportions,
                    "model_type": local_token_stats["model_type"],
                    "total_sequences": sum(local_token_stats.get("total_sequences", 0) for _ in gathered_counts),
                }

                # Log aggregated stats on rank 0
                if rank == 0 and log:
                    log.info("\n" + "=" * 50)
                    log.info("Aggregated Chat Token Statistics (All GPUs):")
                    log.info("=" * 50)
                    log.info(f"Model type: {token_stats['model_type']}")
                    log.info(f"Total sequences across all GPUs: {token_stats['total_sequences']}")
                    log.info(f"Total tokens: {total}")
                    log.info(f"User tokens: {total_counts['user']} ({proportions['user']:.1%})")
                    log.info(f"Assistant tokens: {total_counts['assistant']} ({proportions['assistant']:.1%})")
                    log.info(f"System tokens: {total_counts['system']} ({proportions['system']:.1%})")
                    log.info(f"Special tokens: {total_counts['special']} ({proportions['special']:.1%})")
                    log.info(f"Other tokens: {total_counts['other']} ({proportions['other']:.1%})")
                    if total_counts["pad"] > 0:
                        log.info(f"Pad tokens: {total_counts['pad']} ({proportions['pad']:.1%})")
                    log.info("=" * 50 + "\n")
            else:
                token_stats = local_token_stats

        # Aggregate selected token statistics across all GPUs
        selected_token_stats = None
        if "local_selected_stats" in locals() and local_selected_stats is not None:
            if dist.is_initialized() and dist.get_world_size() > 1:
                # Gather selected token counts from all ranks
                local_selected_counts = local_selected_stats["counts"]
                gathered_selected_counts = [None] * world_size
                dist.all_gather_object(gathered_selected_counts, local_selected_counts)

                # Sum up counts from all ranks
                total_selected_counts = {
                    "user": sum(counts["user"] for counts in gathered_selected_counts),
                    "assistant": sum(counts["assistant"] for counts in gathered_selected_counts),
                    "system": sum(counts["system"] for counts in gathered_selected_counts),
                    "special": sum(counts["special"] for counts in gathered_selected_counts),
                    "other": sum(counts["other"] for counts in gathered_selected_counts),
                    "pad": sum(counts["pad"] for counts in gathered_selected_counts),
                    "total": sum(counts["total"] for counts in gathered_selected_counts),
                }

                # Calculate global proportions
                total_selected = total_selected_counts["total"]
                selected_proportions = {}
                for role in ["user", "assistant", "system", "special", "other", "pad"]:
                    selected_proportions[role] = (
                        total_selected_counts[role] / total_selected if total_selected > 0 else 0.0
                    )

                # Update selected_token_stats with aggregated data
                selected_token_stats = {
                    "counts": total_selected_counts,
                    "proportions": selected_proportions,
                    "total_positions": total_selected,
                }

                # Log aggregated selected stats on rank 0
                if rank == 0 and log:
                    log.info("\n" + "=" * 50)
                    log.info("Aggregated Selected Token Statistics (All GPUs):")
                    log.info("=" * 50)
                    log.info(f"Total selected positions: {total_selected}")
                    log.info(f"User tokens: {total_selected_counts['user']} ({selected_proportions['user']:.1%})")
                    log.info(
                        f"Assistant tokens: {total_selected_counts['assistant']} ({selected_proportions['assistant']:.1%})"
                    )
                    log.info(f"System tokens: {total_selected_counts['system']} ({selected_proportions['system']:.1%})")
                    log.info(
                        f"Special tokens: {total_selected_counts['special']} ({selected_proportions['special']:.1%})"
                    )
                    log.info(f"Other tokens: {total_selected_counts['other']} ({selected_proportions['other']:.1%})")
                    if total_selected_counts["pad"] > 0:
                        log.info(f"Pad tokens: {total_selected_counts['pad']} ({selected_proportions['pad']:.1%})")
                    log.info("=" * 50 + "\n")
            else:
                selected_token_stats = local_selected_stats

        # Now calculate variance recovery on the full dataset
        print(
            f"Calculating variance recovered with {all_A.shape[0]} vectors from {all_A.shape[0] // num_positions} sequences."
        )

        # Check sample size and warn if small
        n_samples = all_A.shape[0]
        if n_samples < 1000:
            print(f"⚠️  Warning: Small sample size ({n_samples}). Variance recovery estimates may be biased.")

        # Compute variance explained (Equation 10 from https://arxiv.org/abs/2404.16014)
        # Using ddof=1 for unbiased variance estimation
        total_variance = np.var(all_A, axis=0, ddof=1).sum()
        residual_variance = np.var(all_A - all_A_hat, axis=0, ddof=1).sum()
        variance_recovery = 1 - (residual_variance / total_variance) if total_variance > 0 else 0.0
        mse_for_new_variance = np.mean((all_A - all_A_hat) ** 2, axis=0).sum()
        mse_variance_recovery = 1 - (mse_for_new_variance / total_variance) if total_variance > 0 else 0.0

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
            z_alpha = scipy.stats.norm.ppf([alpha / 2, 1 - alpha / 2])

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
        # Vector-wise norms for downstream histogram
        activation_norms = np.linalg.norm(all_A, axis=1)

        # R-squared
        ss_res = np.sum((all_A - all_A_hat) ** 2)
        ss_tot = np.sum((all_A - mean_A) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Calculate MSE breakdown by token role if available
        mse_by_role = {}
        variance_recovery_by_role = {}
        if "all_token_roles" in locals() and all_token_roles and compute_variance_by_role:
            # Group MSEs and activations by token role
            role_mses = {}
            role_activations = {}
            role_reconstructions = {}

            for i, (mse, role) in enumerate(zip(all_mses, all_token_roles)):
                if role not in role_mses:
                    role_mses[role] = []
                    role_activations[role] = []
                    role_reconstructions[role] = []
                role_mses[role].append(mse)
                role_activations[role].append(all_A[i])
                role_reconstructions[role].append(all_A_hat[i])

            # Calculate average MSE and variance recovery for each role
            for role, mses in role_mses.items():
                mse_by_role[role] = {"avg_mse": float(np.mean(mses)), "count": len(mses), "std": float(np.std(mses))}

                # Calculate variance recovery for this role
                if len(role_activations[role]) > 10:  # Only compute if we have enough samples
                    role_A = np.array(role_activations[role])
                    role_A_hat = np.array(role_reconstructions[role])

                    # Compute variance recovery for this role
                    role_total_variance = np.var(role_A, axis=0, ddof=1).sum()
                    role_residual_variance = np.var(role_A - role_A_hat, axis=0, ddof=1).sum()
                    role_variance_recovery = (
                        1 - (role_residual_variance / role_total_variance) if role_total_variance > 0 else 0.0
                    )

                    variance_recovery_by_role[role] = {
                        "variance_recovery": float(role_variance_recovery),
                        "total_variance": float(role_total_variance),
                        "residual_variance": float(role_residual_variance),
                        "count": len(role_activations[role]),
                    }

        result_dict = {
            "k": k,
            "avg_mse": float(np.mean(all_mses)),
            "variance_recovery": float(variance_recovery),
            "variance_recovery_bias": float(var_recovery_bias),
            "variance_recovery_bias_corrected": float(var_recovery_bias_corrected),
            "variance_recovery_std": float(var_recovery_std),
            "variance_recovery_ci_lower": float(var_recovery_ci[0]),
            "variance_recovery_ci_upper": float(var_recovery_ci[1]),
            "variance_recovery_bca_ci_lower": float(var_recovery_bca_ci[0]),
            "variance_recovery_bca_ci_upper": float(var_recovery_bca_ci[1]),
            "mse_std": float(mse_std),
            "mse_ci_lower": float(mse_ci[0]),
            "mse_ci_upper": float(mse_ci[1]),
            "r_squared": float(r_squared),
            "r_squared_std": float(r_squared_std),
            "r_squared_ci_lower": float(r_squared_ci[0]),
            "r_squared_ci_upper": float(r_squared_ci[1]),
            "total_positions": int(total_positions),
            "original_variance": float(var_A_direct),
            "original_mean": float(mean_A),
            "mean_variance_mean_over_hidden": np.var(all_A - all_A_hat, axis=0, ddof=1).mean(),
            "residual_variance_mean_over_hidden": np.var(all_A - all_A_hat, axis=0, ddof=1).mean(),
            # Extra data for histograms
            "mse_values": all_mses.tolist(),
            "activation_norms": activation_norms.tolist(),
            "mse_for_new_variance": float(mse_for_new_variance),
            "mse_variance_recovery": float(mse_variance_recovery),
        }

        # Add token stats if available
        if "token_stats" in locals() and token_stats is not None:
            result_dict["token_stats"] = token_stats

        # Add selected token stats if available
        if "selected_token_stats" in locals() and selected_token_stats is not None:
            result_dict["selected_token_stats"] = selected_token_stats

        # Add MSE by role if available
        if mse_by_role:
            result_dict["mse_by_role"] = mse_by_role

        # Add variance recovery by role if available
        if variance_recovery_by_role:
            result_dict["variance_recovery_by_role"] = variance_recovery_by_role

        return result_dict


def setup_distributed_eval(cfg: DictConfig):
    """Initialize distributed training environment."""
    rank, world_size, local_rank = init_distributed()
    setup_for_distributed(rank == 0)

    # Set device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if cfg.get("seed") is not None:
        set_seed(cfg.seed + rank)

    return rank, world_size, device


def prepare_val_dataloader(
    cfg: DictConfig,
    rank: int,
    world_size: int,
    device: torch.device,
    log=None,
    analyzer=None,
    max_val_samples_req=1000,
    dataloader_batch_size=32,
    do_not_extract_activations_val=True,
):
    """Prepare validation dataloader."""
    # Extract dataset paths from config with safe defaults
    dataset_cfg = cfg.get("dataset", {})

    activation_dir = dataset_cfg.get("activation_dir", "./data/SimpleStories_train")
    val_activation_dir = dataset_cfg.get("val_activation_dir", activation_dir)

    # Check if on-the-fly generation is enabled
    on_the_fly_cfg = dataset_cfg.get("on_the_fly", {})
    on_the_fly_enabled = on_the_fly_cfg.get("enabled", False)

    # For evaluation, we typically don't need on-the-fly generation if we have pre-computed activations
    # Check if we have a valid validation activation directory
    if on_the_fly_enabled and val_activation_dir and Path(val_activation_dir).exists():
        # If we have pre-computed validation activations, disable on-the-fly for evaluation
        if log and is_main():
            log.info(
                f"Found pre-computed validation activations at {val_activation_dir}. Disabling on-the-fly generation for evaluation."
            )
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
        if analyzer and hasattr(analyzer, "orig_model") and analyzer.orig_model is not None:
            orig_model_for_gen = analyzer.orig_model
            tokenizer_for_gen = analyzer.tokenizer
            generation_device = device
            rank_for_gen = rank
            world_size_for_gen = world_size
            if log and is_main():
                log.info("Using original model from analyzer for on-the-fly generation")
        else:
            # Load the original model and tokenizer for generation
            orig_model_name = cfg.get("orig_model_name", None)

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
                    torch_dtype=torch.bfloat16 if cfg.get("use_bf16", True) else torch.float32,
                    load_in_8bit=False,
                    base_to_use=None,  # Will load its own model
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
        max_val_samples_req=max_val_samples_req,
        log=log,
        orig_model_for_gen=orig_model_for_gen,
        tokenizer_for_gen=tokenizer_for_gen,
        generation_device=generation_device,
        rank=rank_for_gen,
        world_size=world_size_for_gen,
        samples_per_regeneration_cycle=None,  # Not needed for validation
        do_not_extract_activations_val=do_not_extract_activations_val,
    )

    if val_dataset is None or len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty")

    # Determine num_workers based on CPU count and world size
    num_dataloader_workers = 0
    if world_size > 0:
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            workers_per_gpu = cpu_count // world_size
            if workers_per_gpu >= 4:
                num_dataloader_workers = min(workers_per_gpu // 2, 8)  # Simplified heuristic
            elif workers_per_gpu >= 2:
                num_dataloader_workers = 1
    elif is_main():  # world_size is 0 or less, which is unusual.
        log.warning(f"Could not determine optimal num_workers (world_size={world_size}). Defaulting to 0.")

    if True:
        if num_dataloader_workers > 0 and is_main():
            log.info("On-the-fly generation enabled, overriding num_dataloader_workers to 0.")
        num_dataloader_workers = 0

    if is_main():
        log.info(f"Num DataLoader workers set to: {num_dataloader_workers}")

    log.info(f"Length of dataset: {len(val_dataset)}")
    val_loader = get_dataloader_for_distributed(
        val_dataset,
        batch_size=dataloader_batch_size,
        world_size=world_size,
        rank=rank,
        shuffle=False,  # No shuffle for validation
        collate_fn=collate,
        num_workers=num_dataloader_workers,
        pin_memory=True,
        persistent_workers=num_dataloader_workers > 0,
    )
    return val_loader, orig_model_for_gen


@hydra.main(version_base=None, config_path="../conf", config_name="gemma3_CHAT_27b_frozen_nopostfix")
def main(cfg: DictConfig) -> None:
    """Run best-of-K sweep analysis."""

    # Always ensure eval config exists and is a DictConfig
    if "eval" not in cfg or not isinstance(cfg.eval, DictConfig):
        raise ValueError("No eval config found in checkpoint or current config.")
        default_eval = {
            "checkpoint_path": cfg.get("checkpoint_path", ""),
            "k_values": [1, 4, 16],
            "batch_size": 32,
            "max_batches": None,
            "temperature": 1.0,
            "max_generation_batch_size": 32,
            "use_bf16": True,
            "output_dir": "eval_results",
            "output_file": "k_sweep_results.json",
            "num_positions": 1,
        }
        OmegaConf.set_struct(cfg, False)
        cfg.eval = OmegaConf.create(default_eval)
        OmegaConf.set_struct(cfg, True)
        raise ValueError("No eval config found in checkpoint or current config. Setting default eval config.")

    eval_cfg = cfg.eval
    cfg_orig = copy.deepcopy(cfg)
    # Setup distributed
    rank, world_size, device = setup_distributed_eval(cfg)
    is_main_process = is_main()

    if is_main_process:
        print("Starting best-of-K sweep analysis...")
        print(f"Checkpoint: {eval_cfg.get('checkpoint_path', 'Not specified')}")
        print(f"K values: {eval_cfg.get('k_values', [1, 4, 16])}")

    checkpoint_path_str = eval_cfg.get("checkpoint_path", "")
    if not checkpoint_path_str:
        raise ValueError(
            "checkpoint_path must be specified in eval config or on command line with +eval.checkpoint_path=/path/to/checkpoint.pt"
        )
    checkpoint_path = Path(checkpoint_path_str)
    if checkpoint_path.is_dir():
        # Find youngest file in the directory (recursively)
        all_files = [f for f in checkpoint_path.rglob("*") if f.is_file()]
        if not all_files:
            raise FileNotFoundError(f"No checkpoint files found in directory: {checkpoint_path}")
        youngest_file = max(all_files, key=lambda f: f.stat().st_mtime)
        checkpoint_path = youngest_file
        checkpoint_path_str = str(checkpoint_path)
        eval_cfg["checkpoint_path"] = checkpoint_path_str
        cfg.eval["checkpoint_path"] = checkpoint_path_str
        print(f"Using youngest checkpoint file: {checkpoint_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Merge checkpoint config with eval config, ensuring eval config is always present and up to date
    if isinstance(eval_cfg, DictConfig) and "eval" in eval_cfg:
        eval_cfg = eval_cfg.eval

    if "config" in ckpt:
        print("Checkpoint config found, merging with eval config")
        merged_cfg = OmegaConf.create(ckpt["config"])
        OmegaConf.set_struct(merged_cfg, False)
        if "eval" not in merged_cfg or not isinstance(merged_cfg.eval, DictConfig):
            merged_cfg.eval = OmegaConf.create({})
        for key, value in eval_cfg.items():
            merged_cfg.eval[key] = value
        if "dataset" not in merged_cfg:
            raise ValueError("No dataset config found in checkpoint or current config.")
        if "data" not in merged_cfg:
            merged_cfg.data = OmegaConf.create({"num_workers": 0})
        if "seed" in cfg:
            merged_cfg.seed = cfg.seed
        OmegaConf.set_struct(merged_cfg, True)
        cfg = merged_cfg
        eval_cfg = cfg.eval
    else:
        if "eval" not in cfg or not isinstance(cfg.eval, DictConfig):
            raise ValueError("No eval config found in checkpoint or current config.")
        eval_cfg = cfg.eval

    # --- Fix: ensure command-line overrides (e.g. +eval.max_val_samples=...) always take precedence ---
    # This is needed because OmegaConf merging can cause checkpoint config to override CLI args.
    # We want CLI args to always win, so we re-apply them after merging.
    # This is a workaround for Hydra/OmegaConf's merging order.

    # Re-apply any eval.* keys from the original cfg (which includes CLI overrides)
    # to the final eval_cfg, so they take precedence.
    eval_cfg = dict(eval_cfg)
    if "eval" in cfg_orig and isinstance(cfg_orig.eval, DictConfig):
        for k, v in cfg_orig.eval["eval"].items():
            eval_cfg[k] = v
        for k, v in cfg_orig.eval.items():
            if k == "eval":
                continue
            eval_cfg[k] = v
    cfg.eval = eval_cfg

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        log.addHandler(handler)
    max_val_samples_req = eval_cfg.get("max_val_samples", 100000)
    if is_main_process:
        log.info(f"old eval cfg: {eval_cfg}")
        log.info(f"new eval cfg: {cfg.eval}")

    if is_main_process:
        log.info(f"cfg: {cfg}")
    if is_main_process:
        log.info("\n")
        log.info(OmegaConf.to_yaml(cfg))
        log.info("\n")
    # Check if cache exists before creating dataloader
    should_skip_extraction = False
    cache_path = None
    cache_exists = False
    if eval_cfg.get("load_store", False) and eval_cfg.get("num_positions", 1) == 1:
        print("Loading from cache")
        cache_dir = Path(eval_cfg.get("cache_dir", "eval_results/vector_cache"))
        model_name = cfg.get("orig_model_name", cfg.get("model_name"))
        ckpt_temp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "config" in ckpt_temp:
            layer = ckpt_temp["config"]["layer_l"]
        else:
            raise ValueError(f"Layer not found in checkpoint {ckpt_temp.keys()}")
        if model_name and "dataset" in cfg:
            dataset_cfg_dict = OmegaConf.to_container(cfg.dataset, resolve=True)
            # Get sequence length from dataset config
            seq_len = dataset_cfg_dict.get("seq_len", 1024)  # Default to 1024 if not specified
            cache_path = get_cache_path(cache_dir, model_name, layer, seq_len, rank)
            if cache_path.exists():
                print(f"Cache exists at {cache_path}, will skip dataloader entirely")
                cache_exists = True

    if cache_exists:
        cached_data = load_cached_vectors(cache_path)
        if cached_data is None:
            raise RuntimeError(f"Cache at {cache_path} could not be loaded.")
        log.info(f"Using cached vectors from {cache_path} at rank {rank}")
        cached_all_A = cached_data["all_A"].to(device)
        cached_positions = cached_data["all_positions"].to(device)
        cached_token_ids = cached_data["all_token_ids"].to(device)
        cached_all_input_ids = cached_data["all_input_ids"].to(device)
        cached_samples = cached_all_A.shape[0] // eval_cfg.get("num_positions", 1)
        if eval_cfg.get("max_batches"):
            requested_samples = eval_cfg["max_batches"] * world_size
        else:
            requested_samples = max_val_samples_req
        if cached_samples > requested_samples:
            vectors_to_use = requested_samples * eval_cfg.get("num_positions", 1)
            log.info(
                f"Cache has {cached_all_A.shape[0]} vectors ({cached_samples} samples), using first {vectors_to_use} vectors ({requested_samples} samples)"
            )
            cached_all_A = cached_all_A[:vectors_to_use]
            cached_positions = cached_positions[:vectors_to_use]
            cached_token_ids = cached_token_ids[:vectors_to_use]
            cached_all_input_ids = cached_all_input_ids[:vectors_to_use]
        elif cached_samples == requested_samples:
            log.info(f"Using all {cached_all_A.shape[0]} cached vectors ({cached_samples} samples)")
        else:
            raise ValueError(
                f"Cache has {cached_all_A.shape[0]} vectors ({cached_samples} samples), but requested {requested_samples} samples"
            )
        orig_model_name_for_analyzer = None
        if cfg.get("orig_model_name", None) is not None:
            orig_model_name_for_analyzer = cfg.get("orig_model_name")

        orig_model_for_gen = OrigWrapper(
            orig_model_name_for_analyzer,
            torch_dtype=torch.bfloat16 if cfg.get("use_bf16", True) else torch.float32,
            load_in_8bit=False,
            base_to_use=None,  # Will load its own model
        )
        orig_model_for_gen.to(device)
        evaluator = BestOfKEvaluator(
            str(checkpoint_path),
            device,
            use_bf16=cfg.eval.get("use_bf16", True),
            orig_model_name=orig_model_name_for_analyzer,
            orig_model_wrapper=orig_model_for_gen,
            use_hard_encoder_string=cfg.eval.get("use_hard_encoder_string", False),
        )
        k_values = cfg.eval.get("k_values", None)
        all_results = []

        def to_serializable(obj):
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
                print(f"\nEvaluating with K={k}... and world size {world_size}")
            log.info(f"starting rank {rank}")
            results = evaluator.evaluate_with_k(
                cached_all_A,
                k=k,
                max_batches=cfg.eval.get("max_batches"),
                temperature=cfg.eval.get("temperature", 1.0),
                max_generation_batch_size=cfg.eval.get("max_generation_batch_size", 32),
                vector_extraction_batch_size=cfg.eval.get("vector_extraction_batch_size", 8),
                do_bootstrap=cfg.eval.get("do_bootstrap", False),
                load_store=cfg.eval.get("load_store", False),
                cache_dir=Path(cfg.eval.get("cache_dir", "eval_results/vector_cache")),
                model_name=cfg.get("orig_model_name", cfg.get("model_name")),
                dataset_cfg=OmegaConf.to_container(cfg.dataset, resolve=True) if "dataset" in cfg else {},
                rank=rank,
                max_val_samples=max_val_samples_req,
                log=log,
            )
            if is_main_process:
                all_results.append(results)
                if "mse_std" in results:
                    print(
                        f"K={k}: MSE={results['avg_mse']:.6f}±{results['mse_std']:.6f}, Variance Recovered={results['variance_recovery']:.4f}±{results['variance_recovery_std']:.4f} with 95% CI [{results['variance_recovery_ci_lower']:.4f}, {results['variance_recovery_ci_upper']:.4f}] and confidence interval for bias [{results['variance_recovery_bca_ci_lower']:.4f}, {results['variance_recovery_bca_ci_upper']:.4f}]"
                    )
                else:
                    print(
                        f"K={k}: MSE={results['avg_mse']:.6f}, Variance Recovered={results['variance_recovery']:.4f}, R²={results['r_squared']:.4f}"
                    )
                if "mse_variance_recovery" in results:
                    print(
                        f"K={k}:Variance Recovered={results['mse_variance_recovery']} with MSE for new variance={results['mse_for_new_variance']:.6f}"
                    )
            if dist.is_initialized() and world_size > 1:
                dist.barrier()
        if is_main_process:
            output_dir = Path(cfg.eval.get("output_dir", "eval_results"))
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / cfg.eval.get("output_file", "k_sweep_results.json")
            final_results = {
                "checkpoint_path": str(checkpoint_path),
                "eval_config": OmegaConf.to_container(cfg.eval, resolve=True),
                "results_by_k": all_results,
                "summary": {
                    "k_values": [float(k) for k in k_values],
                    "mse_values": [float(r["avg_mse"]) for r in all_results],
                    "variance_recovery_values": [float(r["variance_recovery"]) for r in all_results],
                    "r_squared_values": [float(r["r_squared"]) for r in all_results],
                    "new_variance_recovery_values": [float(r["mse_variance_recovery"]) for r in all_results],
                    "new_variance_recovery_mse_values": [float(r["mse_for_new_variance"]) for r in all_results],
                },
            }
            with open(output_file, "w") as f:
                json.dump(to_serializable(final_results), f, indent=2)
            print(f"\nResults saved to: {output_file}")
            print("\nSummary:")
            print("K\tVar Recovery (Raw)\tVar Recovery (Bias-Corrected)\tBias\t\tR²")
            print("-" * 100)
            for r in all_results:
                bias_corrected = r.get("variance_recovery_bias_corrected", r["variance_recovery"])
                bias = r.get("variance_recovery_bias", 0)
                print(
                    f"{r['k']}\t{r['variance_recovery']:.4f}±{r.get('variance_recovery_std', 0):.4f}\t\t"
                    f"{bias_corrected:.4f}\t\t\t{bias:.4f}\t"
                    f"{r['r_squared']:.4f}±{r.get('r_squared_std', 0):.4f}"
                )
            print("\n95% Confidence Intervals:")
            print("K\tVar Recovery CI (Percentile)\tVar Recovery CI (BCa)")
            print("-" * 80)
            for r in all_results:
                if "variance_recovery_ci_lower" in r:
                    print(
                        f"{r['k']}\t[{r['variance_recovery_ci_lower']:.4f}, {r['variance_recovery_ci_upper']:.4f}]\t\t"
                        f"[{r.get('variance_recovery_bca_ci_lower', r['variance_recovery_ci_lower']):.4f}, "
                        f"{r.get('variance_recovery_bca_ci_upper', r['variance_recovery_ci_upper']):.4f}]"
                    )
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    # ... existing code to construct dataloader and proceed as before ...
    OmegaConf.set_struct(cfg, False)
    if "dataset" not in cfg:
        cfg["dataset"] = OmegaConf.create({})
    if "on_the_fly" not in cfg["dataset"]:
        cfg["dataset"]["on_the_fly"] = OmegaConf.create({})
    cfg["dataset"]["on_the_fly"]["generation_batch_size"] = eval_cfg.get("dataloader_fwd_pass_batch_size", 16)
    if "data" not in cfg:
        cfg["data"] = OmegaConf.create({})
    cfg["data"]["batch_size"] = eval_cfg.get("dataloader_batch_size", 128)
    if is_main_process:
        log.info(f"Using max_val_samples: {max_val_samples_req}")
        log.info(f"Using generation_batch_size: {cfg['dataset']['on_the_fly']['generation_batch_size']}")
        log.info(f"Using dataloader_batch_size: {eval_cfg.get('dataloader_batch_size', 128)}")
    OmegaConf.set_struct(cfg, True)
    do_not_extract = eval_cfg.get("num_positions", 1) != 1 or should_skip_extraction
    log.info(f"preparing val loader  on rank {rank}")
    val_loader, orig_model_for_gen = prepare_val_dataloader(
        cfg,
        rank,
        world_size,
        device,
        log,
        analyzer=None,
        max_val_samples_req=max_val_samples_req,
        dataloader_batch_size=eval_cfg.get("dataloader_batch_size", 128),
        do_not_extract_activations_val=do_not_extract,
    )
    log.info(f"prepared val loader on rank {rank}")

    orig_model_name_for_analyzer = None
    if cfg.get("orig_model_name", None) is not None:
        orig_model_name_for_analyzer = cfg.get("orig_model_name")

    evaluator = BestOfKEvaluator(
        str(checkpoint_path),
        device,
        use_bf16=cfg.eval.get("use_bf16", True),
        orig_model_name=orig_model_name_for_analyzer,
        orig_model_wrapper=orig_model_for_gen,
        use_hard_encoder_string=cfg.eval.get("use_hard_encoder_string", False),
    )

    all_results = []
    k_values = cfg.eval.get("k_values", [1, 4, 16])

    def to_serializable(obj):
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
            print(f"\nEvaluating with K={k}... and world size {world_size}")
        log.info(f"starting rank {rank}")
        val_loader_iter = iter(val_loader)
        results = evaluator.evaluate_with_k(
            val_loader_iter,
            k=k,
            max_batches=cfg.eval.get("max_batches"),
            temperature=cfg.eval.get("temperature", 1.0),
            max_generation_batch_size=cfg.eval.get("max_generation_batch_size", 32),
            vector_extraction_batch_size=cfg.eval.get("vector_extraction_batch_size", 8),
            do_bootstrap=cfg.eval.get("do_bootstrap", False),
            load_store=cfg.eval.get("load_store", False),
            cache_dir=Path(cfg.eval.get("cache_dir", "eval_results/vector_cache")),
            model_name=cfg.get("orig_model_name", cfg.get("model_name")),
            dataset_cfg=OmegaConf.to_container(cfg.dataset, resolve=True) if "dataset" in cfg else {},
            rank=rank,
            max_val_samples=max_val_samples_req,
            log=log,
        )
        if is_main_process:
            all_results.append(results)
            if "mse_std" in results:
                print(
                    f"K={k}: MSE={results['avg_mse']:.6f}±{results['mse_std']:.6f}, Variance Recovered={results['variance_recovery']:.4f}±{results['variance_recovery_std']:.4f} with 95% CI [{results['variance_recovery_ci_lower']:.4f}, {results['variance_recovery_ci_upper']:.4f}] and confidence interval for bias [{results['variance_recovery_bca_ci_lower']:.4f}, {results['variance_recovery_bca_ci_upper']:.4f}]"
                )
            else:
                print(
                    f"K={k}: MSE={results['avg_mse']:.6f}, Variance Recovered={results['variance_recovery']:.4f}, R²={results['r_squared']:.4f}"
                )
        if dist.is_initialized() and world_size > 1:
            dist.barrier()

    # Save results
    if is_main_process:
        # Get output config
        output_dir = Path(cfg.eval.get("output_dir", "eval_results"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / cfg.eval.get("output_file", "k_sweep_results.json")

        final_results = {
            "checkpoint_path": str(checkpoint_path),
            "eval_config": OmegaConf.to_container(cfg.eval, resolve=True),
            "results_by_k": all_results,
            "summary": {
                "k_values": [float(k) for k in k_values],
                "mse_values": [float(r["avg_mse"]) for r in all_results],
                "variance_recovery_values": [float(r["variance_recovery"]) for r in all_results],
                "r_squared_values": [float(r["r_squared"]) for r in all_results],
            },
        }
        with open(output_file, "w") as f:
            json.dump(to_serializable(final_results), f, indent=2)
        print(f"\nResults saved to: {output_file}")
        print("\nSummary:")
        print("K\tVar Recovery (Raw)\tVar Recovery (Bias-Corrected)\tBias\t\tR²")
        print("-" * 100)
        for r in all_results:
            bias_corrected = r.get("variance_recovery_bias_corrected", r["variance_recovery"])
            bias = r.get("variance_recovery_bias", 0)
            print(
                f"{r['k']}\t{r['variance_recovery']:.4f}±{r.get('variance_recovery_std', 0):.4f}\t\t"
                f"{bias_corrected:.4f}\t\t\t{bias:.4f}\t"
                f"{r['r_squared']:.4f}±{r.get('r_squared_std', 0):.4f}"
            )

        # Print confidence intervals
        print("\n95% Confidence Intervals:")
        print("K\tVar Recovery CI (Percentile)\tVar Recovery CI (BCa)")
        print("-" * 80)
        for r in all_results:
            if "variance_recovery_ci_lower" in r:
                print(
                    f"{r['k']}\t[{r['variance_recovery_ci_lower']:.4f}, {r['variance_recovery_ci_upper']:.4f}]\t\t"
                    f"[{r.get('variance_recovery_bca_ci_lower', r['variance_recovery_ci_lower']):.4f}, "
                    f"{r.get('variance_recovery_bca_ci_upper', r['variance_recovery_ci_upper']):.4f}]"
                )

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
