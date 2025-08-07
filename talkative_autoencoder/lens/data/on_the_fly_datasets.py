from __future__ import annotations

import logging
import random
import math
import sys  # For RankInMemoryTrainingCache logging fallback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
from datasets import Dataset as HFDataset
from datasets import load_from_disk  # To avoid conflict with torch.utils.data.Dataset
from torch.utils.data import Dataset  # Removed IterableDataset as we are making RankInMemoryTrainingCache map-style

from transformers import PreTrainedTokenizer
from lens.models.orig import OrigWrapper
# from omegaconf import DictConfig # If we pass the hydra dict directly

log = logging.getLogger(__name__)


def _get_attn_mask(
    sample_batch: List[Dict[str, Any]], tokenizer: PreTrainedTokenizer
) -> List[List[int]]:
    """Helper to get or create attention masks for a batch of samples."""
    masks = []
    for s in sample_batch:
        if "attention_mask" in s and s["attention_mask"] is not None:
            masks.append(
                (torch.tensor(s["input_ids"]) != tokenizer.pad_token_id)
                .long()
                .tolist()
            )
        else:
            # Create a mask assuming pad_token_id is defined
            masks.append(
                (torch.tensor(s["input_ids"]) != tokenizer.pad_token_id)
                .long()
                .tolist()
            )
    return masks


def _dataset_log_fn(log_instance: logging.Logger, message: str, level: str = "info", rank: Optional[int] = None):
    prefix = f"[Rank {rank}] " if rank is not None else "[Rank N/A] "
    if level == "error": 
        log_instance.error(f"{prefix}{message}")
    elif level == "warning":
        log_instance.warning(f"{prefix}{message}")
    else:
        log_instance.info(f"{prefix}{message}")

def _generate_activations_batched_multi(
    orig_model_for_gen: OrigWrapper,
    batch_input_ids_on_device: torch.Tensor, # Expected shape (batch_size, seq_len)
    layer_l: int,
    min_pos_to_select_from: int,
    vectors_per_sequence: int,
    attention_mask_on_device: Optional[torch.Tensor] = None,
    position_selection_strategy: str = 'random'
) -> Tuple[torch.Tensor, torch.Tensor]: # Returns (batch_activations, batch_token_positions)
    """
    Generates multiple activations per sequence for a batch of input_ids using a single forward pass.
    
    Args:
        orig_model_for_gen: The original model wrapper
        batch_input_ids_on_device: Input IDs tensor (batch_size, seq_len)
        layer_l: Layer index to extract from
        min_pos_to_select_from: Minimum position to consider
        vectors_per_sequence: Number of vectors to extract per sequence
        attention_mask_on_device: Optional attention mask
        position_selection_strategy: 'random' or 'midpoint'
    
    Returns:
        Tuple of (activations tensor, positions tensor)
        - activations: shape (batch_size, vectors_per_sequence, hidden_dim)
        - positions: shape (batch_size, vectors_per_sequence)
    """
    if vectors_per_sequence == 1:
        # Use the original single-position method for backward compatibility
        activations, positions = orig_model_for_gen.get_activations_at_positions(
            input_ids=batch_input_ids_on_device,
            layer_idx=layer_l,
            min_pos_to_select_from=min_pos_to_select_from,
            attention_mask=attention_mask_on_device,
            position_selection_strategy=position_selection_strategy,
            no_grad=True
        )
        # Reshape to (B, 1, hidden_dim) and (B, 1) for consistency
        return activations.unsqueeze(1), positions.unsqueeze(1)
    else:
        # Use the new efficient multi-position method
        return orig_model_for_gen.get_activations_at_multiple_positions(
            input_ids=batch_input_ids_on_device,
            layer_idx=layer_l,
            num_positions=vectors_per_sequence,
            min_pos_to_select_from=min_pos_to_select_from,
            attention_mask=attention_mask_on_device,
            position_selection_strategy=position_selection_strategy,
            no_grad=True
        )

# Keep the original function for backward compatibility
def _generate_activations_batched(
    orig_model_for_gen: OrigWrapper,
    batch_input_ids_on_device: torch.Tensor,
    layer_l: int,
    min_pos_to_select_from: int,
    attention_mask_on_device: Optional[torch.Tensor] = None,
    position_selection_strategy: str = 'random'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Original single-vector version for backward compatibility."""
    # OrigWrapper.get_activations_at_positions handles batching directly.
    
    selected_activations, calculated_token_positions = \
        orig_model_for_gen.get_activations_at_positions(
            input_ids=batch_input_ids_on_device,
            layer_idx=layer_l,
            min_pos_to_select_from=min_pos_to_select_from,
            attention_mask=attention_mask_on_device,
            position_selection_strategy=position_selection_strategy,
            no_grad=True
        )
    # selected_activations shape: (batch_size, hidden_dim)
    # calculated_token_positions shape: (batch_size,)
    return selected_activations, calculated_token_positions


class InMemoryValidationDataset(Dataset):
    def __init__(
        self,
        orig_model_for_gen: OrigWrapper,
        tokenizer: PreTrainedTokenizer,
        pretok_dataset_path: str,
        pretok_split_name: str,
        num_val_samples_to_generate: int, # This is TOTAL unique samples
        on_the_fly_config: Dict[str, Any],
        generation_device: torch.device,
        rank: int,
        world_size: int,
        do_not_extract_activations: bool = False
    ):
        self.orig_model_for_gen = orig_model_for_gen
        self.tokenizer = tokenizer
        self.pretok_dataset_path = pretok_dataset_path
        self.pretok_split_name = pretok_split_name
        # num_val_samples_to_generate is the total unique samples across ranks
        self.num_total_val_samples_to_consider = num_val_samples_to_generate
        self.config = on_the_fly_config
        self.generation_device = generation_device
        self.rank = rank
        self.world_size = world_size
        self.generation_batch_size = self.config.get('generation_batch_size', 32) # Default batch size
        self.layer_l = self.config['layer_l']
        self.min_pos = self.config['min_pos']
        self.position_selection_strategy = self.config.get('position_selection_strategy', 'random')
        self.do_not_extract_activations = do_not_extract_activations
        self.val_vectors_per_sequence = self.config.get('val_vectors_per_sequence', 1)
        self.shuffle_after_generation = self.config.get('shuffle_after_generation', True)
        
        # Norm filtering configuration
        self.filter_large_norms = self.config.get('filter_large_norms', False)
        self.max_activation_norm = self.config.get('max_activation_norm', 300000.0)

        self.data_store: List[Dict[str, Any]] = []
        # Ensure model is on device *before* generation.
        self.orig_model_for_gen.to(self.generation_device)
        _dataset_log_fn(log, f"Rank {self.rank} (ValDS): Moved orig_model_for_gen to {self.generation_device} for validation data generation.", rank=self.rank)
        if self.filter_large_norms:
            _dataset_log_fn(log, f"Rank {self.rank} (ValDS): Norm filtering enabled with max norm threshold: {self.max_activation_norm}", rank=self.rank)
        self._generate_and_store_data()
        
        # Shuffle if requested
        if self.shuffle_after_generation and len(self.data_store) > 0:
            import random
            random.shuffle(self.data_store)
            _dataset_log_fn(log, f"Rank {self.rank} shuffled {len(self.data_store)} validation samples.", rank=self.rank)
        
        _dataset_log_fn(log, f"Rank {self.rank} finished generating {len(self.data_store)} validation samples for its shard.", rank=self.rank)

    def _generate_and_store_data(self):
        """
        Generates data on this rank by processing its shard of the 
        pretokenized validation dataset in batches.
        """
        try:
            full_pretok_path = Path(self.pretok_dataset_path) / self.pretok_split_name
            if not full_pretok_path.exists():
                _dataset_log_fn(log, f"Pretokenized validation data not found at: {full_pretok_path}", "error", rank=self.rank)
                self.data_store = []
                return
            
            full_pretok_ds: HFDataset = load_from_disk(str(full_pretok_path))
        except Exception as e:
            _dataset_log_fn(log, f"Failed to load pretokenized validation dataset from {self.pretok_dataset_path}/{self.pretok_split_name}: {e}", "error", rank=self.rank)

            self.data_store = []
            return

        num_total_samples_in_split = len(full_pretok_ds)
        
        target_total_samples = self.num_total_val_samples_to_consider
        if target_total_samples <= 0:
             target_total_samples = num_total_samples_in_split
        
        actual_total_samples_to_consider = min(target_total_samples, num_total_samples_in_split)

        if actual_total_samples_to_consider < target_total_samples and target_total_samples != num_total_samples_in_split :
            _dataset_log_fn(log, f"Requested {target_total_samples} total validation samples, but split '{self.pretok_split_name}' only has {num_total_samples_in_split}. Considering {actual_total_samples_to_consider} samples in total.", "warning", rank=self.rank)

        
        if actual_total_samples_to_consider == 0:
            _dataset_log_fn(log, f"No validation samples will be generated; effective number of samples to consider from '{self.pretok_split_name}' is 0.", "warning", rank=self.rank)

            self.data_store = []
            return

        dataset_to_process = full_pretok_ds.select(range(actual_total_samples_to_consider))

        if self.world_size > 1:
            sharded_pretok_ds_for_rank = dataset_to_process.shard(
                num_shards=self.world_size, index=self.rank, contiguous=True
            )
        else:
            sharded_pretok_ds_for_rank = dataset_to_process
        
        num_samples_on_this_rank = len(sharded_pretok_ds_for_rank)
        total_vectors_on_this_rank = num_samples_on_this_rank * self.val_vectors_per_sequence

        if num_samples_on_this_rank == 0:
            _dataset_log_fn(log, f"Rank {self.rank} has no samples assigned for validation after sharding. Skipping generation.", rank=self.rank)

            self.data_store = []
            return
            
        _dataset_log_fn(log, f"Rank {self.rank} will generate {total_vectors_on_this_rank} validation vectors ({num_samples_on_this_rank} sequences x {self.val_vectors_per_sequence} vectors/seq) from its shard (total unique: {actual_total_samples_to_consider}) using batch size {self.generation_batch_size}.", rank=self.rank)

        # Track discarded samples
        discarded_count = 0
        processed_count = 0
        total_generated = 0
        
        # We need to keep track of indices for A/A' pairing
        available_indices = list(range(len(sharded_pretok_ds_for_rank)))
        current_idx = 0
        
        with torch.no_grad():
            while len(self.data_store) < total_vectors_on_this_rank and current_idx < len(available_indices):
                # Calculate how many more samples we need
                activations_left = total_vectors_on_this_rank - len(self.data_store)
                remaining_needed = math.ceil(activations_left / self.val_vectors_per_sequence)
                batch_size = min(self.generation_batch_size, remaining_needed, len(available_indices) - current_idx)
                
                if batch_size == 0:
                    break
                
                # Get batch indices
                batch_indices = available_indices[current_idx:current_idx + batch_size]
                current_idx += batch_size
                
                # Generate A' indices ensuring they're different from A
                batch_indices_prime = batch_indices[:]
                if len(batch_indices) > 1:
                    random.shuffle(batch_indices_prime)
                    for i in range(len(batch_indices)):
                        if batch_indices[i] == batch_indices_prime[i]:
                            swap_idx = (i + 1) % len(batch_indices)
                            batch_indices_prime[i], batch_indices_prime[swap_idx] = (
                                batch_indices_prime[swap_idx],
                                batch_indices_prime[i],
                            )
                
                batch_samples_A = [sharded_pretok_ds_for_rank[idx] for idx in batch_indices]
                batch_samples_A_prime = [sharded_pretok_ds_for_rank[idx] for idx in batch_indices_prime]

                if not batch_samples_A: continue

                batch_input_ids_A_list = [s["input_ids"] for s in batch_samples_A]
                batch_input_ids_A_prime_list = [s["input_ids"] for s in batch_samples_A_prime]

                batch_attn_mask_A_list = _get_attn_mask(batch_samples_A, self.tokenizer)
                batch_attn_mask_A_prime_list = _get_attn_mask(batch_samples_A_prime, self.tokenizer)

                batch_input_ids_A_device = torch.tensor(batch_input_ids_A_list, dtype=torch.long, device=self.generation_device)
                batch_input_ids_A_prime_device = torch.tensor(batch_input_ids_A_prime_list, dtype=torch.long, device=self.generation_device)
                batch_attn_mask_A_device = torch.tensor(batch_attn_mask_A_list, dtype=torch.long, device=self.generation_device)
                batch_attn_mask_A_prime_device = torch.tensor(batch_attn_mask_A_prime_list, dtype=torch.long, device=self.generation_device)

                if batch_input_ids_A_device.numel() == 0 or batch_attn_mask_A_device.numel() == 0:
                    _dataset_log_fn(log, f"Empty tensor detected in batch. Skipping batch.", "warning", rank=self.rank)
                    continue
                if batch_input_ids_A_prime_device.numel() == 0 or batch_attn_mask_A_prime_device.numel() == 0:
                    _dataset_log_fn(log, f"Empty tensor detected in batch. Skipping batch.", "warning", rank=self.rank)
                    continue
                
                if not self.do_not_extract_activations:
                    # Generate multiple vectors per sequence
                    batch_acts_A, batch_poss_A = _generate_activations_batched_multi(
                        self.orig_model_for_gen, batch_input_ids_A_device, 
                        self.layer_l, self.min_pos, self.val_vectors_per_sequence,
                        batch_attn_mask_A_device, self.position_selection_strategy
                    )
                    batch_acts_A_prime, batch_poss_A_prime = _generate_activations_batched_multi(
                        self.orig_model_for_gen, batch_input_ids_A_prime_device,
                        self.layer_l, self.min_pos, self.val_vectors_per_sequence,
                        batch_attn_mask_A_prime_device, self.position_selection_strategy
                    )
                else:
                    # Create dummy tensors with the right shape
                    batch_acts_A = torch.empty((len(batch_input_ids_A_list), self.val_vectors_per_sequence, 0))
                    batch_acts_A_prime = torch.empty((len(batch_input_ids_A_list), self.val_vectors_per_sequence, 0))
                    batch_poss_A = torch.empty((len(batch_input_ids_A_list), self.val_vectors_per_sequence), dtype=torch.long)
                    batch_poss_A_prime = torch.empty((len(batch_input_ids_A_list), self.val_vectors_per_sequence), dtype=torch.long)

                # Store each vector as a separate sample
                for vec_idx in range(self.val_vectors_per_sequence):
                    for i in range(len(batch_input_ids_A_list)):
                        total_generated += 1
                        
                        # Check norms if filtering is enabled
                        if self.filter_large_norms and not self.do_not_extract_activations:
                            norm_A = torch.norm(batch_acts_A[i, vec_idx]).item()
                            norm_A_prime = torch.norm(batch_acts_A_prime[i, vec_idx]).item()
                            
                            if norm_A > self.max_activation_norm or norm_A_prime > self.max_activation_norm:
                                discarded_count += 1
                                if discarded_count % 10 == 0:
                                    _dataset_log_fn(log, f"Discarded {discarded_count} samples so far due to large norms (threshold: {self.max_activation_norm})", rank=self.rank)
                                continue
                        
                        self.data_store.append({
                            "A": batch_acts_A[i, vec_idx].cpu() if not self.do_not_extract_activations else None,
                            "A_prime": batch_acts_A_prime[i, vec_idx].cpu() if not self.do_not_extract_activations else None,
                            "input_ids_A": torch.tensor(batch_input_ids_A_list[i], dtype=torch.long),
                            "input_ids_A_prime": torch.tensor(batch_input_ids_A_prime_list[i], dtype=torch.long),
                            "token_pos_A": batch_poss_A[i, vec_idx].item() if not self.do_not_extract_activations else None,
                            "token_pos_A_prime": batch_poss_A_prime[i, vec_idx].item() if not self.do_not_extract_activations else None,
                            "layer_idx": self.layer_l,
                            "vector_idx": vec_idx,  # Track which vector this is
                            "source_seq_idx": batch_indices[i],  # Track source sequence
                        })
                        processed_count += 1
                
                if processed_count > 0 and processed_count % (self.generation_batch_size * 5) < self.generation_batch_size:
                    _dataset_log_fn(log, f"Generated {processed_count}/{total_vectors_on_this_rank} val vectors (discarded {discarded_count} due to large norms)...", rank=self.rank)
        
        if self.filter_large_norms and discarded_count > 0:
            _dataset_log_fn(log, f"Validation generation complete. Generated {len(self.data_store)} samples, discarded {discarded_count} samples with norms > {self.max_activation_norm} (total generated: {total_generated})", rank=self.rank)
        
        return len(self.data_store)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not self.data_store:
            raise IndexError(
                "Dataset is empty. Generation failed or no samples were specified."
            )
        return self.data_store[idx]

    def __len__(self) -> int:
        return len(self.data_store)

    def __getstate__(self):
        """Drop heavy / non-picklable objects when the dataset is pickled for
        DataLoader worker processes.  Workers only need the already-built
        CPU `data_store`."""
        st = self.__dict__.copy()
        st['orig_model_for_gen'] = None      # avoid CUDA tensors & huge model
        st['tokenizer'] = None               # not needed in workers
        return st

# --- Training Dataset Components ---
#from lens.training.train_aux import _dataset_log_fn

# Map-style Dataset for on-the-fly training data cache per rank
class RankInMemoryTrainingCache(Dataset):
    def __init__(
        self,
        orig_model_for_gen: OrigWrapper, 
        tokenizer: PreTrainedTokenizer,
        pretok_dataset_path: str,
        pretok_split_name: str,
        on_the_fly_config: Dict[str, Any],
        generation_device: torch.device, 
        rank: int,
        world_size: int,
        initial_cache_size: int,
        logger: logging.Logger,
    ):
        self.orig_model_for_gen = orig_model_for_gen
        self.tokenizer = tokenizer
        self.pretok_dataset_path = pretok_dataset_path
        self.pretok_split_name = pretok_split_name
        self.config = on_the_fly_config
        self.generation_device = generation_device
        self.rank = rank
        self.world_size = world_size
        self.log = logger
        
        self.layer_l = self.config['layer_l']
        self.min_pos = self.config['min_pos']
        self.generation_batch_size = self.config.get('generation_batch_size', 32)
        self.position_selection_strategy = self.config.get('position_selection_strategy', 'random')
        self.vectors_per_sequence = self.config.get('vectors_per_sequence', 1)
        self.shuffle_after_generation = self.config.get('shuffle_after_generation', True)
        logger.info("Position selection strategy: %s", self.position_selection_strategy)
        logger.info("Vectors per sequence: %s", self.vectors_per_sequence)
        
        # Norm filtering configuration
        self.filter_large_norms = self.config.get('filter_large_norms', False)
        self.max_activation_norm = self.config.get('max_activation_norm', 300000.0)
        
        if self.filter_large_norms:
            logger.info("Norm filtering enabled with max norm threshold: %s", self.max_activation_norm)

        self.data_store: List[Dict[str, Any]] = []
        self.total_samples_in_pretokenised_dataset = 0
        
        # Track ordering across regenerations
        self._shuffled_indices: List[int] = []  # holds a permutation of shard indices
        self._cursor: int = 0  # next unread position in _shuffled_indices
        
        # Load the pretokenized shard once
        self.pretok_dataset_shard: Optional[HFDataset] = self._load_pretok_shard()
        
        # Initialise index order if shard loaded
        if self.pretok_dataset_shard is not None:
            self._shuffled_indices = list(range(len(self.pretok_dataset_shard)))
            random.shuffle(self._shuffled_indices)
        
        # Initialize with data if requested
        if initial_cache_size > 0 and self.pretok_dataset_shard is not None:
            self.regenerate_cache(num_samples_to_generate=initial_cache_size)

    def _log(self, message: str, level: str = "info"):
        _dataset_log_fn(self.log, message, level, rank=self.rank)

    def _load_pretok_shard(self) -> Optional[HFDataset]:
        try:
            full_pretok_path = Path(self.pretok_dataset_path) / self.pretok_split_name
            if not full_pretok_path.exists():
                self._log(f"Pretokenized training data not found at: {full_pretok_path}", "error")
                return None
            
            dataset_to_shard: HFDataset = load_from_disk(str(full_pretok_path))
            self.total_samples_in_pretokenised_dataset = len(dataset_to_shard)
            
            if self.world_size > 1:
                shard = dataset_to_shard.shard(
                    num_shards=self.world_size, index=self.rank, contiguous=True
                )
            else:
                shard = dataset_to_shard
            
            self._log(f"Loaded pretokenized training data. Shard size for rank {self.rank}: {len(shard)} samples.")
            return shard
        except Exception as e:
            self._log(f"Failed to load/shard pretokenized training dataset: {e}", "error")
            return None

    def _next_indices(self, n: int) -> List[int]:
        """Return the next *n* indices from the shard, reshuffling when exhausted."""
        if self.pretok_dataset_shard is None or n <= 0:
            return []

        out: List[int] = []
        while len(out) < n:
            remaining = len(self._shuffled_indices) - self._cursor
            if remaining == 0:
                random.shuffle(self._shuffled_indices)
                self._cursor = 0
                remaining = len(self._shuffled_indices)

            take = min(n - len(out), remaining)
            out.extend(self._shuffled_indices[self._cursor : self._cursor + take])
            self._cursor += take
        return out

    def regenerate_cache(self, num_samples_to_generate: int) -> None:
        """Modified to support multiple vectors per sequence and shuffling."""
        self.data_store = []
        
        if self.pretok_dataset_shard is None or len(self.pretok_dataset_shard) == 0:
            self._log(f"Cannot regenerate cache, pretokenized data source is empty for rank {self.rank}.", "error")
            return
        
        shard_size = len(self.pretok_dataset_shard)
        # Adjust for multiple vectors per sequence
        effective_samples_needed = num_samples_to_generate // self.vectors_per_sequence
        self._log(f"Regenerating cache with {num_samples_to_generate} samples ({effective_samples_needed} sequences × {self.vectors_per_sequence} vectors) from shard of {shard_size} sequences.", "info")
        
        # Check if cycling will occur
        if effective_samples_needed > shard_size:
            cycles_needed = effective_samples_needed / shard_size
            self._log(
                f"WARNING: Effective sequences needed ({effective_samples_needed}) exceeds shard size ({shard_size}). "
                f"Will cycle through dataset ~{cycles_needed:.1f} times. "
                f"This may lead to overfitting on repeated data.", 
                "warning"
            )
        
        # Only import tqdm if needed
        use_tqdm = self.rank == 0
        if use_tqdm:
            from tqdm import tqdm
            progress_bar = tqdm(total=num_samples_to_generate, desc="Generating cache", leave=True)
        else:
            progress_bar = None
        
        generated_count = 0
        discarded_count = 0
        total_attempts = 0
        
        with torch.no_grad():
            while generated_count < num_samples_to_generate:
                # Get more indices if we need to keep generating
                needed_samples = num_samples_to_generate - generated_count
                # Request extra indices to account for potential discards
                extra_factor = 1.5 if self.filter_large_norms else 1.0
                sequences_needed = math.ceil(needed_samples / self.vectors_per_sequence)
                indices_to_request = int(sequences_needed * extra_factor)
                indices_to_use = self._next_indices(indices_to_request)
                
                idx_cursor = 0
                while generated_count < num_samples_to_generate and idx_cursor < len(indices_to_use):
                    batch_end = min(idx_cursor + self.generation_batch_size, len(indices_to_use))
                    batch_indices = indices_to_use[idx_cursor:batch_end]
                    idx_cursor = batch_end
                    
                    batch_samples = [self.pretok_dataset_shard[i] for i in batch_indices]  # type: ignore
                    
                    batch_attn_mask_list = _get_attn_mask(batch_samples, self.tokenizer)
                    batch_input_ids_tensor = torch.tensor([s['input_ids'] for s in batch_samples], dtype=torch.long, device=self.generation_device)
                    batch_attn_mask_tensor = torch.tensor(batch_attn_mask_list, dtype=torch.long, device=self.generation_device)
                    
                    if batch_input_ids_tensor.numel() == 0 or batch_attn_mask_tensor.numel() == 0:
                        self._log(f"Empty tensor detected in batch {batch_indices}. Skipping batch.", "warning")
                        continue

                    # Generate multiple vectors per sequence
                    batch_activations, batch_positions = _generate_activations_batched_multi(
                        self.orig_model_for_gen, batch_input_ids_tensor, 
                        self.layer_l, self.min_pos, self.vectors_per_sequence,
                        batch_attn_mask_tensor, self.position_selection_strategy
                    )
                    
                    # Store each vector as a separate sample
                    for vec_idx in range(self.vectors_per_sequence):
                        for i in range(len(batch_samples)):
                            total_attempts += 1
                            
                            # Check norm if filtering is enabled
                            if self.filter_large_norms:
                                norm = torch.norm(batch_activations[i, vec_idx]).item()
                                if norm > self.max_activation_norm:
                                    discarded_count += 1
                                    if discarded_count % 100 == 0:
                                        self._log(f"Discarded {discarded_count} samples so far due to large norms (threshold: {self.max_activation_norm})")
                                    continue
                            
                            self.data_store.append({
                                "A": batch_activations[i, vec_idx].cpu(),
                                "input_ids_A": torch.tensor(batch_samples[i]['input_ids'], dtype=torch.long),
                                "token_pos_A": batch_positions[i, vec_idx].item(),
                                "layer_idx": self.layer_l,
                                "vector_idx": vec_idx,  # Track which vector this is
                                "source_seq_idx": batch_indices[i],  # Track source sequence
                            })
                            generated_count += 1
                            if progress_bar is not None:
                                progress_bar.update(1)
                            
                            if generated_count >= num_samples_to_generate:
                                break
                    
                    if generated_count >= num_samples_to_generate:
                        break
                
                # Safety check to prevent infinite loops
                if total_attempts > num_samples_to_generate * 10:
                    self._log(f"WARNING: Stopping generation after {total_attempts} attempts. Only generated {generated_count}/{num_samples_to_generate} samples.", "warning")
                    break
                    
        if progress_bar is not None:
            progress_bar.close()
        
        # Shuffle the cache if requested
        if self.shuffle_after_generation and len(self.data_store) > 0:
            import random
            random.shuffle(self.data_store)
            self._log(f"Shuffled {len(self.data_store)} training samples after generation.")
        
        # Final logging with cycle and discard information
        if self.filter_large_norms and discarded_count > 0:
            self._log(f"Cache regeneration complete. Generated {len(self.data_store)} samples, discarded {discarded_count} samples with norms > {self.max_activation_norm} (total attempts: {total_attempts})")
        else:
            self._log(f"Cache regeneration complete. Generated {len(self.data_store)} samples.")

    def __len__(self) -> int:
        return len(self.data_store)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self.data_store):
            raise IndexError(f"Index {idx} out of range for cache size {len(self.data_store)}")
        return self.data_store[idx]

    def __getstate__(self):
        """Strip heavy / non-picklable fields when DataLoader workers pickle us."""
        state = self.__dict__.copy()
        state['orig_model_for_gen'] = None
        state['tokenizer'] = None
        state['pretok_dataset_shard'] = None
        return state
