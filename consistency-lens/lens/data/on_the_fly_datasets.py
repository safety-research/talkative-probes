from __future__ import annotations

import logging
import sys  # For RankInMemoryTrainingCache logging fallback
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Tuple  # Added Callable

import torch
import torch.distributed as dist
from datasets import Dataset as HFDataset
from datasets import load_from_disk  # To avoid conflict with torch.utils.data.Dataset
from torch.utils.data import Dataset  # Removed IterableDataset as we are making RankInMemoryTrainingCache map-style

from transformers import PreTrainedTokenizer
from lens.models.orig import OrigWrapper
# from omegaconf import DictConfig # If we pass the hydra dict directly

log = logging.getLogger(__name__)


def _dataset_log_fn(log_instance: logging.Logger, message: str, level: str = "info", rank: Optional[int] = None):
    prefix = f"[Rank {rank}] " if rank is not None else "[Rank N/A] "
    if level == "error": 
        log_instance.error(f"{prefix}{message}")
    elif level == "warning":
        log_instance.warning(f"{prefix}{message}")
    else:
        log_instance.info(f"{prefix}{message}")

def _generate_activations_batched(
    orig_model_for_gen: OrigWrapper,
    batch_input_ids_on_device: torch.Tensor, # Expected shape (batch_size, seq_len)
    layer_l: int,
    min_pos_to_select_from: int,
    attention_mask_on_device: Optional[torch.Tensor] = None,
    position_selection_strategy: str = 'random'
) -> Tuple[torch.Tensor, torch.Tensor]: # Returns (batch_activations, batch_token_positions)
    """
    Generates activations for a batch of input_ids using OrigWrapper.get_activations_at_positions.
    Assumes orig_model_for_gen.model is already on the same device as batch_input_ids_on_device.
    """
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

        self.data_store: List[Dict[str, Any]] = []
        # Ensure model is on device *before* generation.
        self.orig_model_for_gen.to(self.generation_device)
        _dataset_log_fn(log, f"Rank {self.rank} (ValDS): Moved orig_model_for_gen to {self.generation_device} for validation data generation.", rank=self.rank)
        self._generate_and_store_data()
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

        if num_samples_on_this_rank == 0:
            _dataset_log_fn(log, f"Rank {self.rank} has no samples assigned for validation after sharding. Skipping generation.", rank=self.rank)

            self.data_store = []
            return
            
        _dataset_log_fn(log, f"Rank {self.rank} will generate {num_samples_on_this_rank} validation samples from its shard (total unique: {actual_total_samples_to_consider}) using batch size {self.generation_batch_size}.", rank=self.rank)


        indices_A_all = list(range(num_samples_on_this_rank))
        indices_A_prime_all = indices_A_all[:]
        if num_samples_on_this_rank > 1:
            import random
            random.shuffle(indices_A_prime_all)
            for i in range(num_samples_on_this_rank):
                if indices_A_all[i] == indices_A_prime_all[i]:
                    swap_idx = (i + 1) % num_samples_on_this_rank
                    indices_A_prime_all[i], indices_A_prime_all[swap_idx] = (
                        indices_A_prime_all[swap_idx],
                        indices_A_prime_all[i],
                    )
        
        processed_count = 0
        with torch.no_grad():
            for batch_start_idx in range(0, num_samples_on_this_rank, self.generation_batch_size):
                batch_end_idx = min(batch_start_idx + self.generation_batch_size, num_samples_on_this_rank)
                current_batch_indices_A = indices_A_all[batch_start_idx:batch_end_idx]
                current_batch_indices_A_prime = indices_A_prime_all[batch_start_idx:batch_end_idx]

                batch_samples_A = [sharded_pretok_ds_for_rank[idx] for idx in current_batch_indices_A]
                batch_samples_A_prime = [sharded_pretok_ds_for_rank[idx] for idx in current_batch_indices_A_prime]

                if not batch_samples_A: continue

                batch_input_ids_A_list = [s["input_ids"] for s in batch_samples_A]
                batch_input_ids_A_prime_list = [s["input_ids"] for s in batch_samples_A_prime]

                def get_attn_mask(sample_batch: List[Dict[str, Any]], tokenizer: PreTrainedTokenizer) -> List[List[int]]:
                    masks = []
                    for s in sample_batch:
                        if "attention_mask" in s and s["attention_mask"] is not None:
                            masks.append(s["attention_mask"])
                        else:
                            masks.append((torch.tensor(s["input_ids"]) != tokenizer.pad_token_id).long().tolist())
                    return masks

                batch_attn_mask_A_list = get_attn_mask(batch_samples_A, self.tokenizer)
                batch_attn_mask_A_prime_list = get_attn_mask(batch_samples_A_prime, self.tokenizer)

                batch_input_ids_A_device = torch.tensor(batch_input_ids_A_list, dtype=torch.long, device=self.generation_device)
                batch_input_ids_A_prime_device = torch.tensor(batch_input_ids_A_prime_list, dtype=torch.long, device=self.generation_device)
                batch_attn_mask_A_device = torch.tensor(batch_attn_mask_A_list, dtype=torch.long, device=self.generation_device)
                batch_attn_mask_A_prime_device = torch.tensor(batch_attn_mask_A_prime_list, dtype=torch.long, device=self.generation_device)


                batch_act_A, batch_pos_A = _generate_activations_batched(self.orig_model_for_gen, batch_input_ids_A_device, self.layer_l, self.min_pos, batch_attn_mask_A_device, self.position_selection_strategy)
                batch_act_A_prime, batch_pos_A_prime = _generate_activations_batched(self.orig_model_for_gen, batch_input_ids_A_prime_device, self.layer_l, self.min_pos, batch_attn_mask_A_prime_device, self.position_selection_strategy)

                for i in range(batch_act_A.size(0)):
                    self.data_store.append({
                        "A": batch_act_A[i].cpu(),
                        "A_prime": batch_act_A_prime[i].cpu(),
                        "input_ids_A": torch.tensor(batch_input_ids_A_list[i], dtype=torch.long),
                        "input_ids_A_prime": torch.tensor(batch_input_ids_A_prime_list[i], dtype=torch.long),
                        "token_pos_A": batch_pos_A[i].item(),
                        "token_pos_A_prime": batch_pos_A_prime[i].item(),
                        "layer_idx": self.layer_l,
                    })
                    processed_count += 1
                
                if processed_count > 0 and processed_count % (self.generation_batch_size * 5) < self.generation_batch_size :
                     _dataset_log_fn(log, f"Generated {processed_count}/{num_samples_on_this_rank} val samples...", rank=self.rank)
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

        self.data_store: List[Dict[str, Any]] = []
        self.total_samples_in_pretokenised_dataset = 0
        
        # Load the pretokenized shard once
        self.pretok_dataset_shard: Optional[HFDataset] = self._load_pretok_shard()
        
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

    def regenerate_cache(self, num_samples_to_generate: int) -> None:
        """Clear cache and generate new samples."""
        self.data_store = []
        
        if self.pretok_dataset_shard is None or len(self.pretok_dataset_shard) == 0:
            self._log(f"Cannot regenerate cache, pretokenized data source is empty for rank {self.rank}.", "error")
            return
        
        shard_size = len(self.pretok_dataset_shard)
        self._log(f"Regenerating cache with {num_samples_to_generate} new samples from shard of {shard_size} samples.", "info")
        
        # Check if cycling will occur
        if num_samples_to_generate > shard_size:
            cycles_needed = num_samples_to_generate / shard_size
            self._log(
                f"WARNING: Cache size ({num_samples_to_generate}) exceeds shard size ({shard_size}). "
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
        
        # if self.orig_model_for_gen.model != se
        #     self.orig_model_for_gen.to(self.generation_device)
        
        shard_iter = iter(self.pretok_dataset_shard)
        generated_count = 0
        cycle_count = 0
        samples_in_current_cycle = 0
        
        with torch.no_grad():
            while generated_count < num_samples_to_generate:
                batch_samples = []
                for _ in range(min(self.generation_batch_size, num_samples_to_generate - generated_count)):
                    try:
                        sample = next(shard_iter)
                        samples_in_current_cycle += 1
                    except StopIteration:
                        # Dataset exhausted, cycling back to beginning
                        cycle_count += 1
                        if cycle_count == 1:
                            self._log(
                                f"Dataset shard exhausted after {samples_in_current_cycle} samples. "
                                f"Starting cycle #{cycle_count + 1}...", 
                                "warning"
                            )
                        elif cycle_count % 5 == 0:  # Log every 5 cycles to avoid spam
                            self._log(
                                f"Completed {cycle_count} cycles through dataset shard. "
                                f"Generated {generated_count}/{num_samples_to_generate} samples so far.",
                                "warning"
                            )
                        samples_in_current_cycle = 0
                        shard_iter = iter(self.pretok_dataset_shard)
                        sample = next(shard_iter)
                        samples_in_current_cycle += 1
                    batch_samples.append(sample)
                
                def get_attn_mask(sample_batch: List[Dict[str, Any]], tokenizer: PreTrainedTokenizer) -> List[List[int]]:
                    masks = []
                    for s in sample_batch:
                        if "attention_mask" in s and s["attention_mask"] is not None:
                            masks.append(s["attention_mask"])
                        else:
                            masks.append((torch.tensor(s["input_ids"]) != tokenizer.pad_token_id).long().tolist())
                    return masks

                batch_attn_mask_list = get_attn_mask(batch_samples, self.tokenizer)
                batch_input_ids_tensor = torch.tensor([s['input_ids'] for s in batch_samples], dtype=torch.long, device=self.generation_device)
                batch_attn_mask_tensor = torch.tensor(batch_attn_mask_list, dtype=torch.long, device=self.generation_device)

                batch_activations, batch_positions = _generate_activations_batched(
                    self.orig_model_for_gen, batch_input_ids_tensor, self.layer_l, self.min_pos, batch_attn_mask_tensor, self.position_selection_strategy
                )
                
                for i in range(len(batch_samples)):
                    self.data_store.append({
                        "A": batch_activations[i].cpu(),
                        "input_ids_A": torch.tensor(batch_samples[i]['input_ids'], dtype=torch.long),
                        "token_pos_A": batch_positions[i].item(),
                        "layer_idx": self.layer_l,
                    })
                    generated_count += 1
                    if progress_bar is not None:
                        progress_bar.update(1)
        if progress_bar is not None:
            progress_bar.close()
        
        # Final logging with cycle information
        if cycle_count > 0:
            self._log(
                f"Cache regeneration complete. Generated {len(self.data_store)} samples with {cycle_count} complete cycles "
                f"through the dataset shard (plus {samples_in_current_cycle} samples from partial cycle).",
                "info"
            )
        else:
            self._log(f"Cache regeneration complete. Generated {len(self.data_store)} samples without cycling.", "info")

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
        state['pretok_dataset_shard'] = None
        return state
