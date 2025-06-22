from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Iterator

import torch
from torch.utils.data import Dataset
from datasets import load_from_disk, Dataset as HFDataset # To avoid conflict with torch.utils.data.Dataset
import torch.distributed as dist


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from lens.models.orig import OrigWrapper
    # from omegaconf import DictConfig # If we pass the hydra dict directly

log = logging.getLogger(__name__)


def _calculate_token_pos(
    input_ids_tensor: torch.Tensor, # Shape (seq_len) or (1, seq_len)
    min_pos_to_select_from: int,
    tokenizer_pad_id: Optional[int],
) -> int:
    """
    Calculates the token position for activation extraction, similar to 00_dump_activations.
    Selects a position roughly in the middle of the non-padding tokens,
    respecting min_pos_to_select_from.
    """
    if input_ids_tensor.ndim == 2:
        input_ids_tensor = input_ids_tensor.squeeze(0) # Ensure 1D

    if tokenizer_pad_id is not None:
        non_pad_mask = input_ids_tensor.ne(tokenizer_pad_id)
        non_pad_len = non_pad_mask.sum().item()
    else:
        # Assume no padding if pad_token_id is None
        non_pad_len = input_ids_tensor.size(0)

    # Upper bound for selection is the last actual token
    upper_bound = max(0, non_pad_len - 1)

    # Ensure min_pos is not beyond the actual sequence length
    effective_min_pos = min(min_pos_to_select_from, upper_bound)

    # Calculate the middle position between effective_min_pos and upper_bound
    # This aims for a position after min_pos_to_select_from, towards the end of the content.
    token_pos = (effective_min_pos + upper_bound) // 2

    # Ensure token_pos is at least 0, even if sequence is very short or all padding
    token_pos = max(0, token_pos)
    return int(token_pos)


def _generate_single_activation(
    orig_model_for_gen: OrigWrapper,
    input_ids_on_device: torch.Tensor, # Expected to be on the correct device, shape (1, seq_len)
    layer_l: int,
    min_pos_to_select_from: int,
    tokenizer_pad_id: Optional[int],
) -> Tuple[torch.Tensor, int]:
    """
    Generates a single activation from the original model.
    Assumes orig_model_for_gen.model is already on the same device as input_ids_on_device.
    """
    if input_ids_on_device.ndim == 1:
        input_ids_on_device = input_ids_on_device.unsqueeze(0) # Ensure (1, seq_len)

    with torch.no_grad():
        # The OrigWrapper's model might be a DDP model if not handled carefully.
        # Ensure we're calling the underlying model directly if it's wrapped.
        model_to_use = orig_model_for_gen.model
        if hasattr(model_to_use, 'module'): # Check for DDP wrapper
            model_to_use = model_to_use.module

        outputs = model_to_use(input_ids_on_device, output_hidden_states=True)
    
    # hidden_states is a tuple of (embedding_output, layer1_output, layer2_output, ...)
    # So layer_l output is at index layer_l + 1
    # Shape: (1, seq_len, hidden_dim)
    hidden_state_at_layer = outputs.hidden_states[layer_l + 1]

    token_pos = _calculate_token_pos(
        input_ids_on_device.squeeze(0), # Pass 1D tensor
        min_pos_to_select_from,
        tokenizer_pad_id
    )

    # Extract activation: Shape (hidden_dim)
    activation = hidden_state_at_layer[0, token_pos, :].squeeze()
    return activation, token_pos


class InMemoryValidationDataset(Dataset):
    def __init__(
        self,
        orig_model_for_gen: OrigWrapper,
        tokenizer: PreTrainedTokenizer,
        pretok_dataset_path: str,
        pretok_split_name: str,
        num_val_samples_to_generate: int,
        on_the_fly_config: Dict[str, Any], # Contains layer_l, min_pos etc.
        generation_device: torch.device,
        rank: int,
        world_size: int,
    ):
        self.orig_model_for_gen = orig_model_for_gen
        self.tokenizer = tokenizer
        self.pretok_dataset_path = pretok_dataset_path
        self.pretok_split_name = pretok_split_name
        self.num_val_samples_to_generate = num_val_samples_to_generate
        self.config = on_the_fly_config # e.g., self.config['layer_l'], self.config['min_pos']
        self.generation_device = generation_device
        self.rank = rank
        self.world_size = world_size
        
        self.data_store: List[Dict[str, Any]] = []

        # Ensure model is on the generation device before use
        # This is crucial. The model passed here should be the one designated for data generation.
        self.orig_model_for_gen.to(self.generation_device)

        if self.rank == 0:
            log.info(f"Rank 0 generating {self.num_val_samples_to_generate} validation samples on device {self.generation_device}...")
            self._generate_and_store_data()
            log.info(f"Rank 0 finished generating {len(self.data_store)} validation samples.")
        
        if self.world_size > 1:
            # Synchronize the data_store across all ranks
            # Convert list of dicts to a broadcastable object if necessary, or broadcast element by element.
            # For simplicity, we can use torch.distributed.broadcast_object_list.
            # This requires the objects to be picklable. Tensors should be fine.
            
            # Rank 0 prepares the list to be broadcasted
            object_list_to_broadcast = [self.data_store] if self.rank == 0 else [None]
            
            # Perform the broadcast operation
            # Ensure all processes participate in this call
            dist.broadcast_object_list(object_list_to_broadcast, src=0)
            
            if self.rank != 0:
                self.data_store = object_list_to_broadcast[0] # type: ignore
                log.info(f"Rank {self.rank} received {len(self.data_store)} validation samples from rank 0.")
        
        if not self.data_store and self.num_val_samples_to_generate > 0:
            # This can happen if rank 0 generated no data and then broadcasted an empty list
             log.warning(f"Rank {self.rank}: Validation data store is empty after generation/broadcast. num_val_samples_to_generate was {self.num_val_samples_to_generate}")


    def _generate_and_store_data(self):
        """Generates data on rank 0."""
        try:
            full_pretok_path = Path(self.pretok_dataset_path) / self.pretok_split_name
            if not full_pretok_path.exists():
                log.error(f"Pretokenized validation data not found at: {full_pretok_path}")
                self.data_store = []
                return

            pretok_ds: HFDataset = load_from_disk(str(full_pretok_path))
        except Exception as e:
            log.error(f"Failed to load pretokenized validation dataset from {self.pretok_dataset_path}/{self.pretok_split_name}: {e}")
            self.data_store = []
            return

        samples_to_actually_generate = min(self.num_val_samples_to_generate, len(pretok_ds))
        if samples_to_actually_generate < self.num_val_samples_to_generate:
            log.warning(
                f"Requested {self.num_val_samples_to_generate} validation samples, but pretokenized split "
                f"'{self.pretok_split_name}' only has {len(pretok_ds)} samples. Generating {samples_to_actually_generate}."
            )
        if samples_to_actually_generate == 0 and self.num_val_samples_to_generate > 0:
            log.warning(f"No validation samples will be generated as pretokenized dataset '{self.pretok_split_name}' is empty or too short.")
            self.data_store = []
            return
            
        # Ensure we have at least two distinct samples for A and A_prime if generating more than one sample.
        # If only one sample is requested, A and A_prime will be the same.
        indices_A = list(range(samples_to_actually_generate))
        indices_A_prime = indices_A[:] # Start with a copy

        if samples_to_actually_generate > 1:
            # Simple shuffle for A_prime to ensure they are different from A for most cases.
            # This is a basic approach; more sophisticated pairing might be needed for specific research goals.
            import random
            random.shuffle(indices_A_prime)
            # Ensure A is not paired with itself if possible
            for i in range(samples_to_actually_generate):
                if indices_A[i] == indices_A_prime[i]:
                    swap_idx = (i + 1) % samples_to_actually_generate
                    indices_A_prime[i], indices_A_prime[swap_idx] = indices_A_prime[swap_idx], indices_A_prime[i]


        for i in range(samples_to_actually_generate):
            idx_A = indices_A[i]
            idx_A_prime = indices_A_prime[i]

            pretok_sample_A = pretok_ds[idx_A]
            pretok_sample_A_prime = pretok_ds[idx_A_prime]

            input_ids_A_list = pretok_sample_A['input_ids']
            input_ids_A_prime_list = pretok_sample_A_prime['input_ids']

            # Convert to tensor and move to device for generation
            # Add batch dimension for the model
            input_ids_A_device = torch.tensor(input_ids_A_list, dtype=torch.long, device=self.generation_device).unsqueeze(0)
            input_ids_A_prime_device = torch.tensor(input_ids_A_prime_list, dtype=torch.long, device=self.generation_device).unsqueeze(0)

            act_A, pos_A = _generate_single_activation(
                self.orig_model_for_gen,
                input_ids_A_device,
                self.config['layer_l'],
                self.config['min_pos'],
                self.tokenizer.pad_token_id,
            )

            act_A_prime, pos_A_prime = _generate_single_activation(
                self.orig_model_for_gen,
                input_ids_A_prime_device,
                self.config['layer_l'],
                self.config['min_pos'],
                self.tokenizer.pad_token_id,
            )

            self.data_store.append({
                "A": act_A.cpu(), # Store on CPU
                "A_prime": act_A_prime.cpu(), # Store on CPU
                "input_ids_A": torch.tensor(input_ids_A_list, dtype=torch.long), # Store original list as tensor on CPU
                "input_ids_A_prime": torch.tensor(input_ids_A_prime_list, dtype=torch.long), # Store original list as tensor on CPU
                "token_pos_A": pos_A,
                "token_pos_A_prime": pos_A_prime,
                "layer_idx": self.config['layer_l'],
            })
            if (i + 1) % 100 == 0:
                log.info(f"Rank 0 generated {i+1}/{samples_to_actually_generate} validation samples...")
                
    def __len__(self) -> int:
        return len(self.data_store)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not self.data_store:
            raise IndexError("Dataset is empty. This can happen if generation failed or no samples were specified.")
        return self.data_store[idx]

# --- Training Dataset Components ---

class OnTheFlyTrainingActivationGenerator:
    def __init__(
        self,
        orig_model_for_gen: OrigWrapper,
        tokenizer: PreTrainedTokenizer,
        pretok_dataset_path: str,
        pretok_split_name: str, # Should typically be the training split
        on_the_fly_config: Dict[str, Any],
        generation_device: torch.device,
        rank: int,
        world_size: int,
    ):
        self.orig_model_for_gen = orig_model_for_gen
        self.tokenizer = tokenizer
        self.pretok_dataset_path = pretok_dataset_path
        self.pretok_split_name = pretok_split_name
        self.config = on_the_fly_config
        self.generation_device = generation_device
        self.rank = rank
        self.world_size = world_size

        # Model should be on the correct device
        self.orig_model_for_gen.to(self.generation_device)

        self.sharded_pretok_dataset: Optional[HFDataset] = None
        self.pretok_iterator: Optional[Iterator[Dict[str, Any]]] = None
        self._load_and_shard_pretok_dataset()

    def _load_and_shard_pretok_dataset(self):
        """Loads the pretokenized dataset and shards it for the current rank."""
        try:
            full_pretok_path = Path(self.pretok_dataset_path) / self.pretok_split_name
            if not full_pretok_path.exists():
                log.error(f"Rank {self.rank}: Pretokenized training data not found at: {full_pretok_path}")
                # Set to empty iterable to prevent errors, but this is a problem.
                self.sharded_pretok_dataset = HFDataset.from_dict({}) # Empty dataset
            else:
                dataset_to_shard: HFDataset = load_from_disk(str(full_pretok_path))
                if self.world_size > 1:
                    self.sharded_pretok_dataset = dataset_to_shard.shard(
                        num_shards=self.world_size, index=self.rank, contiguous=True
                    )
                else:
                    self.sharded_pretok_dataset = dataset_to_shard
                log.info(f"Rank {self.rank}: Loaded and sharded pretokenized training data. Shard size: {len(self.sharded_pretok_dataset)} samples.")
        except Exception as e:
            log.error(f"Rank {self.rank}: Failed to load/shard pretokenized training dataset from {self.pretok_dataset_path}/{self.pretok_split_name}: {e}")
            self.sharded_pretok_dataset = HFDataset.from_dict({})

        if self.sharded_pretok_dataset is None or len(self.sharded_pretok_dataset) == 0:
            log.warning(f"Rank {self.rank}: Pretokenized training dataset shard is empty. The generator will produce no samples.")
            self.pretok_iterator = iter([]) # Empty iterator
        else:
            self.pretok_iterator = iter(self.sharded_pretok_dataset) # type: ignore

    def _get_next_pretokenized_sample(self) -> Optional[Dict[str, Any]]:
        """Gets a sample from the sharded pretokenized iterator, cycling if exhausted."""
        if self.pretok_iterator is None: # Should not happen if _load_and_shard_pretok_dataset worked
            return None
        try:
            return next(self.pretok_iterator)
        except StopIteration:
            if self.sharded_pretok_dataset is None or len(self.sharded_pretok_dataset) == 0:
                # If the underlying sharded dataset is empty, there's nothing to iterate over.
                log.warning(f"Rank {self.rank}: Sharded pretokenized dataset is empty, cannot get next sample.")
                return None
            # Cycle through the dataset
            log.info(f"Rank {self.rank}: Cycling pretokenized training data iterator.")
            self.pretok_iterator = iter(self.sharded_pretok_dataset)
            try:
                return next(self.pretok_iterator)
            except StopIteration: # Still empty after reset (e.g., dataset became empty)
                log.error(f"Rank {self.rank}: Pretokenized training data iterator empty even after reset.")
                return None


    def generate_samples(self, num_samples_to_generate: int) -> Iterator[Dict[str, Any]]:
        """A generator function that yields training samples (A activation only)."""
        if self.sharded_pretok_dataset is None or len(self.sharded_pretok_dataset) == 0:
            log.warning(f"Rank {self.rank}: Training generator has no pretokenized data, cannot generate samples.")
            return # Yields nothing

        generated_count = 0
        for _ in range(num_samples_to_generate):
            pretok_sample_A = self._get_next_pretokenized_sample()
            if pretok_sample_A is None:
                log.warning(f"Rank {self.rank}: Ran out of pretokenized samples unexpectedly during generation for cache fill.")
                break # Stop if no more pretokenized data

            input_ids_A_list = pretok_sample_A['input_ids']
            input_ids_A_device = torch.tensor(input_ids_A_list, dtype=torch.long, device=self.generation_device).unsqueeze(0)

            act_A, pos_A = _generate_single_activation(
                self.orig_model_for_gen,
                input_ids_A_device,
                self.config['layer_l'],
                self.config['min_pos'],
                self.tokenizer.pad_token_id,
            )
            
            yield {
                "A": act_A.cpu(), # Store on CPU
                "input_ids_A": torch.tensor(input_ids_A_list, dtype=torch.long), # Store original list as tensor on CPU
                "token_pos_A": pos_A,
                "layer_idx": self.config['layer_l'],
            }
            generated_count += 1
        log.info(f"Rank {self.rank}: Training generator yielded {generated_count} samples for this cache fill.")


class TrainingActivationCache:
    def __init__(
        self,
        generator: OnTheFlyTrainingActivationGenerator,
        cache_config: Dict[str, Any], # e.g., {'type': 'memory', 'size_samples': 10000, 'disk_path': ...}
        rank: int, # For unique disk paths if needed
    ):
        self.generator = generator
        self.cache_type = cache_config.get('type', 'memory')
        self.target_cache_size = cache_config.get('size_samples', 1024) # Number of items to buffer
        self.rank = rank

        self._cache_content: deque = deque()
        self._disk_cache_files: deque = deque() # Stores paths to temporary .pt files for disk cache
        self._disk_cache_dir: Optional[Path] = None

        if self.cache_type == 'disk':
            disk_base_path_str = cache_config.get('disk_path', './.training_activation_cache')
            self._disk_cache_dir = Path(disk_base_path_str) / f"rank_{self.rank}"
            self._disk_cache_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"Rank {self.rank}: Training activation disk cache initialized at {self._disk_cache_dir}")
            # Clean up any old cache files from previous runs for this rank
            for old_file in self._disk_cache_dir.glob("cache_*.pt"):
                try:
                    old_file.unlink()
                except OSError as e:
                    log.warning(f"Rank {self.rank}: Could not delete old cache file {old_file}: {e}")
        
        self.is_filling = False # Simple lock to prevent re-entrant filling


    def _fill_cache_if_needed(self):
        if self.is_filling:
            return # Already filling, wait for it to complete

        if (self.cache_type == 'memory' and not self._cache_content) or \
           (self.cache_type == 'disk' and not self._disk_cache_files):
            
            self.is_filling = True
            log.info(f"Rank {self.rank}: Cache empty, refilling with target {self.target_cache_size} samples (type: {self.cache_type})...")
            
            num_filled = 0
            for i, sample in enumerate(self.generator.generate_samples(self.target_cache_size)):
                if self.cache_type == 'memory':
                    self._cache_content.append(sample)
                elif self.cache_type == 'disk' and self._disk_cache_dir:
                    # Save each sample to a unique file in the rank's disk cache directory
                    # This is inefficient for many small files; consider batching to fewer .pt files if performance is an issue.
                    # For now, one file per sample for simplicity.
                    temp_file_path = self._disk_cache_dir / f"cache_item_{Path(torch.randn(1).hex().lstrip('0x'))}.pt" # Unique enough
                    torch.save(sample, temp_file_path)
                    self._disk_cache_files.append(temp_file_path)
                num_filled +=1
            
            if num_filled < self.target_cache_size and num_filled > 0:
                log.warning(f"Rank {self.rank}: Cache refill generated {num_filled} samples, less than target {self.target_cache_size}. Pretokenized source might be exhausted for this cycle.")
            elif num_filled == 0 and self.target_cache_size > 0 :
                 log.warning(f"Rank {self.rank}: Cache refill generated 0 samples. Pretokenized source might be fully exhausted or empty.")


            log.info(f"Rank {self.rank}: Cache refill complete. Memory cache size: {len(self._cache_content)}, Disk cache files: {len(self._disk_cache_files)}")
            self.is_filling = False


    def pop_sample(self) -> Optional[Dict[str, Any]]:
        self._fill_cache_if_needed()
        
        if self.cache_type == 'memory':
            if not self._cache_content:
                log.warning(f"Rank {self.rank}: Memory cache is empty after fill attempt. Returning None.")
                return None
            return self._cache_content.popleft()
        elif self.cache_type == 'disk':
            if not self._disk_cache_files:
                log.warning(f"Rank {self.rank}: Disk cache is empty after fill attempt. Returning None.")
                return None
            file_path_to_load = self._disk_cache_files.popleft()
            try:
                sample = torch.load(file_path_to_load)
                file_path_to_load.unlink() # Delete after loading
                return sample
            except Exception as e:
                log.error(f"Rank {self.rank}: Error loading/deleting disk cache file {file_path_to_load}: {e}")
                return None # Or attempt to pop next? For now, fail this sample.
        return None # Should not be reached


class OnTheFlyCachedTrainingDataset(Dataset):
    def __init__(
        self,
        cache: TrainingActivationCache,
        # Total number of unique samples in the underlying sharded pretokenized dataset for this rank.
        # This defines the "epoch" length for this rank.
        total_unique_samples_on_rank: int,
        rank: int
    ):
        self.cache = cache
        self.total_samples_in_epoch = total_unique_samples_on_rank
        self.rank = rank
        
        if self.total_samples_in_epoch == 0:
            log.warning(f"Rank {self.rank}: OnTheFlyCachedTrainingDataset initialized with total_unique_samples_on_rank = 0. Dataset will be empty.")

    def __len__(self) -> int:
        # The length of the dataset defines one "epoch" over the sharded pretokenized data.
        # The cache will cycle through the underlying data as needed.
        return self.total_samples_in_epoch

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # The idx is used by the DataLoader's sampler to iterate.
        # We just pop from the cache, which handles refilling.
        if self.total_samples_in_epoch == 0:
             raise IndexError(f"Rank {self.rank}: Attempting to get item from an empty OnTheFlyCachedTrainingDataset.")

        sample = self.cache.pop_sample()
        if sample is None:
            # This is a critical issue if it happens, means cache couldn't provide a sample.
            # Could be due to pretokenized data exhaustion or other errors during cache fill.
            log.error(f"Rank {self.rank}: TrainingActivationCache.pop_sample() returned None for idx {idx}. This may lead to training errors.")
            # What to return here? Raising an error might stop training.
            # Returning a dummy or previous sample could hide issues.
            # For now, let's raise an error to make it visible.
            raise RuntimeError(f"Rank {self.rank}: Failed to retrieve a sample from the training cache for idx {idx}.")
        return sample
