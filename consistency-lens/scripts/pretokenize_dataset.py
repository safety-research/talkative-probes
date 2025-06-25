"""Pre-tokenize datasets to eliminate tokenization bottleneck during activation dumping."""

import logging
from pathlib import Path
import json
import hydra
from omegaconf import DictConfig, OmegaConf

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import dotenv
dotenv.load_dotenv()


def find_text_column(dataset: Dataset) -> str:
    """Heuristically find the text column in a dataset."""
    # Look for common text-related keywords in column names
    for col in dataset.column_names:
        if any(keyword in col.lower() for keyword in ["text", "content", "document"]):
            return col
    
    # As a fallback, find the first column that is of string type
    if len(dataset) > 0:
        for col in dataset.column_names:
            if isinstance(dataset[0][col], str):
                return col
    
    # If no suitable column is found, raise an error
    raise ValueError(f"Could not automatically find a text column in dataset with columns: {dataset.column_names}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Setup logging
    log = logging.getLogger(__name__)
    
    # Get config values
    pretokenize_cfg = cfg.pretokenize
    activation_dumper_cfg = cfg.activation_dumper
    
    # Get dataset info
    dataset_name = activation_dumper_cfg["hf_dataset_name"]
    tokenizer_name = cfg["tokenizer_name"]
    seq_len = activation_dumper_cfg["seq_len"]
    
    # Get pre-tokenization parameters
    output_dir_cfg = pretokenize_cfg.get("output_dir")
    num_proc = pretokenize_cfg.get("num_proc", 4)
    batch_size = pretokenize_cfg.get("batch_size", 1000)
    force = pretokenize_cfg.get("force", False)
    
    # Output directory - now includes seq_len for versioning
    if output_dir_cfg:
        output_dir = Path(output_dir_cfg)
    else:
        output_dir = Path("data/pretokenized") / f"{dataset_name.replace('/', '_')}_seq{seq_len}"
    
    # Check if already exists and force flag
    if output_dir.exists() and not force:
        log.info(f"Pretokenized data already exists at {output_dir}")
        log.info("Use pretokenize.force=true to re-tokenize")
        return
    
    log.info(f"Pre-tokenizing dataset: {dataset_name}")
    log.info(f"Using tokenizer: {tokenizer_name}")
    log.info(f"Sequence length: {seq_len}")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Using {num_proc} processes")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # --- Dataset Loading and Splitting ---
    train_split_name = "train"
    val_split_name_from_cfg = activation_dumper_cfg.get("val_hf_split")
    
    datasets_to_process = {}

    # Decide how to get train and validation datasets
    if not val_split_name_from_cfg or val_split_name_from_cfg == train_split_name:
        # Case: No validation split specified, or it's the same as train.
        # Create a validation split from the training data.
        log.info(f"No separate validation split specified. Creating one from '{train_split_name}'.")
        
        try:
            full_train_dataset = load_dataset(dataset_name, split=train_split_name, trust_remote_code=True, num_proc=num_proc)
        except Exception as e:
            log.error(f"Failed to load '{train_split_name}' split for '{dataset_name}': {e}")
            raise

        validation_split_size = pretokenize_cfg.get("validation_split_size", 0.05)
        seed = pretokenize_cfg.get("seed", 42)
        
        if len(full_train_dataset) < 2:
            raise ValueError(f"Train dataset '{train_split_name}' has fewer than 2 samples ({len(full_train_dataset)}), cannot create a validation split.")

        log.info(f"Splitting '{train_split_name}' with test_size={validation_split_size} and seed={seed}.")
        
        split_dataset = full_train_dataset.train_test_split(test_size=validation_split_size, shuffle=True, seed=seed)
        
        datasets_to_process['train'] = split_dataset['train']
        datasets_to_process['validation'] = split_dataset['test']
        
        log.info(f"Created train split with {len(datasets_to_process['train'])} samples.")
        log.info(f"Created validation split with {len(datasets_to_process['validation'])} samples.")

    else:
        # Case: A separate validation split is specified. Load both.
        log.info(f"Using specified train ('{train_split_name}') and validation ('{val_split_name_from_cfg}') splits.")
        
        try:
            train_dataset = load_dataset(dataset_name, split=train_split_name, trust_remote_code=True, num_proc=num_proc)
            datasets_to_process['train'] = train_dataset
            log.info(f"Loaded train split '{train_split_name}' with {len(train_dataset)} samples.")
        except Exception as e:
            log.error(f"Failed to load train split '{train_split_name}': {e}")
            raise

        try:
            val_dataset = load_dataset(dataset_name, split=val_split_name_from_cfg, trust_remote_code=True, num_proc=num_proc)
            datasets_to_process['validation'] = val_dataset
            log.info(f"Loaded validation split '{val_split_name_from_cfg}' with {len(val_dataset)} samples.")
        except Exception as e:
            log.error(f"Failed to load specified validation split '{val_split_name_from_cfg}': {e}")
            raise

    # --- Find Text Column and Tokenize ---
    # Find text column from the first available dataset, assuming it's consistent across splits.
    if not datasets_to_process:
        log.error("No datasets were loaded or created. Cannot proceed.")
        return
        
    sample_dataset = next(iter(datasets_to_process.values()))
    text_column = find_text_column(sample_dataset)
    log.info(f"Using text column: '{text_column}' for all splits.")
    
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_special_tokens_mask=False,  # Save space
        )

    for split_name, dataset in datasets_to_process.items():
        log.info(f"\nProcessing split: {split_name} ({len(dataset)} samples)")
        
        # Apply tokenization
        log.info("Tokenizing...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {split_name}",
        )
        
        # Save tokenized data
        split_output_dir = output_dir / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"Saving to {split_output_dir}")
        tokenized_dataset.save_to_disk(str(split_output_dir))
        
        # Save metadata
        metadata = {
            "dataset_name": dataset_name,
            "tokenizer_name": tokenizer_name,
            "seq_len": seq_len,
            "num_samples": len(tokenized_dataset),
            "split": split_name,
            "columns": tokenized_dataset.column_names,
        }
        
        with open(split_output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        log.info(f"✓ Completed {split_name}: {len(tokenized_dataset)} samples")
    
    log.info("\n✓ Pre-tokenization complete!")
    log.info(f"To use, update your config to load from: {output_dir}")


if __name__ == "__main__":
    main()