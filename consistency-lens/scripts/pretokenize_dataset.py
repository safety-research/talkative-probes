#!/usr/bin/env python3
"""Pre-tokenize datasets to eliminate tokenization bottleneck during activation dumping."""

import logging
from pathlib import Path
import json
import hydra
from omegaconf import DictConfig, OmegaConf

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Setup logging
    log = logging.getLogger(__name__)
    
    # Get config values - use defaults if pretokenize section doesn't exist
    pretokenize_cfg = cfg.pretokenize
    activation_dumper_cfg = cfg.activation_dumper
    
    # Get dataset info
    dataset_name = activation_dumper_cfg["hf_dataset_name"]
    tokenizer_name = cfg["tokenizer_name"]
    seq_len = activation_dumper_cfg["seq_len"]
    
    # Handle both dict and DictConfig
    output_dir_cfg = pretokenize_cfg.output_dir
    num_proc = pretokenize_cfg.num_proc
    batch_size = pretokenize_cfg.batch_size
    force = pretokenize_cfg.force
    
    # Output directory
    if output_dir_cfg:
        output_dir = Path(output_dir_cfg)
    else:
        output_dir = Path("data/pretokenized") / dataset_name.replace("/", "_")
    
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
    
    # Process each split
    splits_to_process = ["train"]
    val_split = activation_dumper_cfg.get("val_hf_split")
    if val_split:
        splits_to_process.append(val_split)
    
    for split in splits_to_process:
        log.info(f"\nProcessing split: {split}")
        
        # Load dataset
        try:
            dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
            log.info(f"Loaded {len(dataset)} samples")
        except Exception as e:
            log.warning(f"Could not load split {split}: {e}")
            continue
        
        # Find text column
        text_column = None
        for col in dataset.column_names:
            if any(keyword in col.lower() for keyword in ["text", "content", "document"]):
                text_column = col
                break
        
        if text_column is None:
            # Try first string column
            for col in dataset.column_names:
                if len(dataset) > 0 and isinstance(dataset[0][col], str):
                    text_column = col
                    break
        
        if text_column is None:
            log.error(f"Could not find text column in dataset columns: {dataset.column_names}")
            continue
        
        log.info(f"Using text column: {text_column}")
        
        # Tokenize function
        def tokenize_function(examples):
            return tokenizer(
                examples[text_column],
                padding="max_length",
                truncation=True,
                max_length=seq_len,
                return_special_tokens_mask=False,  # Save space
            )
        
        # Apply tokenization
        log.info("Tokenizing...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=dataset.column_names,  # Keep only tokenized data
            desc=f"Tokenizing {split}",
        )
        
        # Save
        split_output_dir = output_dir / split
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"Saving to {split_output_dir}")
        tokenized_dataset.save_to_disk(str(split_output_dir))
        
        # Save metadata
        metadata = {
            "dataset_name": dataset_name,
            "tokenizer_name": tokenizer_name,
            "seq_len": seq_len,
            "num_samples": len(tokenized_dataset),
            "split": split,
            "columns": tokenized_dataset.column_names,
        }
        
        with open(split_output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        log.info(f"✓ Completed {split}: {len(tokenized_dataset)} samples")
    
    log.info("\n✓ Pre-tokenization complete!")
    log.info(f"To use, update your config to load from: {output_dir}")


if __name__ == "__main__":
    main() 