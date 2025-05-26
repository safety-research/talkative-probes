#!/usr/bin/env python3
"""Reorganize activation storage to a consistent structure.

Target structure:
data/activations/{model_name}/layer_{N}/{dataset_name}_{split}/
├── metadata.json
├── rank_0/
│   ├── metadata.json
│   └── shard_*.pt
└── rank_1/
    ├── metadata.json
    └── shard_*.pt
"""

import argparse
import json
import shutil
from pathlib import Path
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_activation_directories(base_path: Path) -> List[Tuple[Path, Dict]]:
    """Find all activation directories and analyze their structure."""
    activation_dirs = []
    
    # Pattern 1: data/activations/{model}/layer_{N}/{dataset}
    pattern1_dirs = list(base_path.glob("data/activations/**/layer_*/*/"))
    
    # Pattern 2: data/{model}/layer_{N}/activations
    pattern2_dirs = list(base_path.glob("data/**/layer_*/activations/"))
    
    # Pattern 3: data/corpus/{dataset} (might have activations)
    pattern3_dirs = list(base_path.glob("data/corpus/*/"))
    
    all_dirs = pattern1_dirs + pattern2_dirs + pattern3_dirs
    
    for dir_path in all_dirs:
        if dir_path.name == "pretokenized":
            continue
            
        info = analyze_directory(dir_path)
        if info["has_activations"]:
            activation_dirs.append((dir_path, info))
    
    return activation_dirs


def analyze_directory(dir_path: Path) -> Dict:
    """Analyze a directory to understand its structure."""
    info = {
        "has_activations": False,
        "has_metadata": False,
        "has_rank_dirs": False,
        "num_shards": 0,
        "model_name": None,
        "layer": None,
        "dataset": None,
        "split": None,
    }
    
    # Check for activation files
    pt_files = list(dir_path.glob("**/*.pt"))
    info["num_shards"] = len(pt_files)
    info["has_activations"] = len(pt_files) > 0
    
    # Check for metadata
    if (dir_path / "metadata.json").exists():
        info["has_metadata"] = True
        try:
            with open(dir_path / "metadata.json", "r") as f:
                metadata = json.load(f)
                config = metadata.get("config", {})
                info["model_name"] = config.get("model_name")
                info["layer"] = config.get("layer_idx")
                info["dataset"] = config.get("hf_dataset_name")
                info["split"] = config.get("split_name")
        except:
            pass
    
    # Check for rank directories
    rank_dirs = list(dir_path.glob("rank_*"))
    info["has_rank_dirs"] = len(rank_dirs) > 0
    
    # Try to infer from path if metadata missing
    if not info["model_name"]:
        parts = dir_path.parts
        for i, part in enumerate(parts):
            if "layer_" in part:
                info["layer"] = int(part.split("_")[1])
                # Model name might be before layer
                if i > 0:
                    potential_model = parts[i-1]
                    if "/" in potential_model:
                        info["model_name"] = potential_model
                    else:
                        # Check if it's under a known structure
                        if i > 1 and parts[i-2] == "activations":
                            info["model_name"] = parts[i-1]
    
    # Infer dataset from path
    if not info["dataset"]:
        if "SimpleStories" in str(dir_path):
            info["dataset"] = "SimpleStories/SimpleStories"
        elif "openwebtext" in str(dir_path):
            info["dataset"] = "Skylion007/openwebtext"
        elif "pile" in str(dir_path).lower():
            info["dataset"] = "EleutherAI/pile"
    
    # Infer split from path
    if not info["split"]:
        path_str = str(dir_path).lower()
        if "_test" in path_str or "/test" in path_str:
            info["split"] = "test"
        elif "_train" in path_str or "/train" in path_str:
            info["split"] = "train"
        elif "_val" in path_str or "/val" in path_str:
            info["split"] = "validation"
    
    return info


def get_target_path(model_name: str, layer: int, dataset: str, split: str) -> Path:
    """Get the target path for activations."""
    # Normalize model name (remove special chars)
    model_name_clean = model_name.replace("/", "_")
    dataset_clean = dataset.replace("/", "_") if dataset else "unknown"
    
    return Path(f"data/activations/{model_name_clean}/layer_{layer}/{dataset_clean}_{split}")


def reorganize_activations(dry_run: bool = True):
    """Reorganize all activations to consistent structure."""
    base_path = Path.cwd()
    activation_dirs = find_activation_directories(base_path)
    
    logger.info(f"Found {len(activation_dirs)} directories with activations")
    
    moves = []
    
    for dir_path, info in activation_dirs:
        if not all([info["model_name"], info["layer"], info["split"]]):
            logger.warning(f"Skipping {dir_path} - missing info: {info}")
            continue
        
        target_path = get_target_path(
            info["model_name"], 
            info["layer"], 
            info["dataset"],
            info["split"]
        )
        
        # Don't move if already in correct location
        if dir_path.resolve() == (base_path / target_path).resolve():
            logger.info(f"Already in correct location: {dir_path}")
            continue
        
        moves.append((dir_path, target_path))
    
    if not moves:
        logger.info("No reorganization needed!")
        return
    
    logger.info(f"\nPlanned moves ({len(moves)} total):")
    for src, dst in moves:
        logger.info(f"  {src} -> {dst}")
    
    if dry_run:
        logger.info("\nDRY RUN - no changes made. Run with --execute to apply changes.")
        return
    
    # Execute moves
    for src, dst in moves:
        logger.info(f"\nMoving {src} -> {dst}")
        
        # Create parent directories
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        # If destination exists, we need to merge
        if dst.exists():
            logger.warning(f"Destination exists, merging: {dst}")
            
            # Move rank directories
            for rank_dir in src.glob("rank_*"):
                dst_rank = dst / rank_dir.name
                if dst_rank.exists():
                    # Merge shards
                    for shard in rank_dir.glob("*.pt"):
                        dst_shard = dst_rank / shard.name
                        if not dst_shard.exists():
                            shutil.move(str(shard), str(dst_shard))
                        else:
                            logger.warning(f"Shard exists, skipping: {dst_shard}")
                else:
                    shutil.move(str(rank_dir), str(dst))
            
            # Update metadata
            if (src / "metadata.json").exists():
                update_metadata(src / "metadata.json", dst / "metadata.json")
            
            # Remove empty source
            if not any(src.iterdir()):
                src.rmdir()
        else:
            # Simple move
            shutil.move(str(src), str(dst))
        
        logger.info(f"  ✓ Completed")


def update_metadata(src_meta_path: Path, dst_meta_path: Path):
    """Merge metadata files."""
    if not dst_meta_path.exists():
        shutil.copy2(str(src_meta_path), str(dst_meta_path))
        return
    
    # Load both
    with open(src_meta_path, "r") as f:
        src_meta = json.load(f)
    with open(dst_meta_path, "r") as f:
        dst_meta = json.load(f)
    
    # Merge shards
    dst_meta["shards"].extend(src_meta.get("shards", []))
    dst_meta["total_samples"] += src_meta.get("total_samples", 0)
    
    # Save
    with open(dst_meta_path, "w") as f:
        json.dump(dst_meta, f, indent=2)


def create_symlinks():
    """Create symlinks for backward compatibility."""
    # Add any necessary symlinks here
    pass


def main():
    parser = argparse.ArgumentParser(description="Reorganize activation storage")
    parser.add_argument("--execute", action="store_true", help="Execute the reorganization (default is dry run)")
    parser.add_argument("--symlinks", action="store_true", help="Create backward compatibility symlinks")
    
    args = parser.parse_args()
    
    reorganize_activations(dry_run=not args.execute)
    
    if args.symlinks and args.execute:
        create_symlinks()


if __name__ == "__main__":
    main()