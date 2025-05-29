import re
from pathlib import Path
import logging
from hydra.core.hydra_config import HydraConfig
from datetime import datetime

def get_project_root() -> Path:
    """Get the project root directory (consistency-lens folder)."""
    # This assumes the utils module is one level down from project root,
    # e.g., consistency-lens/lens/utils/path_utils.py
    # So, Path(__file__).parent.parent.parent
    return Path(__file__).parent.parent.parent.absolute()

def resolve_path(path_str: str) -> Path:
    """Resolve a path relative to the project root if it's a relative path."""
    path = Path(path_str)
    if not path.is_absolute():
        project_root = get_project_root()
        return project_root / path
    return path

def _get_hydra_config_name() -> str | None:
    """Attempts to get the config name from Hydra's context."""
    config_name = None
    try:
        hydra_cfg = HydraConfig.get()
        if hasattr(hydra_cfg, 'job') and hasattr(hydra_cfg.job, 'config_name'):
            config_name = hydra_cfg.job.config_name
        elif hasattr(hydra_cfg, 'runtime') and hasattr(hydra_cfg.runtime, 'choices'):
            config_name = hydra_cfg.runtime.choices.get('config_name', 'config')
    except Exception: # Broad exception as the original code does
        pass
    return config_name

def extract_dataset_info(activation_dir: str) -> dict:
    """Extract model name, layer, and dataset info from activation directory path.
    
    Expected formats:
    - .../SimpleStories_SimpleStories-5M/layer_5/SimpleStories_train
    - .../dataset_name/model_name/layer_X/split_name/
    - .../SimpleStories_train (direct path)
    """
    parts = Path(activation_dir).parts
    info = {
        'model_name': None,
        'layer': None,
        'dataset': None,
        'split': None
    }
    
    for i, part in enumerate(parts):
        if part.startswith('layer_'):
            layer_match = re.match(r'layer_(\d+)', part)
            if layer_match:
                info['layer'] = int(layer_match.group(1))
                if i > 0:
                    model_part = parts[i-1]
                    if '_' in model_part:
                        dataset_prefix, model_name = model_part.split('_', 1)
                        info['dataset'] = dataset_prefix
                        info['model_name'] = model_name.replace('_', '/')
                    else:
                        info['model_name'] = model_part
                        if i > 1:
                            info['dataset'] = parts[i-2]
                
                if i < len(parts) - 1:
                    split_part = parts[i+1]
                    if '_' in split_part:
                        dataset_name, split = split_part.rsplit('_', 1)
                        if split in ['train', 'test', 'val', 'validation']:
                            info['split'] = split
                            if not info['dataset']:
                                info['dataset'] = dataset_name
                    else:
                        info['split'] = split_part
                break
    
    if not info['dataset'] and parts:
        final_part = parts[-1]
        if '_' in final_part:
            dataset_name, split = final_part.rsplit('_', 1)
            if split in ['train', 'test', 'val', 'validation']:
                info['dataset'] = dataset_name
                info['split'] = split
    
    return info

def generate_run_name(config: dict, dataset_info: dict, resume_from: str = None, hydra_config_name: str = None, run_suffix: str = None) -> str:
    """Generate a descriptive run name based on config and dataset info."""
    components = []
    
    if hydra_config_name and hydra_config_name != 'config':
        clean_config = hydra_config_name.replace('_config', '').replace('.yaml', '')
        components.append(clean_config)
    
    if dataset_info['dataset']:
        dataset_name = dataset_info['dataset']
        dataset_map = {
            'SimpleStories': 'SS', 'openwebtext': 'OWT', 'pile': 'Pile',
        }
        dataset_short = dataset_map.get(dataset_name, dataset_name)
        if len(dataset_short) > 8 and dataset_short not in dataset_map.values():
            dataset_short = ''.join([w[0].upper() for w in dataset_short.replace('_', ' ').replace('-', ' ').split()])
        components.append(dataset_short)
    
    if dataset_info['model_name']:
        model_name = dataset_info['model_name']
        model_map = {
            'SimpleStories/SimpleStories-5M': '5M', 'openai-community/gpt2': 'GPT2',
            'gpt2': 'GPT2', 'gpt2-medium': 'GPT2-M', 'gpt2-large': 'GPT2-L', 'gpt2-xl': 'GPT2-XL',
        }
        model_short = model_map.get(model_name, model_name.split('/')[-1])
        components.append(model_short)
    
    if dataset_info['layer'] is not None:
        components.append(f"L{dataset_info['layer']}")
    
    freeze_schedule = config.get('freeze_schedule', {})
    if freeze_schedule.get('enabled', False):
        unfreeze_at = freeze_schedule.get('unfreeze_at', '')
        if 'epoch' in str(unfreeze_at).lower(): components.append('unfreeze')
        else: components.append('prog-unfreeze')
    else:
        decoder_cfg = config.get('trainable_components', {}).get('decoder', {})
        encoder_cfg = config.get('trainable_components', {}).get('encoder', {})
        if decoder_cfg.get('base_model', False) or encoder_cfg.get('base_model', False):
            components.append('full')
        else: components.append('frozen')
    
    lr = config.get('learning_rate', 1e-4)
    components.append(f"lr{lr:.0e}".replace('e-0', 'e-').replace('e+0', 'e'))
    
    t_text = config.get('t_text', 10)
    components.append(f"t{t_text}")
    
    num_epochs = config.get('num_train_epochs', 0)
    max_steps = config.get('max_train_steps', 0)
    if num_epochs > 0: components.append(f"{num_epochs}ep")
    elif max_steps > 0:
        if max_steps >= 1000: components.append(f"{max_steps//1000}k")
        else: components.append(f"{max_steps}s")
    
    if resume_from: components.append("resume")
    
    timestamp = datetime.now().strftime("%m%d_%H%M")
    components.append(timestamp)
    
    if run_suffix: components.append(run_suffix)
    
    return "_".join(components)

def setup_activation_dirs(config: dict, log: logging.Logger) -> tuple[str, str | None]:
    """Sets up and returns paths for training and validation activation directories."""
    model_name_clean = config['model_name'].replace("/", "_")
    layer_l = config.get('layer_l', 5)

    cli_activation_dir = config.get('activation_dir')
    base_activation_dir_str = cli_activation_dir if cli_activation_dir is not None else config['activation_dumper']['output_dir']
    base_activation_path = resolve_path(base_activation_dir_str)
    activation_dir = str(base_activation_path.parent / model_name_clean / f"layer_{layer_l}" / base_activation_path.name)
    log.info(f"Main activation directory: {activation_dir}")

    effective_val_activation_dir: str | None = None
    base_val_activation_dir_str = config.get('val_activation_dir')
    if base_val_activation_dir_str:
        base_val_path = resolve_path(base_val_activation_dir_str)
        effective_val_activation_dir = str(base_val_path.parent / model_name_clean / f"layer_{layer_l}" / base_val_path.name)
        log.info(f"Validation activation directory: {effective_val_activation_dir}")
    
    return activation_dir, effective_val_activation_dir

def setup_checkpoint_dir_and_config(config: dict, run_name: str, log: logging.Logger) -> str:
    """Sets up checkpoint directory, updates config, and returns directory path."""
    checkpoint_config = config.get('checkpoint', {})
    base_checkpoint_dir = resolve_path(checkpoint_config.get('output_dir', 'outputs'))
    run_checkpoint_dir = base_checkpoint_dir / run_name
    
    # Ensure the directory exists (important for CheckpointManager)
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_config['output_dir'] = str(run_checkpoint_dir)
    config['checkpoint'] = checkpoint_config # Update the original config dict
    log.info(f"Checkpoint directory: {run_checkpoint_dir}")
    return str(run_checkpoint_dir)
