import os
import torch
import logging
from lens.utils.logging import log_init as lens_log_init # Renamed to avoid clash

def init_wandb_run(config: dict, run_name: str, dataset_info: dict, 
                   resume_checkpoint_path: str | None, hydra_config_name: str | None,
                   log: logging.Logger) -> str | None:
    """Initializes Weights & Biases for the training run."""
    
    wandb_config = config.get('wandb', {})
    if not wandb_config.get('enabled', True): # Assume enabled by default
        log.info("W&B logging is disabled in the configuration.")
        return None

    wandb_run_id = config.get('wandb_resume_id')
    wandb_resume_mode = None

    if resume_checkpoint_path and not wandb_run_id:
        if not os.path.exists(resume_checkpoint_path):
            # This case should be handled before calling, but as a safeguard:
            log.warning(f"Resume checkpoint path specified but not found: {resume_checkpoint_path}")
        else:
            try:
                checkpoint_data = torch.load(resume_checkpoint_path, map_location='cpu')
                wandb_run_id = checkpoint_data.get('wandb_run_id')
                if wandb_run_id:
                    log.info(f"Found wandb run ID in checkpoint: {wandb_run_id}")
                    wandb_resume_mode = "must"
            except Exception as e:
                log.error(f"Could not load WandB run ID from checkpoint {resume_checkpoint_path}: {e}")


    wandb_init_kwargs = {
        'project': wandb_config.get('project', 'consistency-lens'),
        'name': run_name,
        'config': config,
        'mode': wandb_config.get('mode', 'online'),
        'tags': []
    }

    slurm_job_id = os.environ.get('SLURM_JOB_ID', None)
    if slurm_job_id:
        slurm_info = {
            'slurm_job_id': slurm_job_id,
            'slurm_job_name': os.environ.get('SLURM_JOB_NAME', 'unknown'),
            'slurm_nodelist': os.environ.get('SLURM_NODELIST', 'unknown'),
            'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID', None),
        }
        # Ensure config passed to wandb is mutable if it's part of init_kwargs['config']
        if 'slurm_info' not in wandb_init_kwargs['config']:
             wandb_init_kwargs['config']['slurm_info'] = {} # Ensure slurm_info key exists
        wandb_init_kwargs['config']['slurm_info'].update(slurm_info)

        wandb_init_kwargs['tags'].append(f"slurm-{slurm_job_id}")
        wandb_init_kwargs['tags'].append(f"node-{slurm_info['slurm_nodelist']}")
        log.info(f"Running under SLURM job ID: {slurm_job_id} on nodes: {slurm_info['slurm_nodelist']}")

    if dataset_info.get('dataset'):
        wandb_init_kwargs['tags'].append(f"dataset-{dataset_info['dataset']}")
    if dataset_info.get('model_name'):
        model_tag = dataset_info['model_name'].replace('/', '-')
        wandb_init_kwargs['tags'].append(f"model-{model_tag}")
    if dataset_info.get('layer') is not None:
        wandb_init_kwargs['tags'].append(f"layer-{dataset_info['layer']}")
    
    if hydra_config_name and hydra_config_name != 'config':
        wandb_init_kwargs['tags'].append(f"config-{hydra_config_name}")

    if wandb_run_id:
        wandb_init_kwargs['id'] = wandb_run_id
        wandb_init_kwargs['resume'] = wandb_resume_mode or "allow"

    current_wandb_run_id = lens_log_init(**wandb_init_kwargs)
    log.info(f"W&B initialized. Run ID: {current_wandb_run_id}, Name: {run_name}")
    return current_wandb_run_id
