from omegaconf import DictConfig, OmegaConf
from lens.training.schedules import parse_schedule_config as parse_schedule_config_from_schedules

def load_and_normalize_config(hydra_cfg: DictConfig) -> dict:
    """
    Converts a Hydra DictConfig to a plain Python dictionary and
    parses flexible schedule notations.
    """
    config = OmegaConf.to_container(hydra_cfg, resolve=True)
    config = parse_schedule_config_from_schedules(config) # Use the one from schedules.py
    return config
