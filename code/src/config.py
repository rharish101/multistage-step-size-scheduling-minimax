"""Hyper-param config handling."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class Config:
    """Class to hold hyper-parameter configs.

    Attributes:
        gen_type: The type of generator (must be one of: linear, nn)
        batch_size: The batch size for training
        sched: The choice of scheduler (must be one of: step, var)
        gen_lr: The maximum learning rate for the generator
        crit_lr: The maximum learning rate for the critic
        decay: The learning rate decay factor
        total_steps: The total steps for training the model
        real_mu: The mean of the "real" data
        real_sigma: The standard deviation for the "real" data
        seed: The global random seed (for reproducibility)
    """

    gen_type: str = "linear"
    batch_size: int = 100
    sched: str = "step"
    gen_lr: float = 1e-2
    crit_lr: float = 1e-2
    decay: float = 2.0
    start_phase_len: int = 500
    total_steps: int = 20000
    real_mu: float = 0.0
    real_sigma: float = 0.1
    seed: int = 0


def load_config(config_path: Optional[Path]) -> Config:
    """Load the hyper-param config at the given path.

    If the path is None, then the default config is returned.
    """
    if config_path is not None:
        with open(config_path, "rb") as f:
            args = yaml.safe_load(f)
    else:
        args = {}
    return Config(**args)


def update_config(config: Config, updates: Dict[str, Any]) -> Config:
    """Return a new config by adding the updated values to the given config.

    Args:
        config: The source config
        updates: The mapping of the keys that need to be updated with the newer
            values

    Returns:
        The new updated config
    """
    return Config(**{**vars(config), **updates})
