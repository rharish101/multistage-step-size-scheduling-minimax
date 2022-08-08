"""Hyper-param config handling."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class Config:
    """Class to hold hyper-parameter configs.

    Attributes:
        batch_size: The batch size for training
        sched: The choice of scheduler (must be one of: const, step, var, poly,
            poly-sqrt)
        x_lr: The learning rate for minimization
        y_lr: The learning rate for maximization
        decay: The learning rate decay factor
        phase_scale: The scaling factor for the phase length for the "step"
            scheduler
        start_phase_len: The starting phase length for the "var" scheduler
        total_steps: The total steps for training the model
        seed: The global random seed (for reproducibility)
    """

    batch_size: int = 100
    sched: str = "step"
    x_lr: float = 1e-1
    y_lr: float = 5e-1
    decay: float = 2.0
    phase_scale: float = 1.0
    start_phase_len: int = 500
    total_steps: int = 20000
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
