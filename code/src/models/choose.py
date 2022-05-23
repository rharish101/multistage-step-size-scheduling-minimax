"""Utility function to choose a model."""
from pytorch_lightning import LightningModule

from ..config import Config
from .rls import RLSHighConditionNum, RLSLowConditionNum
from .wgan import WGAN


def get_model(task: str, config: Config) -> LightningModule:
    """Get the appropriate model for the requested task.

    Args:
        task: The string specifying the optimization task
        config: The hyper-param config
    """
    if task.startswith("wgan/"):
        return WGAN(config, gen_type=task.split("/")[1])
    elif task == "rls/low":
        return RLSLowConditionNum(config)
    elif task == "rls/high":
        return RLSHighConditionNum(config)
    else:
        raise ValueError(f"Invalid task: {task}")
