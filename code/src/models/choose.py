"""Utility function to choose a model."""
from pytorch_lightning import LightningModule

from ..config import Config
from ..utils import invalid_task_error
from .covar import CovarWGAN
from .cifar10 import CIFAR10GAN
from .rls import RLSHighConditionNum, RLSLowConditionNum


def get_model(task: str, config: Config) -> LightningModule:
    """Get the appropriate model for the requested task.

    Args:
        task: The string specifying the optimization task
        config: The hyper-param config

    Returns:
        The model for the requested task
    """
    desc = task.split("/")
    if desc[0] == "covar":
        return CovarWGAN(config, gen_type=task.split("/")[1])
    elif desc[0] == "rls":
        if desc[2] == "stoc":
            stochastic = True
        elif desc[2] == "full":
            stochastic = False
        else:
            invalid_task_error(task)

        if desc[1] == "low":
            return RLSLowConditionNum(config, stochastic)
        elif desc[1] == "high":
            return RLSHighConditionNum(config, stochastic)
        else:
            invalid_task_error(task)
    elif desc[0] == "cifar10":
        return CIFAR10GAN(config)
    else:
        invalid_task_error(task)
