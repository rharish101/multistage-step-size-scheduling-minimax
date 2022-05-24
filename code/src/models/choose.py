"""Utility function to choose a model."""
from itertools import product
from typing import Final, Iterable, NoReturn

from pytorch_lightning import LightningModule

from ..config import Config
from .rls import RLSHighConditionNum, RLSLowConditionNum
from .wgan import WGAN


def _combine_desc(*args: Iterable[str]) -> Iterable[str]:
    """Get the task names by combining allowed names.

    Example:
        >>> list(_combine_desc(["wgan"], ["linear", "nn"]))
        ["wgan/linear", "wgan/nn"]
    """
    return map(lambda x: "/".join(x), product(*args))


AVAIL_TASKS: Final = [
    *_combine_desc(["wgan"], ["linear", "nn"]),
    *_combine_desc(["rls"], ["low", "high"]),
]


def _invalid_task_err(task: str) -> NoReturn:
    raise ValueError(f"Invalid task: {task}")


def get_model(task: str, config: Config) -> LightningModule:
    """Get the appropriate model for the requested task.

    Args:
        task: The string specifying the optimization task
        config: The hyper-param config
    """
    desc = task.split("/")
    if desc[0] == "wgan":
        return WGAN(config, gen_type=task.split("/")[1])
    elif desc[0] == "rls":
        if desc[1] == "low":
            return RLSLowConditionNum(config)
        elif desc[1] == "high":
            return RLSHighConditionNum(config)
        else:
            _invalid_task_err(task)
    else:
        _invalid_task_err(task)
