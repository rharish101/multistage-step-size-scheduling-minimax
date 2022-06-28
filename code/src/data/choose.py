"""Utility function to choose a model."""
from typing import Tuple

from torch import Tensor
from torch.utils.data import Dataset

from ..config import Config
from ..utils import invalid_task_error
from .common import FirstItemDataset
from .rls import RLSDataset
from .wgan import StandardNormalDataset


def get_datasets(
    task: str, config: Config
) -> Tuple[Dataset[Tensor], Dataset[Tensor]]:
    """Get the appropriate training and validation datasets for the requested task.

    Args:
        task: The string specifying the optimization task
        config: The hyper-param config

    Returns:
        The training dataset for the requested task
        The validation dataset for the requested task
    """
    desc = task.split("/")

    if desc[0] == "wgan":
        train_dataset = StandardNormalDataset(config)
        val_dataset: Dataset[Tensor] = FirstItemDataset(
            StandardNormalDataset(config)
        )

    elif desc[0] == "rls":
        if desc[2] == "stoc":
            stochastic = True
        elif desc[2] == "full":
            stochastic = False
        else:
            invalid_task_error(task)

        train_dataset = RLSDataset(config, stochastic=stochastic)
        val_dataset = FirstItemDataset(RLSDataset(config, stochastic=False))

    else:
        invalid_task_error(task)

    return train_dataset, val_dataset
