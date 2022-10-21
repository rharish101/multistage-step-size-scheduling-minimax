"""Utility function to choose a model."""
from typing import Tuple

from torch import Tensor
from torch.utils.data import Dataset

from ..config import Config
from ..utils import invalid_task_error
from .cifar10 import CIFAR10TrainDataset, CIFAR10ValDataset
from .common import FirstItemDataset
from .covar import StandardNormalDataset
from .rls import RLSDataset


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

    if desc[0] == "covar":
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

        train_dataset = RLSDataset(config, stochastic=stochastic, mode=desc[1])
        val_dataset = FirstItemDataset(
            RLSDataset(config, stochastic=False, mode=desc[1])
        )

    elif desc[0] == "cifar10":
        train_dataset = CIFAR10TrainDataset(config)
        val_dataset = CIFAR10ValDataset(config)

    else:
        invalid_task_error(task)

    return train_dataset, val_dataset
