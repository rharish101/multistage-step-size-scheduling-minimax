"""Datasets for each task."""
from typing import Iterable, Tuple, TypeVar

import torch
from torch.utils.data import Dataset, IterableDataset

from .config import Config
from .models.rls import NUM_EXAMPLES
from .utils import invalid_task_error

T = TypeVar("T")


def get_datasets(
    task: str, config: Config
) -> Tuple[Dataset[torch.Tensor], Dataset[torch.Tensor]]:
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
        val_dataset: Dataset[torch.Tensor] = FirstItemDataset(
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


class FirstItemDataset(Dataset[T]):
    """Wrapper for a dataset that returns only the first item."""

    def __init__(self, dataset: Dataset[T]):
        """Store the wrapped dataset."""
        self.dataset = dataset

    def __getitem__(self, idx: int) -> T:
        """Return the first item of the wrapped dataset."""
        return next(iter(self.dataset))

    def __len__(self) -> int:
        """Return the length."""
        return 1


class StandardNormalDataset(IterableDataset):
    """Returns samples from the standard normal distribution."""

    def __init__(self, config: Config):
        """Store params used during sampling.

        Args:
            config: The hyper-param config
        """
        self.config = config

    def __iter__(self) -> Iterable[torch.Tensor]:
        """Generate batched samples."""
        while True:
            yield torch.randn(self.config.batch_size, 1)


class RLSDataset(IterableDataset):
    """Returns indices used for sampling "rows" from the RLS dataset.

    The actual dataset should be within the RLS model instance.
    """

    def __init__(self, config: Config, stochastic: bool):
        """Store params used during sampling.

        Args:
            config: The hyper-param config
            stochastic: Whether to sample mini-batches
        """
        self.config = config
        self.stochastic = stochastic

    def __iter__(self) -> Iterable[torch.Tensor]:
        """Generate batched indices."""
        while True:
            if self.stochastic:
                yield torch.randperm(NUM_EXAMPLES)[: self.config.batch_size]
            else:
                yield torch.arange(NUM_EXAMPLES)
