"""Datasets for the toy WGAN."""
from typing import Iterable, TypeVar

import torch
from torch.utils.data import Dataset, IterableDataset

T = TypeVar("T")


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

    def __init__(self, batch_size: int):
        """Store params used during sampling.

        Args:
            batch_size: The size of each batch of samples
        """
        self.batch_size = batch_size

    def __iter__(self) -> Iterable[torch.Tensor]:
        """Generate batched samples."""
        while True:
            yield torch.randn(self.batch_size, 1)
