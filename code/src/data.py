"""Dataloaders for the GAN."""
from typing import Iterable

import torch
from torch.utils.data import IterableDataset


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
