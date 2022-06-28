"""Class definitions for datasets used in toy WGAN models."""
from typing import Iterable

import torch

from .common import RNGDatasetBase


class StandardNormalDataset(RNGDatasetBase[torch.Tensor]):
    """Returns samples from the standard normal distribution."""

    def __iter__(self) -> Iterable[torch.Tensor]:
        """Generate batched samples."""
        rng = self._get_rng()
        while True:
            yield torch.randn(self.config.batch_size, 1, generator=rng)
