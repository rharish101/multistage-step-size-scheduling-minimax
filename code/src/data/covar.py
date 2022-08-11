"""Class definitions for datasets used for covariance matrix learning."""
from typing import Iterable

import torch

from ..models.covar import DIMS
from .common import RNGDatasetBase


class StandardNormalDataset(RNGDatasetBase[torch.Tensor]):
    """Returns samples from the standard normal distribution."""

    def __iter__(self) -> Iterable[torch.Tensor]:
        """Generate batched samples."""
        rng = self._get_rng()
        while True:
            yield torch.randn(self.config.batch_size, DIMS, generator=rng)
