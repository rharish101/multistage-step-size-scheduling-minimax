"""Class definitions for datasets used in robust least squares."""
from typing import Iterable

import torch

from ..config import Config
from ..models.rls import NUM_EXAMPLES
from .common import RNGDatasetBase


class RLSDataset(RNGDatasetBase[torch.Tensor]):
    """Returns indices used for sampling "rows" from the RLS dataset.

    The actual dataset should be within the RLS model instance.
    """

    def __init__(self, config: Config, stochastic: bool):
        """Store params used during sampling.

        Args:
            config: The hyper-param config
            stochastic: Whether to sample mini-batches
        """
        super().__init__(config)
        self.stochastic = stochastic

    def __iter__(self) -> Iterable[torch.Tensor]:
        """Generate batched indices."""
        rng = self._get_rng()
        while True:
            if self.stochastic:
                yield torch.randperm(NUM_EXAMPLES, generator=rng)[
                    : self.config.batch_size
                ]
            else:
                yield torch.arange(NUM_EXAMPLES)
