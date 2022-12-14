"""Class definitions for datasets used in robust least squares."""
from typing import Iterable

import torch
from torch.utils.data import IterableDataset

from ..config import Config
from ..models.rls import RLSHighConditionNum, RLSLowConditionNum
from .common import RNGDatasetBase


class RLSDataset(IterableDataset[torch.Tensor], RNGDatasetBase[torch.Tensor]):
    """Returns indices used for sampling "rows" from the RLS dataset.

    The actual dataset should be within the RLS model instance.
    """

    def __init__(self, config: Config, stochastic: bool, mode: str):
        """Store params used during sampling.

        Args:
            config: The hyper-param config
            stochastic: Whether to sample mini-batches
            mode: The target RLS dataset. "low" for low condition number,
                "high" for high condition number.
        """
        super().__init__(config)
        self.stochastic = stochastic

        if mode == "low":
            self.num_examples = RLSLowConditionNum.NUM_EXAMPLES
        elif mode == "high":
            self.num_examples = RLSHighConditionNum.NUM_EXAMPLES
        else:
            raise ValueError(f"Invalid RLS dataset mode: {mode}")

    def __iter__(self) -> Iterable[torch.Tensor]:
        """Generate batched indices."""
        rng = self._get_rng()
        while True:
            if self.stochastic:
                yield torch.randperm(self.num_examples, generator=rng)[
                    : self.config.batch_size
                ]
            else:
                yield torch.arange(self.num_examples)
