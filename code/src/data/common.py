"""Common classes and utilies for datasets."""
from typing import TypeVar

from torch import Generator
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from ..config import Config

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


class RNGDatasetBase(IterableDataset[T]):
    """Base class for a dataset that yields by sampling with an RNG."""

    def __init__(self, config: Config):
        """Store params used during sampling.

        Args:
            config: The hyper-param config
        """
        self.config = config

    def _get_rng(self) -> Generator:
        """Get an RNG seeded using the initial seed and the worker ID.

        The initial seed will be combined with a worker-specific ID during
        multi-process data sampling.

        This is meant to be run during data loading, not during initialization.
        """
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        return Generator().manual_seed(self.config.seed + worker_id)
