"""Class definition for the CIFAR10 dataset."""
from math import ceil
from typing import Final, Iterable, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import IterableDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from ..config import Config
from .common import RNGDatasetBase

# The type of each item in the dataset
CIFAR10ItemType = Tuple[Tensor, Tensor]

NOISE_DIMS: Final = 128  # The number of dimensions of the generator's input
IMG_DIMS: Final = 3  # The number of dimensions of the generator's output

# The mean and standard deviation for each image channel
IMG_MEAN: Final = (0.4914, 0.4822, 0.4465)
IMG_STD_DEV: Final = (0.2470, 0.2435, 0.2616)


def get_transform() -> Module:
    """Get the image augmentations for CIFAR10.

    Returns:
        The image transformation function
    """
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD_DEV),
    ]
    return transforms.Compose(transform_list)


class CIFAR10TrainDataset(
    IterableDataset[CIFAR10ItemType], RNGDatasetBase[CIFAR10ItemType]
):
    """Dataset for training a GAN on CIFAR10."""

    def __init__(self, config: Config):
        """Load the CIFAR10 dataset."""
        super().__init__(config)
        self.data = CIFAR10(
            "datasets",
            train=True,
            transform=get_transform(),
            download=True,
        )

    def __iter__(self) -> Iterable[CIFAR10ItemType]:
        """Generate batched samples.

        Yields:
            A batch of random noise for the generator
            A batch of CIFAR10 images
        """
        rng = self._get_rng()
        while True:
            for start_idx in range(0, len(self.data), self.config.batch_size):
                noise = torch.randn(
                    self.config.batch_size, NOISE_DIMS, generator=rng
                )
                data = [
                    self.data[idx]
                    for idx in range(
                        start_idx,
                        min(
                            len(self.data), start_idx + self.config.batch_size
                        ),
                    )
                ]
                img, classes = zip(*data)
                yield noise, torch.stack(img)


class CIFAR10ValDataset(RNGDatasetBase[CIFAR10ItemType]):
    """Dataset for validating a GAN on CIFAR10."""

    def __init__(self, config: Config):
        """Load the CIFAR10 dataset."""
        super().__init__(config)
        self.data = CIFAR10(
            "datasets",
            train=False,
            transform=get_transform(),
            download=True,
        )

        rng = self._get_rng()
        self.noises = []
        for _ in range(0, len(self.data), self.config.batch_size):
            noise = torch.randn(
                self.config.batch_size, NOISE_DIMS, generator=rng
            )
            self.noises.append(noise)

    def __len__(self) -> int:
        """Return the length of the batched dataset."""
        return ceil(len(self.data) / self.config.batch_size)

    def __getitem__(self, index: int) -> CIFAR10ItemType:
        """Return the sample for the given index.

        Returns:
            A batch of random noise for the generator
            A batch of CIFAR10 images
        """
        start_idx = index * self.config.batch_size
        data = [
            self.data[idx]
            for idx in range(
                start_idx,
                min(len(self.data), start_idx + self.config.batch_size),
            )
        ]
        img, classes = zip(*data)
        return self.noises[index], torch.stack(img)
