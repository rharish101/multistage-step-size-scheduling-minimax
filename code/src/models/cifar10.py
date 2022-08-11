"""Class definitions for the CIFAR10 GAN."""
from typing import Any, Dict, Final, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import (
    AdaptiveAvgPool2d,
    AvgPool2d,
    BatchNorm2d,
    BCEWithLogitsLoss,
    Conv2d,
    Flatten,
    Identity,
    Linear,
    Module,
    PixelShuffle,
    ReLU,
    Sequential,
    Tanh,
    Unflatten,
)
from torch.nn.utils.parametrizations import spectral_norm
from torch.optim import Adam

from ..config import Config
from ..data.cifar10 import IMG_DIMS, NOISE_DIMS
from ..schedulers import get_scheduler
from .base import BaseModel


class PixelShuffleSelf(Module):
    """Use PixelShuffle with the input concatenated along the channel axis."""

    def __init__(self, upscale_factor: int):
        """Initialize the pixel shuffle layer."""
        super().__init__()
        self.upscale_factor = upscale_factor
        self.pixel_shuffle = PixelShuffle(upscale_factor)

    def forward(self, x: Tensor) -> Tensor:
        """Pixel-shuffle the inputs."""
        concat_x = torch.concat([x] * self.upscale_factor**2, dim=-3)
        return self.pixel_shuffle(concat_x)


class ResBlockUp(Module):
    """A residual block for the generator.

    The main connection consists of:
        - BatchNorm2d
        - ReLU
        - PixelShuffle (optional)
        - Conv2d
        - BatchNorm2d
        - ReLU
        - Conv2d

    The skip connection optionally consists of:
        - PixelShuffle
        - Conv2d
    """

    def __init__(self, channels: int, upsample: int = 2):
        """Initialize the layers.

        Args:
            channels: The number of channels for the input and output
            upsample: The scale factor for upsampling the inputs
        """
        super().__init__()

        self.main = Sequential(
            BatchNorm2d(channels),
            ReLU(),
            PixelShuffleSelf(upsample) if upsample > 1 else Identity(),
            Conv2d(channels, channels, 3, padding="same", bias=False),
            BatchNorm2d(channels),
            ReLU(),
            Conv2d(channels, channels, 3, padding="same", bias=False),
        )

        if upsample > 1:
            self.skip = Sequential(
                PixelShuffleSelf(upsample),
                Conv2d(channels, channels, 3, padding="same", bias=False),
            )
        else:
            self.skip = Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Get the residual block outputs."""
        return self.main(x) + self.skip(x)


class ResBlockDown(Module):
    """A residual block for the discriminator.

    The main connection consists of:
        - ReLU
        - Conv2d
        - ReLU
        - Conv2d
        - AvgPool2d (optional)

    The skip connection optionally consists of:
        - Conv2d
        - AvgPool2d
    """

    def __init__(
        self,
        out_channels: int,
        in_channels: Optional[int] = None,
        downsample: int = 2,
    ):
        """Initialize the layers.

        Args:
            out_channels: The number of channels for the output
            in_channels: The number of channels for the input
            downsample: The scale factor for downsampling the inputs
        """
        super().__init__()

        if in_channels is None:
            in_channels = out_channels

        self.main = Sequential(
            ReLU(),
            spectral_norm(
                Conv2d(in_channels, out_channels, 3, padding="same")
            ),
            ReLU(),
            spectral_norm(
                Conv2d(out_channels, out_channels, 3, padding="same")
            ),
            AvgPool2d(downsample) if downsample > 1 else Identity(),
        )

        self.skip = Sequential()
        if downsample > 1 or in_channels != out_channels:
            self.skip.append(
                spectral_norm(
                    Conv2d(in_channels, out_channels, 3, padding="same")
                )
            )
        if downsample > 1:
            self.skip.append(AvgPool2d(downsample))

    def forward(self, x: Tensor) -> Tensor:
        """Get the residual block outputs."""
        return self.main(x) + self.skip(x)


class Generator(Module):
    """A CNN-based generator with residual connections."""

    CHANNELS: Final = 256

    def __init__(self) -> None:
        """Initialize the layers."""
        super().__init__()
        self.model = Sequential(
            Linear(NOISE_DIMS, self.CHANNELS * 4 * 4, bias=False),
            Unflatten(-1, (self.CHANNELS, 4, 4)),
            ResBlockUp(self.CHANNELS),
            ResBlockUp(self.CHANNELS),
            ResBlockUp(self.CHANNELS),
            BatchNorm2d(self.CHANNELS),
            ReLU(),
            Conv2d(self.CHANNELS, IMG_DIMS, 3, padding="same"),
            Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Get the model output."""
        return self.model(x)


class Discriminator(Module):
    """A CNN-based discriminator with residual connections."""

    CHANNELS: Final = 128

    def __init__(self) -> None:
        """Initialize the layers."""
        super().__init__()
        self.model = Sequential(
            ResBlockDown(self.CHANNELS, in_channels=IMG_DIMS),
            ResBlockDown(self.CHANNELS),
            ResBlockDown(self.CHANNELS, downsample=1),
            ResBlockDown(self.CHANNELS, downsample=1),
            ReLU(),
            AdaptiveAvgPool2d(1),
            Flatten(start_dim=-3),
            spectral_norm(Linear(self.CHANNELS, 1)),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Get the model output."""
        return self.model(x)


class CIFAR10GAN(BaseModel):
    """The CIFAR10 GAN model."""

    def __init__(self, config: Config):
        """Initialize and store everything needed for training.

        Args:
            config: The hyper-param config
        """
        super().__init__()
        self.config = config

        self.gen = Generator()
        self.disc = Discriminator()
        self.loss = BCEWithLogitsLoss()

    def configure_optimizers(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return the optimizers and schedulers for the GAN."""
        gen_optim = Adam(
            self.gen.parameters(),
            lr=self.config.x_lr,
            betas=(self.config.x_momentum, self.config.x_adaptivity),
        )
        disc_optim = Adam(
            self.disc.parameters(),
            lr=self.config.y_lr,
            betas=(self.config.x_momentum, self.config.x_adaptivity),
        )

        gen_sched = get_scheduler(gen_optim, self.config)
        disc_sched = get_scheduler(disc_optim, self.config)

        gen_config = {
            "optimizer": gen_optim,
            "lr_scheduler": {"scheduler": gen_sched, "interval": "step"},
        }
        disc_config = {
            "optimizer": disc_optim,
            "lr_scheduler": {"scheduler": disc_sched, "interval": "step"},
        }
        # Train the discriminator first, then the generator
        return disc_config, gen_config

    def training_step(
        self, batch: Tensor, batch_idx: int, optimizer_idx: int
    ) -> Tensor:
        """Run one training step."""
        noise, real = batch
        fake = self.gen(noise)
        disc_fake = self.disc(fake).reshape(-1)

        if optimizer_idx == 0:  # Discriminator update
            disc_real = self.disc(real).reshape(-1)
            loss = self.loss(disc_fake, torch.zeros_like(disc_fake))
            loss += self.loss(disc_real, torch.ones_like(disc_real))
        else:  # Generator update
            loss = self.loss(disc_fake, torch.ones_like(disc_fake))

        # Log only during the critic update, since only then is the total GAN
        # loss calculated
        if (
            batch_idx % self.trainer.log_every_n_steps == 0
            and optimizer_idx == 0
        ):
            with torch.no_grad():
                self._log_gan_train_metrics(-loss)

        return loss

    def _log_gan_train_metrics(self, loss: Tensor) -> None:
        """Log all metrics.

        Args:
            loss: The overall GAN loss
        """
        gx_list = [
            p.grad.detach().reshape(-1)
            for p in self.gen.parameters()
            if p.grad is not None
        ]
        if gx_list:
            gx = torch.cat(gx_list)
            self.log("gradients/x", torch.linalg.norm(gx))

        gy_list = [
            p.grad.detach().reshape(-1)
            for p in self.disc.parameters()
            if p.grad is not None
        ]
        if gy_list:
            gy = torch.cat(gy_list)
            self.log("gradients/y", torch.linalg.norm(gy))

        gen_sched, disc_sched = self.lr_schedulers()
        self.log("learning_rate/x", gen_sched.get_last_lr()[0])
        self.log("learning_rate/y", disc_sched.get_last_lr()[0])

        self.log("metrics/gan_loss", loss)
