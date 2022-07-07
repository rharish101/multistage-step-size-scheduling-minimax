"""Class definitions for WGAN models for covariance matrix learning."""
from typing import Any, Dict, Final, Tuple

import torch
import torch.nn.functional as F
from torch.nn import Bilinear, Linear, Module, Parameter
from torch.optim import SGD

from ..config import Config
from ..schedulers import get_scheduler
from .base import BaseModel

DATA_SEED: Final = 0  # The seed for the "dataset"
DIMS: Final = 3  # The number of dimensions of the generator's output


class LinearGenerator(Module):
    """A linear generator."""

    def __init__(self) -> None:
        """Initialize the model weights."""
        super().__init__()
        self.linear = Linear(DIMS, DIMS, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get the model output."""
        out = self.linear(x)
        return out


class NNGenerator(Module):
    """A non-linear neural-net-based generator."""

    HIDDEN_SIZE: Final = 5

    def __init__(self) -> None:
        """Initialize the model weights."""
        super().__init__()
        self.fc_1 = Linear(DIMS, self.HIDDEN_SIZE)
        self.fc_2 = Linear(self.HIDDEN_SIZE, DIMS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get the model output."""
        return self.fc_2(F.relu(self.fc_1(x)))


class PLCritic(Module):
    """The Polyak–Łojasiewicz critic."""

    def __init__(self) -> None:
        """Initialize the model weights."""
        super().__init__()
        self.bilinear = Bilinear(DIMS, DIMS, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get the model output."""
        return self.bilinear(x, x).squeeze(-1)


class WGAN(BaseModel):
    """The WGAN model."""

    REG_WT: Final = 0.3  # The L2 regularization weight
    REAL_STDDEV: Final = 0.1  # The standard deviation for the "real" data

    def __init__(self, config: Config, gen_type: str):
        """Initialize and store everything needed for training.

        Args:
            config: The hyper-param config
            gen_type: The type of generator (must be one of: linear, nn)
        """
        super().__init__()
        self.config = config

        rng = torch.Generator().manual_seed(DATA_SEED)
        cholesky = (
            self.REAL_STDDEV * torch.randn(DIMS, DIMS, generator=rng).tril()
        )
        cholesky[range(DIMS), range(DIMS)] = cholesky.diag().abs()
        self.cholesky = Parameter(cholesky, requires_grad=False)

        self.critic = PLCritic()
        if gen_type == "linear":
            self.gen = LinearGenerator()
        elif gen_type == "nn":
            self.gen = NNGenerator()
        else:
            raise ValueError(f"Invalid generator type: {gen_type}")

    def configure_optimizers(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return the optimizers and schedulers for the GAN."""
        gen_optim = SGD(self.gen.parameters(), lr=self.config.x_lr)
        crit_optim = SGD(self.critic.parameters(), lr=self.config.y_lr)

        gen_sched = get_scheduler(gen_optim, self.config)
        crit_sched = get_scheduler(crit_optim, self.config)

        gen_config = {
            "optimizer": gen_optim,
            "lr_scheduler": {"scheduler": gen_sched, "interval": "step"},
        }
        crit_config = {
            "optimizer": crit_optim,
            "lr_scheduler": {"scheduler": crit_sched, "interval": "step"},
        }
        # Train the generator first, then the critic
        return gen_config, crit_config

    def training_step(
        self, batch: torch.Tensor, batch_idx: int, optimizer_idx: int
    ) -> torch.Tensor:
        """Run one training step."""
        real = batch @ self.cholesky.T
        fake = self.gen(batch)
        critic_fake = self.critic(fake).reshape(-1)

        if optimizer_idx == 1:  # Critic update
            critic_real = self.critic(real).reshape(-1)
            wass_dist = critic_real.mean() - critic_fake.mean()
            loss = self.REG_WT * self.critic.bilinear.weight.norm() - wass_dist
        else:  # Generator update
            loss = -critic_fake.mean()

        # Log only during the critic update, since only then is the Wasserstein
        # distance calculated
        if (
            batch_idx % self.trainer.log_every_n_steps == 0
            and optimizer_idx == 1
        ):
            with torch.no_grad():
                self._log_gan_train_metrics(fake, wass_dist)

        return loss

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        """Run one validation step."""
        real = batch @ self.cholesky.T
        fake = self.gen(batch)

        critic_fake = self.critic(fake).reshape(-1)
        critic_real = self.critic(real).reshape(-1)
        wass_dist = critic_real.mean() - critic_fake.mean()

        self._log_gan_metrics(fake, wass_dist)

    def _log_gan_train_metrics(
        self, fake: torch.Tensor, wass_dist: torch.Tensor
    ) -> None:
        """Log all metrics.

        Args:
            fake: The output of the generator
            wass_dist: The Wasserstein distance
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
            for p in self.critic.parameters()
            if p.grad is not None
        ]
        if gy_list:
            gy = torch.cat(gy_list)
            self.log("gradients/y", torch.linalg.norm(gy))

        gen_sched, crit_sched = self.lr_schedulers()
        self.log("learning_rate/x", gen_sched.get_last_lr()[0])
        self.log("learning_rate/y", crit_sched.get_last_lr()[0])

        self._log_gan_metrics(fake, wass_dist)

    def _log_gan_metrics(
        self, fake: torch.Tensor, wass_dist: torch.Tensor
    ) -> None:
        self.log("metrics/wasserstein", wass_dist)

        real_cov = self.cholesky @ self.cholesky.T
        distance = (fake.T.cov() - real_cov).norm()
        self.log("metrics/distance", distance)
