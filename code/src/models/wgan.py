"""Class definitions for toy WGAN models."""
from typing import Any, Dict, Final, Tuple

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.nn import Linear, Module, Parameter
from torch.optim import SGD

from ..config import Config
from ..schedulers import get_scheduler


class LinearGenerator(Module):
    """A linear generator."""

    def __init__(self) -> None:
        """Initialize the model weights."""
        super().__init__()
        self.linear = Linear(1, 1)
        self.linear.weight.data.fill_(1)
        self.linear.bias.data.fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get the model output."""
        out = self.linear(x)
        return out


class NNGenerator(Module):
    """A non-linear neural-net-based generator."""

    def __init__(self) -> None:
        """Initialize the model weights."""
        super().__init__()
        self.fc_1 = Linear(1, 5)
        self.fc_2 = Linear(5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get the model output."""
        return self.fc_2(F.relu(self.fc_1(x)))


class PLCritic(Module):
    """The Polyak–Łojasiewicz critic."""

    def __init__(self) -> None:
        """Initialize the model weights."""
        super().__init__()
        self.theta_1 = Parameter(torch.zeros(1))
        self.theta_2 = Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get the model output."""
        return self.theta_1 * x + self.theta_2 * x**2


class WGAN(LightningModule):
    """The WGAN model."""

    REAL_MEAN: Final = 0.0  #: The mean of the "real" data
    REAL_STDDEV: Final = 0.1  #: The standard deviation for the "real" data

    def __init__(self, config: Config, gen_type: str):
        """Initialize and store everything needed for training.

        Args:
            config: The hyper-param config
            gen_type: The type of generator (must be one of: linear, nn)
        """
        super().__init__()
        self.config = config

        self.critic = PLCritic()
        if gen_type == "linear":
            self.gen = LinearGenerator()
        elif gen_type == "nn":
            self.gen = NNGenerator()
        else:
            raise ValueError(f"Invalid generator type: {gen_type}")

    def configure_optimizers(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return the optimizers and schedulers for the GAN."""
        gen_optim = SGD(self.gen.parameters(), lr=self.config.gen_lr)
        crit_optim = SGD(self.critic.parameters(), lr=self.config.crit_lr)

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
        # Train the critic first, then the generator
        return crit_config, gen_config

    def training_step(
        self, batch: torch.Tensor, batch_idx: int, optimizer_idx: int
    ) -> torch.Tensor:
        """Run one training step."""
        real = self.REAL_MEAN + self.REAL_STDDEV * batch
        fake = self.gen(batch)
        critic_fake = self.critic(fake).reshape(-1)

        if batch_idx % self.trainer.log_every_n_steps == 0:
            with torch.no_grad():
                self.log_gan_metrics(fake)

        if optimizer_idx == 0:  # Critic update
            critic_real = self.critic(real).reshape(-1)
            return (
                -critic_real.mean()
                + critic_fake.mean()
                + 0.001 * (self.critic.theta_1**2 + self.critic.theta_2**2)
            )

        else:  # Generator update
            return -critic_fake.mean()

    def log_gan_metrics(self, fake: torch.Tensor) -> None:
        """Log all metrics."""
        gx_list = [
            p.grad.detach().reshape(-1)
            for p in self.critic.parameters()
            if p.grad is not None
        ]
        if gx_list:
            gx = torch.cat(gx_list)
            self.log("grad_x", torch.linalg.norm(gx))

        gy_list = [
            p.grad.detach().reshape(-1)
            for p in self.gen.parameters()
            if p.grad is not None
        ]
        if gy_list:
            gy = torch.cat(gy_list)
            self.log("grad_y", torch.linalg.norm(gy))

        loss_hist = (
            torch.abs(fake.mean() - self.REAL_MEAN) ** 2
            + torch.abs(fake.std() - self.REAL_STDDEV) ** 2
        )
        self.log("loss_hist", loss_hist)

        crit_sched, gen_sched = self.lr_schedulers()
        self.log("learning_rate/generator", gen_sched.get_last_lr()[0])
        self.log("learning_rate/critic", crit_sched.get_last_lr()[0])
