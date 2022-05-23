"""Class definitions for robust least squares."""
from typing import Any, Dict, Final, Tuple

import torch
from pytorch_lightning import LightningModule
from torch.nn import Parameter
from torch.optim import SGD

from ..config import Config
from ..schedulers import get_scheduler


class RLS(LightningModule):
    """The robust least squares model with a soft constraint.

    Adapted from: http://arxiv.org/abs/2002.09621
    """

    NUM_EXAMPLES: Final = 1000
    NUM_FEATURES: Final = 500
    CONSTR_WT: Final = 3.0
    NOISE_STDDEV: Final = 0.01

    def __init__(self, config: Config):
        """Initialize and store everything needed for training.

        Args:
            config: The hyper-param config
        """
        super().__init__()
        self.config = config

        self.x = Parameter(torch.zeros(self.NUM_FEATURES, 1))
        self.y = Parameter(torch.zeros(self.NUM_EXAMPLES, 1))

        self.A = torch.randn(self.NUM_EXAMPLES, self.NUM_FEATURES)
        self.M = torch.eye(self.NUM_EXAMPLES)

        x_star = torch.randn(self.NUM_FEATURES, 1)
        epsilon = torch.randn(self.NUM_EXAMPLES, 1) * self.NOISE_STDDEV
        self.y_0 = self.A @ x_star + epsilon

    def configure_optimizers(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return the optimizers and schedulers for the GAN."""
        x_optim = SGD([self.x], lr=self.config.gen_lr)
        y_optim = SGD([self.y], lr=self.config.crit_lr)

        x_sched = get_scheduler(x_optim, self.config)
        y_sched = get_scheduler(y_optim, self.config)

        x_config = {
            "optimizer": x_optim,
            "lr_scheduler": {"scheduler": x_sched, "interval": "step"},
        }
        y_config = {
            "optimizer": y_optim,
            "lr_scheduler": {"scheduler": y_sched, "interval": "step"},
        }
        # Train y first, then x
        return y_config, x_config

    def training_step(
        self, batch: torch.Tensor, batch_idx: int, optimizer_idx: int
    ) -> torch.Tensor:
        """Run one training step."""
        term_1 = self.A.to(self.x.device) @ self.x - self.y
        term_2 = self.y - self.y_0.to(self.y.device)
        loss = (
            term_1.T @ self.M.to(term_1.device) @ term_1
            - self.CONST_WT * term_2.T @ self.M.to(term_2.device) @ term_2
        )

        if batch_idx % self.trainer.log_every_n_steps == 0:
            with torch.no_grad():
                self._log_rls_metrics(batch_idx, loss)

        if optimizer_idx == 0:  # y update
            return -loss
        else:  # x update
            return loss

    def _log_rls_metrics(self, global_step: int, loss: torch.Tensor) -> None:
        """Log all metrics."""
        if self.x.grad is not None:
            gx = self.x.grad.detach().reshape(-1)
            self.log("grad_x", torch.linalg.norm(gx))

        if self.y.grad is not None:
            gy = self.y.grad.detach().reshape(-1)
            self.log("grad_y", torch.linalg.norm(gy))

        crit_sched, gen_sched = self.lr_schedulers()
        self.log("learning_rate/generator", gen_sched.get_last_lr()[0])
        self.log("learning_rate/critic", crit_sched.get_last_lr()[0])

        self.log("loss_hist", loss)
