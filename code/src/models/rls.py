"""Class definitions for robust least squares."""
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Final, Tuple

import torch
from pytorch_lightning.utilities.seed import isolate_rng, seed_everything
from torch.distributions import MultivariateNormal
from torch.nn import Parameter
from torch.optim import SGD

from ..config import Config
from ..schedulers import get_scheduler
from .base import BaseModel

DATA_SEED: Final = 0  # The seed for the "dataset"
NOISE_STDDEV: Final = 0.1  # The standard deviation of the added noise


class RLSBase(ABC, BaseModel):
    """Base class for a robust least squares model with a soft constraint.

    Adapted from: http://arxiv.org/abs/2002.09621
    """

    def __init__(
        self,
        config: Config,
        stochastic: bool,
        num_examples: int = 1000,
        num_features: int = 500,
        constr_wt: float = 3.0,
    ):
        """Initialize and store everything needed for training.

        Args:
            config: The hyper-param config
            stochastic: Whether to have stochasticity in the gradients
            num_examples: The number of examples in the input matrix
            num_features: The number of features in the input matrix
            constr_wt: The weight of the constraint term
        """
        super().__init__()
        self.config = config
        self.stochastic = stochastic
        self.constr_wt = constr_wt

        self.x = Parameter(torch.randn(num_features, 1))
        self.y = Parameter(torch.randn(num_examples, 1))

        # Seed the "dataset" separately
        with isolate_rng():
            seed_everything(DATA_SEED)

            self.A = Parameter(self._get_input_matrix(), requires_grad=False)
            self.M = Parameter(self._get_norm_matrix(), requires_grad=False)

            x_orig = torch.randn(num_features, 1)
            epsilon = torch.randn(num_examples, 1) * NOISE_STDDEV

        self.y_0 = Parameter(self.A @ x_orig + epsilon, requires_grad=False)

        term_1 = torch.linalg.pinv(self.A.T @ self.M @ self.A)
        self.x_star = Parameter(
            term_1 @ self.A.T @ self.M @ self.y_0,
            requires_grad=False,
        )

        self.y_star = Parameter(
            (constr_wt * self.y_0 - self.A @ self.x_star) / (constr_wt - 1),
            requires_grad=False,
        )

        term_2 = self.A @ self.x_star - self.y_0
        self.g_star = Parameter(
            self.constr_wt / (self.constr_wt - 1) * term_2.T @ self.M @ term_2,
            requires_grad=False,
        )

    @abstractmethod
    def _get_input_matrix(self) -> torch.Tensor:
        """Return the input matrix for the RLS problem."""

    @abstractmethod
    def _get_norm_matrix(self) -> torch.Tensor:
        """Return the norm matrix for matrix (semi-)norm."""

    def configure_optimizers(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return the optimizers and schedulers for the GAN."""
        x_optim = SGD([self.x], lr=self.config.x_lr)
        y_optim = SGD([self.y], lr=self.config.y_lr)

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
        # Train x first, then y
        return x_config, y_config

    def training_step(
        self, batch: torch.Tensor, batch_idx: int, optimizer_idx: int
    ) -> torch.Tensor:
        """Run one training step."""
        # Stochastic indices to only choose certain "examples" for a batch
        term_1 = self.A[batch] @ self.x - self.y[batch]
        term_2 = self.y[batch] - self.y_0[batch]
        term_3 = self.M[batch][:, batch]
        loss = (
            term_1.T @ term_3 @ term_1
            - self.constr_wt * term_2.T @ term_3 @ term_2
        )

        if batch_idx % self.trainer.log_every_n_steps == 0:
            with torch.no_grad():
                self._log_rls_train_metrics()

        if optimizer_idx == 1:  # y update
            return -loss
        else:  # x update
            return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Run one validation step."""
        self._log_rls_metrics()

    def _log_rls_train_metrics(self) -> None:
        """Log metrics, including training-specific ones."""
        term_1 = self.A @ self.x - self.y
        gx = 2 * self.A.T @ self.M @ term_1
        self.log("gradients/x", torch.linalg.norm(gx))

        gy = 2 * self.M @ (self.constr_wt * (self.y_0 - self.y) - term_1)
        self.log("gradients/y", torch.linalg.norm(gy))

        gen_sched, crit_sched = self.lr_schedulers()
        self.log("learning_rate/x", gen_sched.get_last_lr()[0])
        self.log("learning_rate/y", crit_sched.get_last_lr()[0])

        self._log_rls_metrics()

    def _log_rls_metrics(self) -> None:
        """Log general metrics."""
        x_dist = (self.x - self.x_star).squeeze()
        y_dist = (self.y - self.y_star).squeeze()
        dist = x_dist.dot(x_dist) + y_dist.dot(y_dist)
        self.log("metrics/distance", dist)

        term_1 = self.A @ self.x - self.y
        term_2 = self.y - self.y_0
        f_xy = (
            term_1.T @ self.M @ term_1
            - self.constr_wt * term_2.T @ self.M @ term_2
        )
        self.log("metrics/loss", f_xy)

        term_3 = self.A @ self.x - self.y_0
        g_x = (
            self.constr_wt / (self.constr_wt - 1) * term_3.T @ self.M @ term_3
        )
        potential = 2 * g_x - self.g_star - f_xy
        self.log("metrics/potential", potential)


class RLSLowConditionNum(RLSBase):
    """The robust least squares model with a low condition number."""

    NUM_EXAMPLES: Final = 1000  # The number of examples in the input matrix
    NUM_FEATURES: Final = 500  # The number of features in the input matrix

    def __init__(self, config: Config, stochastic: bool):
        """Initialize and store everything needed for training.

        Args:
            config: The hyper-param config
            stochastic: Whether to have stochasticity in the gradients
        """
        super().__init__(
            config,
            stochastic=stochastic,
            num_examples=self.NUM_EXAMPLES,
            num_features=self.NUM_FEATURES,
        )

    def _get_input_matrix(self) -> torch.Tensor:
        return torch.randn(self.NUM_EXAMPLES, self.NUM_FEATURES)

    def _get_norm_matrix(self) -> torch.Tensor:
        return torch.eye(self.NUM_EXAMPLES)


class RLSHighConditionNum(RLSBase):
    """The robust least squares model with a high condition number."""

    NUM_EXAMPLES: Final = 1000  # The number of examples in the input matrix
    NUM_FEATURES: Final = 500  # The number of features in the input matrix
    CONSTR_WT: Final = 1.5  # The weight of the constraint term
    RANK_FRACTION: Final = 0.95  # The rank as a fraction of dimensionality
    EIGENVAL_MIN: Final = 0.2  # Minimum value of non-zero eigenvalues
    EIGENVAL_MAX: Final = 1.8  # Maximum value of non-zero eigenvalues

    def __init__(self, config: Config, stochastic: bool):
        """Initialize and store everything needed for training.

        Args:
            config: The hyper-param config
            stochastic: Whether to have stochasticity in the gradients
        """
        super().__init__(
            config,
            stochastic,
            num_examples=self.NUM_EXAMPLES,
            num_features=self.NUM_FEATURES,
            constr_wt=self.CONSTR_WT,
        )

    def _get_input_matrix(self) -> torch.Tensor:
        A_covar = torch.empty(self.NUM_FEATURES, self.NUM_FEATURES)
        for i in range(self.NUM_FEATURES):
            for j in range(self.NUM_FEATURES):
                A_covar[i, j] = 2 ** (-math.fabs(i - j) / 10)

        A_distr = MultivariateNormal(torch.zeros(self.NUM_FEATURES), A_covar)
        return A_distr.sample([self.NUM_EXAMPLES])

    def _get_norm_matrix(self) -> torch.Tensor:
        eigenvecs = torch.linalg.qr(
            torch.randn(self.NUM_EXAMPLES, self.NUM_EXAMPLES)
        )[0]
        rank = int(self.RANK_FRACTION * self.NUM_EXAMPLES)
        eigenvals = torch.cat(
            [
                torch.rand(rank) * (self.EIGENVAL_MAX - self.EIGENVAL_MIN)
                + self.EIGENVAL_MIN,
                torch.zeros(self.NUM_EXAMPLES - rank),
            ]
        )
        return eigenvecs.T @ torch.diagflat(eigenvals) @ eigenvecs
