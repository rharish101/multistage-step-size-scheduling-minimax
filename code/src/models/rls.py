"""Class definitions for robust least squares."""
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Final, Tuple

import torch
from pytorch_lightning import LightningModule
from torch.distributions import MultivariateNormal
from torch.nn import Parameter
from torch.optim import SGD

from ..config import Config
from ..schedulers import get_scheduler


class RLSBase(ABC, LightningModule):
    """Base class for a robust least squares model with a soft constraint.

    Adapted from: http://arxiv.org/abs/2002.09621
    """

    def __init__(
        self,
        config: Config,
        stochastic: bool,
        constr_wt: float = 3.0,
        num_examples: int = 1000,
        num_features: int = 500,
        noise_stddev: float = 0.1,
    ):
        """Initialize and store everything needed for training.

        Args:
            config: The hyper-param config
            stochastic: Whether to have stochasticity in the gradients
            constr_wt: The weight of the constraint term
            num_examples: The number of examples in the input matrix
            num_features: The number of features in the input matrix
            noise_stddev: The standard deviation of the added noise
        """
        super().__init__()
        self.config = config
        self.stochastic = stochastic
        self.constr_wt = constr_wt
        self.num_examples = num_examples
        self.num_features = num_features

        self.x = Parameter(torch.randn(num_features, 1))
        self.y = Parameter(torch.randn(num_examples, 1))

        self.A = Parameter(self._get_input_matrix(), requires_grad=False)
        self.M = Parameter(self._get_norm_matrix(), requires_grad=False)

        x_orig = torch.randn(num_features, 1)
        epsilon = torch.randn(num_examples, 1) * noise_stddev
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
        # Stochastic indices to only choose certain "examples" for a batch
        if self.stochastic:
            idxs = torch.randperm(self.num_examples)[: self.config.batch_size]
        else:
            idxs = torch.arange(self.num_examples)

        term_1 = self.A[idxs] @ self.x - self.y[idxs]
        term_2 = self.y[idxs] - self.y_0[idxs]
        term_3 = self.M[idxs][:, idxs]
        loss = (
            term_1.T @ term_3 @ term_1
            - self.constr_wt * term_2.T @ term_3 @ term_2
        )

        if batch_idx % self.trainer.log_every_n_steps == 0:
            with torch.no_grad():
                self._log_rls_metrics(batch_idx)

        if optimizer_idx == 0:  # y update
            return -loss
        else:  # x update
            return loss

    def _log_rls_metrics(self, global_step: int) -> None:
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

        x_dist = (self.x - self.x_star).squeeze()
        y_dist = (self.y - self.y_star).squeeze()
        dist = x_dist.dot(x_dist) + y_dist.dot(y_dist)
        self.log("loss/distance", dist)

        term_1 = self.A @ self.x - self.y
        term_2 = self.y - self.y_0
        f_xy = (
            term_1.T @ self.M @ term_1
            - self.constr_wt * term_2.T @ self.M @ term_2
        )
        term_3 = self.A @ self.x - self.y_0
        g_x = (
            self.constr_wt / (self.constr_wt - 1) * term_3.T @ self.M @ term_3
        )
        potential = 2 * g_x - self.g_star - f_xy
        self.log("loss/potential", potential)


class RLSLowConditionNum(RLSBase):
    """The robust least squares model with a low condition number."""

    def __init__(self, config: Config, stochastic: bool):
        """Initialize and store everything needed for training.

        Args:
            config: The hyper-param config
            stochastic: Whether to have stochasticity in the gradients
        """
        super().__init__(config, stochastic)

    def _get_input_matrix(self) -> torch.Tensor:
        return torch.randn(self.num_examples, self.num_features)

    def _get_norm_matrix(self) -> torch.Tensor:
        return torch.eye(self.num_examples)


class RLSHighConditionNum(RLSBase):
    """The robust least squares model with a high condition number."""

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
        super().__init__(config, stochastic, constr_wt=self.CONSTR_WT)

    def _get_input_matrix(self) -> torch.Tensor:
        A_covar = torch.empty(self.num_features, self.num_features)
        for i in range(self.num_features):
            for j in range(self.num_features):
                A_covar[i, j] = 2 ** (-math.fabs(i - j) / 10)

        A_distr = MultivariateNormal(torch.zeros(self.num_features), A_covar)
        return A_distr.sample([self.num_examples])

    def _get_norm_matrix(self) -> torch.Tensor:
        eigenvecs = torch.linalg.qr(
            torch.randn(self.num_examples, self.num_examples)
        )[0]
        rank = int(self.RANK_FRACTION * self.num_examples)
        eigenvals = torch.cat(
            [
                torch.rand(rank) * (self.EIGENVAL_MAX - self.EIGENVAL_MIN)
                + self.EIGENVAL_MIN,
                torch.zeros(self.num_examples - rank),
            ]
        )
        return eigenvecs.T @ torch.diagflat(eigenvals) @ eigenvecs
