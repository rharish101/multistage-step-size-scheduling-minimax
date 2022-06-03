"""Base class for all model definitions."""
from pytorch_lightning import LightningModule
from torch import Tensor


class NaNLossError(RuntimeError):
    """Signifies that the loss was NaN."""


class BaseModel(LightningModule):
    """Base class containing common behaviour for all custom models."""

    def training_step_end(self, step_outputs: Tensor) -> Tensor:
        """Crash if NaN is encountered."""
        combined_loss = step_outputs.sum()
        if combined_loss.isnan():
            raise NaNLossError
        return combined_loss
