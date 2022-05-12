"""Class definitions for custom learning rate schedulers."""
import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR, StepLR

from .config import Config


def get_scheduler(optim: Optimizer, config: Config):
    """Get a scheduler given its name."""
    if config.sched == "step":
        return StepDecay(optim, config.decay, config.total_steps)
    elif config.sched == "var":
        return VariablePhaseStepDecay(
            optim,
            config.decay,
            config.start_phase_len,
            config.total_steps,
        )
    else:
        raise ValueError(f"Invalid scheduler: {config.sched}")


class StepDecay(StepLR):
    """Classic step-decay scheduler."""

    def __init__(self, optim: Optimizer, decay: float, total_steps: int):
        """Initialize the step decay scheduling function.

        Args:
            optim: The main optimizer
            decay: The decay factor
            total_steps: The total steps for training the model
        """
        total_phases = math.log(total_steps, decay)
        phase_length = int(total_steps / total_phases)
        super().__init__(optim, step_size=phase_length, gamma=1 / decay)


class VariablePhaseStepDecay(MultiStepLR):
    """Step-decay scheduler with increasing phase length."""

    def __init__(
        self,
        optim: Optimizer,
        decay: float,
        start_phase_len: int,
        total_steps: int,
    ):
        """Initialize the step decay scheduling function.

        Args:
            optim: The main optimizer
            decay: The decay factor
            start_phase_len: The phase length of the starting phase
            total_steps: The total steps for training the model
        """
        phase_sizes = [start_phase_len]
        curr_phase_len = start_phase_len
        while phase_sizes[-1] < total_steps:
            curr_phase_len = int(curr_phase_len * decay)
            phase_sizes.append(phase_sizes[-1] + curr_phase_len)

        super().__init__(optim, milestones=phase_sizes, gamma=1 / decay)
