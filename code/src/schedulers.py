"""Class definitions for custom learning rate schedulers."""
import math
from typing import Any, Dict, Final

from hyperopt import hp
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR, StepLR

from .config import Config

# Bounds for learning rate tuning
_MIN_LR: Final = math.log(1e-6)
_MAX_LR: Final = math.log(1e-2)


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
    elif config.sched == "const":
        return LambdaLR(optim, lambda _: 1.0)
    elif config.sched == "poly":
        return LambdaLR(optim, lambda step: 1 / (1 + config.decay * step))
    elif config.sched == "poly-sqrt":
        return LambdaLR(
            optim, lambda step: 1 / (1 + config.decay * math.sqrt(step))
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


def get_hparam_space(config: Config) -> Dict[str, Any]:
    """Get the hyper-param tuning space for the given config."""
    return {
        "gen_lr": hp.loguniform("gen_lr", _MIN_LR, _MAX_LR),
        "crit_lr": hp.loguniform("crit_lr", _MIN_LR, _MAX_LR),
    }
