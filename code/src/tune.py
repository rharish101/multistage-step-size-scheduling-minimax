"""Utilities for tuning hyper-parameters."""
import pickle
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from traceback import print_exception
from typing import Any, Callable, Dict, Final, List, Optional

import numpy as np
import yaml

from .config import Config, update_config
from .train import train

# File name for the best config after tuning
BEST_CONFIG_FILE: Final = "best-hparams.yaml"

# File name for the tuning progress pickle file
PROGRESS_FILE: Final = "progress.pkl"

# Grid of scaling for learning rate tuning
_LR_SCALE_GRID: Final = 3.0 ** np.arange(-4, 5)

# Type for hyper-params
HyperParamsType = Dict[str, Any]


@dataclass
class Progress:
    """Class for saving tuning progress."""

    inputs: List[HyperParamsType] = field(default_factory=list)
    losses: List[Optional[float]] = field(default_factory=list)
    argmin: Optional[HyperParamsType] = None
    amin: Optional[float] = None


def _get_hparam_space(config: Config) -> Dict[str, List[Any]]:
    """Get the hyper-param tuning space for the given config."""
    return {
        "x_lr": (config.x_lr * _LR_SCALE_GRID).tolist(),
        "y_lr": (config.y_lr * _LR_SCALE_GRID).tolist(),
    }


def grid_search(
    objective: Callable[[int, HyperParamsType], Optional[float]],
    space: Dict[str, List[Any]],
    progress_path: Optional[Path] = None,
) -> Progress:
    """Perform grid search to find the lowest value of the objective.

    A KeyboardInterrupt can be sent to terminate grid search early.

    Args:
        objective: The objective function to minimize. The first argument
            should be the current iteration number, and the second argument
            should be the hyper-params.
        space: The mapping of hyper-param names to the list of possible values
            in the grid
        progress_path: The path where to save the progress after every
            evaluation. If a progress file already exists, then evaluation is
            continued from where it left off.

    Returns:
        The progress of the hyper-param search so far.
    """
    if progress_path is None:
        progress = Progress()
    elif progress_path.exists():
        with open(progress_path, "rb") as progress_reader:
            progress = pickle.load(progress_reader)
    else:
        progress = Progress()
        progress_path.parent.mkdir(parents=True, exist_ok=True)

    names, individual_grids = zip(*space.items())
    try:
        for tuning_iter, values in enumerate(product(*individual_grids)):
            if tuning_iter < len(progress.losses):
                continue

            hyper_params = dict(zip(names, values))
            progress.inputs.append(hyper_params)
            loss = objective(tuning_iter, hyper_params)
            progress.losses.append(loss)

            if loss is not None and (
                progress.amin is None or loss < progress.amin
            ):
                progress.amin = loss
                progress.argmin = hyper_params

            if progress_path is not None:
                with open(progress_path, "wb") as progress_writer:
                    pickle.dump(progress, progress_writer)
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt; ending hyper-param search")

    return progress


def tune(
    task: str,
    config: Config,
    objective_tag: str,
    num_gpus: int,
    num_workers: int,
    precision: int,
    log_steps: int,
    log_dir: Path,
    run_name: str,
    minimize: bool = True,
    progress_path: Optional[Path] = None,
) -> Config:
    """Tune hyper-params and return the best config.

    Args:
        task: A string specifying the optimization task
        config: The hyper-param config
        objective_tag: The tag for the metric to be optimized
        num_gpus: The number of GPUs to use (-1 to use all)
        num_workers: The number of workers to use for loading/processing the
            dataset items
        precision: The floating-point precision to use for training the model
        log_steps: The step interval within an epoch for logging
        log_dir: The path to the directory where all logs are to be stored
        run_name: The name for this tuning run
        minimize: Whether the metric is to be minimzed or maximized
        progress_path: The path to the pickled progress file to resume tuning
            from (None to tune from scratch). This overrides `run_name`.

    Returns:
        The metrics for validation at the end of the model
    """
    # The log directory stucture should be as follows:
    # log_dir/task/run_name/eval-{num}/
    # The progress pickle should be at: log_dir/task/run_name/progress.pkl
    if progress_path is not None:
        progress_path = progress_path.resolve()
        log_dir = log_dir.resolve()
        try:
            run_name = str(progress_path.parent.relative_to(log_dir / task))
        except ValueError:
            raise ValueError(
                f'Invalid progress path "{progress_path}"; it should be a '
                f'sub-directory of "{log_dir / task}"'
            )

    def objective(tuning_iter: int, hparams: Dict[str, Any]) -> float:
        new_config = update_config(config, hparams)
        metrics = train(
            task,
            new_config,
            num_gpus=num_gpus,
            num_workers=num_workers,
            precision=precision,
            log_steps=log_steps,
            log_dir=log_dir / task,
            val_steps=0,  # Speed up the execution by skipping validation
            expt_name=run_name,  # Keep all logs inside this folder
            run_name=f"eval-{tuning_iter}",
            save_ckpt=False,  # Save disk space
            progress_bar=False,  # Reduce unnecessary output
        )

        metric = metrics[objective_tag]
        return metric if minimize else -metric

    def objective_wrapper(*args, **kwargs) -> Optional[float]:
        try:
            raw_loss = objective(*args, **kwargs)
            loss = None if np.isnan(raw_loss) else raw_loss
        except Exception as ex:
            print_exception(type(ex), ex, ex.__traceback__)
            loss = None
        return loss

    if progress_path is None:
        progress_path = log_dir / task / run_name / PROGRESS_FILE

    progress = grid_search(
        objective_wrapper,
        space=_get_hparam_space(config),
        progress_path=progress_path,
    )

    if progress.argmin is None:
        raise RuntimeError("Failed to run hyper-param search")

    best_config = update_config(config, progress.argmin)
    with open(
        log_dir / task / run_name / BEST_CONFIG_FILE, "w"
    ) as best_config_file:
        yaml.dump(vars(best_config), best_config_file)

    return best_config
