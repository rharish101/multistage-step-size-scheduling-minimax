"""Utilities for tuning hyper-parameters."""
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Final, Optional

import numpy as np
import yaml
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, space_eval, tpe
from numpy.random import Generator, default_rng

from .config import Config, update_config
from .train import train

# Where to save the best config after tuning
BEST_CONFIG_FILE: Final = "best-hparams.yaml"

# Where to save the pickle file for hyperopt's progress
TRIALS_FILE: Final = "trials.pkl"

# Bounds for learning rate tuning
_MIN_LR: Final = math.log(1e-6)
_MAX_LR: Final = math.log(1e-2)


@dataclass
class _Progress:
    """Class for saving hyperopt progress."""

    trials: Trials
    rng: Generator


def _get_hparam_space(config: Config) -> Dict[str, Any]:
    """Get the hyper-param tuning space for the given config."""
    return {
        "x_lr": hp.loguniform("x_lr", _MIN_LR, _MAX_LR),
        "y_lr": hp.loguniform("y_lr", _MIN_LR, _MAX_LR),
    }


def tune(
    task: str,
    config: Config,
    tuning_steps: int,
    objective_tag: str,
    num_gpus: int,
    num_workers: int,
    precision: int,
    log_steps: int,
    log_dir: Path,
    run_name: str,
    minimize: bool = True,
    trials_path: Optional[Path] = None,
) -> Config:
    """Tune hyper-params and return the best config.

    Args:
        task: A string specifying the optimization task
        config: The hyper-param config
        tuning_steps: The total steps for tuning the model
        objective_tag: The tag for the metric to be optimized
        num_gpus: The number of GPUs to use (-1 to use all)
        num_workers: The number of workers to use for loading/processing the
            dataset items
        precision: The floating-point precision to use for training the model
        log_steps: The step interval within an epoch for logging
        log_dir: The path to the directory where all logs are to be stored
        run_name: The name for this tuning run
        minimize: Whether the metric is to be minimzed or maximized
        trials_path: The path to the pickled trials file to resume tuning from
            (None to tune from scratch). This overrides `run_name`.

    Returns:
        The metrics for validation at the end of the model
    """
    # The log directory stucture should be as follows:
    # log_dir/task/run_name/eval-{num}/
    # The trials pickle should be at: log_dir/task/run_name/trials.pkl
    if trials_path is not None:
        run_name = trials_path.parent.name

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
            expt_name=run_name,  # Keep all logs inside this folder
            run_name=f"eval-{tuning_iter}",
        )

        metric = metrics[objective_tag]
        return metric if minimize else -metric

    def objective_wrapper(*args, **kwargs) -> Dict[str, Any]:
        try:
            loss = objective(*args, **kwargs)
            status = STATUS_FAIL if np.isnan(loss) else STATUS_OK
        except Exception:
            loss = 0.0
            status = STATUS_FAIL

        return {"loss": loss, "status": status}

    if trials_path is None:
        trials = Trials()
        rng = default_rng(config.seed)
        trials_path = log_dir / task / run_name / TRIALS_FILE
    else:
        with open(trials_path, "rb") as trials_reader:
            progress: _Progress = pickle.load(trials_reader)
        trials = progress.trials
        rng = progress.rng

    space = _get_hparam_space(config)

    # To skip saving the pickle file for previously-completed iterations
    evals_done = len(trials.results)

    try:
        for tuning_iter in range(evals_done, tuning_steps):
            fmin(
                lambda args: objective_wrapper(tuning_iter, args),
                space,
                algo=tpe.suggest,
                trials=trials,
                # We need only one iteration, and we've already finished
                # `tuning_iter` iterations
                max_evals=tuning_iter + 1,
                show_progressbar=False,
                rstate=rng,
            )
            with open(trials_path, "wb") as trials_writer:
                pickle.dump(_Progress(trials, rng), trials_writer)
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt; ending hyper-param search")

    best_hparams = space_eval(space, trials.argmin)
    best_config = update_config(config, best_hparams)

    with open(
        log_dir / task / run_name / BEST_CONFIG_FILE, "w"
    ) as best_config_file:
        yaml.dump(vars(best_config), best_config_file)

    return best_config
