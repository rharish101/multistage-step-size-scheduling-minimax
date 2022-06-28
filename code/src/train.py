"""The training function for the models."""
from pathlib import Path
from typing import Dict, Optional

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import isolate_rng, seed_everything
from torch.utils.data import DataLoader

from .config import Config
from .data import get_datasets
from .models import get_model


def train(
    task: str,
    config: Config,
    num_gpus: int,
    num_workers: int,
    precision: int,
    log_steps: int,
    log_dir: Path,
    expt_name: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Dict[str, float]:
    """Train a model for the given task.

    Args:
        task: A string specifying the optimization task
        config: The hyper-param config
        num_gpus: Number of GPUs to use (-1 for all)
        num_workers: Number of worker processes to use for loading data
        precision: Floating-point precision to use (16 implies AMP)
        log_steps: Step interval for logging training metrics
        log_dir: Path to the directory where to save logs and weights
        expt_name: The name for this class of experiments
        run_name: The name for this training run
    """
    # Seed everything, just in case we missed a step using randomness somewhere
    seed_everything(config.seed, workers=True)

    logger = TensorBoardLogger(
        log_dir, name=expt_name, version=run_name, default_hp_metric=False
    )
    logger.log_hyperparams(vars(config))

    # PyTorch layers don't accept a generator argument, so isolate the global
    # RNG
    with isolate_rng():
        seed_everything(config.seed)
        model = get_model(task, config)

    # These should be using RNGs within, so there's no need to isolate the
    # global RNG
    train_dataset, val_dataset = get_datasets(task, config)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=num_gpus != 0,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=num_gpus != 0,
    )

    # Detect if we're using CPUs, because there's no AMP on CPUs
    if num_gpus == 0 or (num_gpus == -1 and not torch.cuda.is_available()):
        precision = max(precision, 32)  # allow 64-bit precision

    trainer = Trainer(
        # Critic and generator have separate steps
        max_steps=config.total_steps * 2,
        # Used to limit the progress bar
        limit_train_batches=config.total_steps,
        logger=logger,
        log_every_n_steps=log_steps,
        gpus=num_gpus,
        auto_select_gpus=num_gpus != 0,
        strategy="ddp",
        precision=precision,
        val_check_interval=1,
    )
    trainer.fit(model, train_dataloaders=train_dataloader)

    # Pytorch Lightning catches KeyboardInterrupt, but doesn't raise it
    if trainer.interrupted:
        raise KeyboardInterrupt

    metrics = trainer.validate(model, dataloaders=val_dataloader)[0]
    return metrics
