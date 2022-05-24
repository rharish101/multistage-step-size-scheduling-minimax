#!/usr/bin/env python
"""Train a GAN model."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from src.config import load_config
from src.data import StandardNormalDataset
from src.models import AVAIL_TASKS, get_model


def main(args: Namespace) -> None:
    """Run the main function."""
    config = load_config(args.config)
    seed_everything(config.seed, workers=True)

    if args.run_name is None:
        run_name = datetime.now().astimezone().isoformat()
    else:
        run_name = args.run_name
    logger = TensorBoardLogger(
        args.log_dir, name=args.task, version=run_name, default_hp_metric=False
    )
    logger.log_hyperparams(vars(config))

    model = get_model(args.task, config)

    # Detect if we're using CPUs, because there's no AMP on CPUs
    if args.num_gpus == 0 or (
        args.num_gpus == -1 and not torch.cuda.is_available()
    ):
        precision = max(args.precision, 32)  # allow 64-bit precision
    else:
        precision = args.precision

    dataloader = DataLoader(
        StandardNormalDataset(config.batch_size),
        batch_size=None,
        num_workers=args.num_workers,
        pin_memory=args.num_gpus != 0,
    )

    trainer = Trainer(
        # Critic and generator have separate steps
        max_steps=config.total_steps * 2,
        # Used to limit the progress bar
        limit_train_batches=config.total_steps,
        logger=logger,
        log_every_n_steps=args.log_steps,
        gpus=args.num_gpus,
        auto_select_gpus=args.num_gpus != 0,
        strategy="ddp",
        precision=precision,
        val_check_interval=1,
    )
    trainer.fit(model, train_dataloaders=dataloader)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Train a GAN model",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "task",
        choices=AVAIL_TASKS,
        help="A string specifying the optimization task",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to a YAML config containing hyper-parameter values",
    )
    parser.add_argument(
        "-g",
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (-1 for all)",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=int,
        default=16,
        help="Floating-point precision to use (16 implies AMP)",
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes to use for loading data",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default="logs",
        help="Path to the directory where to save logs and weights",
    )
    parser.add_argument(
        "--log-steps",
        type=int,
        default=50,
        help="Step interval (within an epoch) for logging training metrics",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="The name for this run (None to use a timestamp)",
    )
    main(parser.parse_args())
