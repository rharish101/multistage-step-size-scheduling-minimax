#!/usr/bin/env python
"""Train a model for the requested task."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

from src.config import load_config, update_config
from src.models import AVAIL_TASKS
from src.train import train


def main(args: Namespace) -> None:
    """Run the main function."""
    if args.run_name is None:
        run_name = datetime.now().astimezone().isoformat()
    else:
        run_name = args.run_name

    config = load_config(args.config)

    for expt_num in range(args.num_expts):
        expt_config = update_config(config, {"seed": config.seed + expt_num})
        train(
            args.task,
            expt_config,
            num_gpus=args.num_gpus,
            num_workers=args.num_workers,
            precision=args.precision,
            log_steps=args.log_steps,
            log_dir=args.log_dir,
            expt_name=args.task,
            run_name=f"{run_name}/expt-{expt_num}"
            if args.num_expts != 1
            else run_name,
        )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Train a model for the requested task",
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
        help="Step interval for logging training metrics",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="The name for this training run (None to use a timestamp)",
    )
    parser.add_argument(
        "--num-expts",
        type=int,
        default=1,
        help="The total number of experiments to run",
    )
    main(parser.parse_args())
