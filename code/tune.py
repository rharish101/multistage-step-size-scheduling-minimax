#!/usr/bin/env python
"""Tune hyper-params of a model for the requested task."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

from src.config import load_config
from src.tune import tune
from src.utils import AVAIL_TASKS


def main(args: Namespace) -> None:
    """Run the main function."""
    if args.run_name is None:
        run_name = datetime.now().astimezone().isoformat()
    else:
        run_name = args.run_name

    config = load_config(args.config)
    tune(
        args.task,
        config,
        objective_tag="metrics/distance",
        num_gpus=args.num_gpus,
        num_workers=args.num_workers,
        precision=args.precision,
        log_steps=args.log_steps,
        log_dir=args.log_dir,
        run_name=run_name,
        progress_path=args.resume_path,
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Tune hyper-params of a model for the requested task",
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
        help="Path to a YAML config containing initial hyper-parameter values",
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
        default=0,
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
        "--resume-path",
        type=Path,
        help="Path to the progress file from where to continue",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="The name for this training run (None to use a timestamp)",
    )
    main(parser.parse_args())
