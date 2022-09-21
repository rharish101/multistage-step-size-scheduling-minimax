#!/usr/bin/env python
"""Generate plots for comparing schedulers."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Final, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tbparse import SummaryReader

from src.config import load_config
from src.utils import AVAIL_TASKS

sns.set()

_TAGS_TO_PLOT: Final = {
    "rls": [
        ("metrics/distance", "Distance"),
        ("metrics/potential", "Potential"),
    ],
    "covar": [
        ("metrics/distance", "Distance"),
        ("gradients/x", "Gradients w.r.t. X"),
    ],
    "cifar10": [
        ("metrics/fid", "FID"),
        ("metrics/inception_score", "Inception Score"),
    ],
}  # The metrics to plot (as separate plots), with their names
_MODE_TO_COL: Final = {
    "sched": "Scheduler",
    "decay": "Decay",
}  # Used for the legend title
_SCHED_TO_NAME: Final = {
    "const": "Constant",
    "step": "Step-decay",
    "var": "Increasing-phase",
    "poly": "Poly-linear",
    "poly-sqrt": "Poly-sqrt",
}  # Used for naming schedulers in the legend


def main(args: Namespace) -> None:
    """Run the main function."""
    data: Optional[pd.DataFrame] = None

    # Sort log dirs for determinism
    for path in sorted(args.log_dir):
        path_data = SummaryReader(path).scalars
        config = load_config(path / "hparams.yaml")

        if args.mode == "sched":
            name: Any = _SCHED_TO_NAME[config.sched]
        elif args.mode == "decay":
            name = config.decay
        path_data[_MODE_TO_COL[args.mode]] = name

        data = pd.concat([data, path_data], ignore_index=True)

    assert data is not None

    for tag, tag_name in _TAGS_TO_PLOT[args.task.split("/")[0]]:
        tag_data = data[data["tag"] == tag]
        axes = sns.lineplot(
            data=tag_data, x="step", y="value", hue=_MODE_TO_COL[args.mode]
        )

        axes.set_xlabel("Steps")
        axes.set_ylabel(tag_name)
        axes.set_yscale("log")

        tag_trimmed = tag.replace("/", "-")
        fig_path = Path(f"{args.prefix}{tag_trimmed}.pdf")
        if not fig_path.parent.exists():
            fig_path.parent.mkdir(parents=True)
        plt.savefig(fig_path, bbox_inches="tight", pad_inches=0)
        axes.clear()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generate plots for comparing schedulers",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "task",
        choices=AVAIL_TASKS,
        help="A string specifying the optimization task",
    )
    parser.add_argument(
        "mode",
        choices=["sched", "decay"],
        help="The hyper-parameter to use for grouping",
    )
    parser.add_argument(
        "log_dir",
        nargs="+",
        type=Path,
        help="Path to the directories containing training logs",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="plots/",
        help="The prefix for naming the figure paths",
    )
    main(parser.parse_args())
