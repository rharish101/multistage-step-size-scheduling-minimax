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

sns.set()

_TAGS_TO_PLOT: Final = ("metrics/distance", "metrics/potential")
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

    for tag in _TAGS_TO_PLOT:
        tag_data = data[data["tag"] == tag]
        axes = sns.lineplot(
            data=tag_data, x="step", y="value", hue=_MODE_TO_COL[args.mode]
        )

        tag_trimmed = tag.split("/")[-1]
        axes.set_xlabel("Steps")
        axes.set_ylabel(tag_trimmed.capitalize())
        axes.set_yscale("log")

        fig_path = Path(f"{args.prefix}{tag_trimmed}.png")
        if not fig_path.parent.exists():
            fig_path.parent.mkdir(parents=True)
        plt.savefig(fig_path, bbox_inches="tight")
        axes.clear()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generate plots for comparing schedulers",
        formatter_class=ArgumentDefaultsHelpFormatter,
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
