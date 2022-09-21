#!/usr/bin/env python
"""Generate plots for comparing schedulers."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Final, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tbparse import SummaryReader

from src.config import load_config
from src.utils import AVAIL_TASKS

sns.set()

_SMOOTH_ALPHA: Final = 0.4  # The smoothing factor
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


def format_row(row):
    """Format one row of the final metrics in scientific notation.

    Example: 5.23 ± 0.14 x 10^-3
    """
    mean_mantissa, mean_exp = f"{row['mean']:.3e}".split("e")
    if np.isinf(row["std"]) or np.isnan(row["std"]):
        adjusted_std_mantissa = row["std"]
    else:
        std_mantissa, std_exp = f"{row['std']:.3e}".split("e")
        adjusted_std_mantissa = float(std_mantissa) * 10 ** (
            int(std_exp) - int(mean_exp)
        )
    return (
        f"{mean_mantissa} ± {adjusted_std_mantissa:.3f} x "
        f"10^{int(mean_exp)}"
    )


def main(args: Namespace) -> None:
    """Run the main function."""
    data: Optional[pd.DataFrame] = None

    # Sort log dirs for determinism
    for path in sorted(args.log_dir):
        path_data = SummaryReader(path).scalars
        config_path = path / "hparams.yaml"
        if not config_path.exists():
            continue
        config = load_config(config_path)

        if args.mode == "sched":
            name: Any = _SCHED_TO_NAME[config.sched]
        elif args.mode == "decay":
            name = config.decay
        path_data[_MODE_TO_COL[args.mode]] = name

        data = pd.concat([data, path_data], ignore_index=True)

    assert data is not None

    for tag, tag_name in _TAGS_TO_PLOT[args.task.split("/")[0]]:
        tag_data = data[data["tag"] == tag].copy()

        # Smooth the values per-tag and per-"group" (eg. per-scheduler)
        tag_data["smoothed"] = tag_data.groupby(_MODE_TO_COL[args.mode])[
            "value"
        ].apply(lambda x: x.ewm(alpha=_SMOOTH_ALPHA).mean())

        # Get mean and std for the final values
        final_metrics = tag_data.groupby(_MODE_TO_COL[args.mode]).apply(
            lambda df: df.loc[df["step"] == df["step"].max(), "smoothed"].agg(
                ["mean", "std"]
            )
        )

        print(f"{tag}:")
        print(final_metrics.apply(format_row, axis=1))

        axes = sns.lineplot(
            data=tag_data,
            x="step",
            y="smoothed",
            hue=_MODE_TO_COL[args.mode],
            ci="sd",
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
