"""Common utilities used in the codebase."""
from itertools import product
from typing import Final, Iterable, NoReturn


def _combine_desc(*args: Iterable[str]) -> Iterable[str]:
    """Get the task names by combining allowed names.

    Example:
        >>> list(_combine_desc(["covar"], ["linear", "nn"]))
        ["covar/linear", "covar/nn"]
    """
    return map(lambda x: "/".join(x), product(*args))


AVAIL_TASKS: Final = [
    *_combine_desc(["covar"], ["linear", "nn"]),
    *_combine_desc(["rls"], ["low", "high"], ["full", "stoc"]),
    "cifar10",
]


def invalid_task_error(task: str) -> NoReturn:
    """Raise an error for an invalid task."""
    raise ValueError(f"Invalid task: {task}")
