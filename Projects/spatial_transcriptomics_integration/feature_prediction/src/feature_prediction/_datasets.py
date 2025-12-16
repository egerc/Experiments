from typing import Any, Callable, Generator

from exp_runner import Variable

from feature_prediction import typing


def dataset_generator(
    dir: str,
) -> Generator[Variable[typing.Dataset], Any, None]: ...
