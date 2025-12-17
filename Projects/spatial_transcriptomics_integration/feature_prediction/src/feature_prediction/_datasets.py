from pathlib import Path
from typing import Any, Callable, Generator

from exp_runner import Variable
from nico2_lib import datasets

from feature_prediction import typing

DATA_DIR = "../data"


def _mouse_small_intestine():
    query = datasets.small_mouse_intestine_merfish(DATA_DIR)
    reference = datasets.small_mouse_intestine_sc(DATA_DIR)
    return query, reference


def dataset_generator(
    dir: str,
) -> Generator[Variable[typing.Dataset], Any, None]:
    datasets = [
        Variable(
            _mouse_small_intestine,
            {
                "spatial_data_path": "mouse_small_intestine_merfish.h5ad",
                "sc_data_path": "mouse_small_intestine_sc",
            },
        )
    ]
    for dataset in datasets:
        yield dataset
