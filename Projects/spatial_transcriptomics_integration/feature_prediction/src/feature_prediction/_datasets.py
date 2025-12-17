from pathlib import Path
from typing import Any, Callable, Generator

from anndata.typing import AnnData
from exp_runner import Variable
from nico2_lib import datasets

from feature_prediction import typing
from feature_prediction._label_transfer import nmf_transfer

DATA_DIR = "../data"


def _mouse_small_intestine() -> tuple[AnnData, AnnData, str, str]:
    query = datasets.small_mouse_intestine_merfish(DATA_DIR)
    reference = datasets.small_mouse_intestine_sc(DATA_DIR)
    reference.X = reference.X.toarray()
    query_ct_key = "annotation"
    reference_ct_key = "cluster"
    query.obs[query_ct_key] = nmf_transfer(query, reference, reference_ct_key)
    return query, reference, query_ct_key, reference_ct_key


def dataset_generator(
    dir: str,
) -> Generator[Variable[typing.Dataset], Any, None]:
    my_datasets = [
        Variable(
            _mouse_small_intestine,
            {
                "spatial_data_path": "mouse_small_intestine_merfish.h5ad",
                "sc_data_path": "mouse_small_intestine_sc",
            },
        )
    ]
    for dataset in my_datasets:
        yield dataset
