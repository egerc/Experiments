"""
_datasets.py

Provides loaders and generators for single-cell datasets.

Includes functions to load query and reference datasets, optionally split
a dataset randomly, and a generator to iterate over available datasets
wrapped as Variable objects for experiments.
"""

from functools import partial, wraps
from pathlib import Path
from typing import Any, Callable, Generator, List, Tuple

import numpy as np
from anndata.typing import AnnData
from exp_runner import Variable
from nico2_lib import datasets

from feature_prediction import typing
from feature_prediction._label_transfer import nmf_transfer
from feature_prediction.utils import adata_dense_mut


def _mouse_small_intestine(dir: str) -> tuple[AnnData, AnnData, str, str]:
    """
    Load mouse small intestine MERFISH query and scRNA-seq reference datasets.

    The query cell types are annotated using NMF label transfer from the reference.

    Parameters
    ----------
    dir : str
        Path to the data directory.

    Returns
    -------
    tuple[AnnData, AnnData, str, str]
        Tuple containing (query, reference, query_ct_key, reference_ct_key)
    """
    query = datasets.small_mouse_intestine_merfish(dir)
    reference = datasets.small_mouse_intestine_sc(dir)
    adata_dense_mut(reference)
    query_ct_key = "annotation"
    reference_ct_key = "cluster"
    query.obs[query_ct_key] = nmf_transfer(query, reference, reference_ct_key)
    return query, reference, query_ct_key, reference_ct_key


def _split_loader_adata(
    loader_func: Callable[[], AnnData], ct_key: str
) -> typing.Dataset:
    """
    Higher-order function that wraps a loader function to randomly split the AnnData.

    Parameters
    ----------
    loader_func : Callable[[], AnnData]
        Function returning an AnnData object to split.
    ct_key : str
        The cell type column name in the AnnData object.

    Returns
    -------
    Callable[[], Tuple[AnnData, AnnData, str, str]]
        Loader function that returns a tuple of (query, reference, query_ct_key, reference_ct_key)
        with the dataset randomly split in half.
    """

    @wraps(loader_func)
    def split_loader() -> Tuple[AnnData, AnnData, str, str]:
        adata = loader_func()

        adata_dense_mut(adata)
        n_cells = adata.n_obs
        shuffled_idx = np.random.permutation(n_cells)
        split_idx = n_cells // 2

        idx1, idx2 = shuffled_idx[:split_idx], shuffled_idx[split_idx:]

        query = adata[idx1].copy()
        reference = adata[idx2].copy()

        return query, reference, ct_key, ct_key

    return split_loader


def dataset_generator(
    dir: str,
) -> Generator[Variable[typing.Dataset], Any, None]:
    """
    Generate available datasets wrapped as Variable objects for experiments.

    Parameters
    ----------
    dir : str
        Path to the data directory.

    Yields
    ------
    Variable[typing.Dataset]
        Dataset loaders with metadata, ready for use in experiments.
    """
    mouse_small_intestine = partial(_mouse_small_intestine, dir=dir)
    mouse_small_intestine_sc = _split_loader_adata(
        partial(datasets.small_mouse_intestine_sc, dir=dir), "cluster"
    )

    my_datasets: List[Variable[typing.Dataset]] = [
        Variable(
            mouse_small_intestine,
            {
                "spatial_data_path": "mouse_small_intestine_merfish.h5ad",
                "sc_data_path": "mouse_small_intestine_sc",
            },
        ),
        Variable(
            mouse_small_intestine_sc,
            {
                "sc_data_path": "mouse_small_intestine_sc",
            },
        ),
    ]
    for dataset in my_datasets:
        yield dataset
