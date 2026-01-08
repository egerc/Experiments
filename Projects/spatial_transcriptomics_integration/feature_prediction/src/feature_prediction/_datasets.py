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
import scanpy as sc
from exp_runner import Variable
from nico2_lib import datasets
from nico2_lib.label_transfer import label_transfer_tacco

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
    query.obs[query_ct_key] = label_transfer_tacco(query, reference, reference_ct_key)
    return query, reference, query_ct_key, reference_ct_key


def _liver_cell_atlas(dir: str) -> Tuple[AnnData, AnnData, str, str]:
    """
    Load Xenium liver section as the query and human liver cell atlas as the reference.

    The query cell types are annotated via NMF label transfer from the reference dataset.

    Parameters
    ----------
    dir : str
        Path to the directory containing the dataset files.

    Returns
    -------
    Tuple[AnnData, AnnData, str, str]
        A tuple containing:
        - query AnnData (Xenium section) with cells to reconstruct
        - reference AnnData (human liver cell atlas) used for prediction
        - query cell type key
        - reference cell type key
    """
    query = datasets.xenium_10x_loader(
        "Xenium_V1_hLiver_nondiseased_section_FFPE", dir=dir
    )
    reference = datasets.human_liver_cell_atlas(dir)
    adata_dense_mut(query)
    adata_dense_mut(reference)
    reference_ct_key = "annot"
    query.obs[reference_ct_key] = nmf_transfer(query, reference, reference_ct_key)
    return query, reference, reference_ct_key, reference_ct_key


def _split_loader_adata(
    loader_func: Callable[[], AnnData], ct_key: str
) -> typing.Dataset:
    """
    Higher-order function that wraps a loader function to randomly split the AnnData.
    The query subset is filtered to the top 500 highly variable genes.

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
        with the dataset randomly split in half and the query filtered to the top 500 HVGs.
    """

    @wraps(loader_func)
    def split_loader() -> Tuple[AnnData, AnnData, str, str]:
        adata = loader_func()
        n_cells = adata.n_obs
        shuffled_idx = np.random.permutation(n_cells)
        split_idx = n_cells // 2

        idx1, idx2 = shuffled_idx[:split_idx], shuffled_idx[split_idx:]
        query = adata[idx1].copy()
        reference = adata[idx2].copy()

        sc.pp.highly_variable_genes(
            query, n_top_genes=500, flavor="seurat_v3", inplace=True
        )
        query = query[:, query.var["highly_variable"]].copy()

        return query, reference, ct_key, ct_key

    return split_loader


def _pbmc3k_processed() -> AnnData:
    adata = sc.read_h5ad(
        "/home/gruengroup/christian/Projects/Experiments/Projects/spatial_transcriptomics_integration/data/pbmc3k_processed/pbmc3k_processed.h5ad"
    )
    return adata


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
    liver_cell_atlas = partial(_liver_cell_atlas, dir=dir)
    liver_cell_atlas_sc = _split_loader_adata(
        partial(datasets.human_liver_cell_atlas, dir=dir), "annot"
    )
    pbmc3k_sc = _split_loader_adata(_pbmc3k_processed, "louvain")

    my_datasets: List[Variable[typing.Dataset]] = [
        Variable(
            mouse_small_intestine,
            {
                "dataset_name": "mouse_small_intestine_spatial",
                "spatial_data_path": "mouse_small_intestine_merfish",
                "spatial_data": "real",
                "sc_data_path": "mouse_small_intestine_sc",
                "spatial_annot_col": "annotation",
                "sc_annot_col": "cluster",
                "organism": "mouse",
                "tissue": "small_intestine",
            },
        ),
         Variable(
            mouse_small_intestine_sc,
            {
                "dataset_name": "mouse_small_intestine_pseudospatial",
                "sc_data_path": "mouse_small_intestine_sc",
                "spatial_data": "pseudo",
                "sc_annot_col": "cluster",
                "organism": "mouse",
                "tissue": "small_intestine",
            },
         ),
         Variable(
            liver_cell_atlas,
            {
                "dataset_name": "liver_cell_atlas_spatial",
                "sc_data_path": "human_liver_cell_atlas",
                "spatial_data": "real",
                "spatial_data_path": "Xenium_V1_hLiver_nondiseased_section_FFPE",
                "spatial_annot_col": "annot",
                "sc_annot_col": "annot",
                "organism": "human",
                "tissue": "liver",
            },
         ),
         Variable(
            liver_cell_atlas_sc,
            {
                "dataset_name": "liver_cell_atlas_pseudospatial",
                "sc_data_path": "human_liver_cell_atlas",
                "spatial_data": "pseudo",
                "sc_annot_col": "annot",
                "organism": "human",
                "tissue": "liver",
            },
         ),
        Variable(
            pbmc3k_sc,
            {
                "dataset_name": "pbmc3k_processed_pseudospatial",
                "spatial_data": "pseudo",
                "sc_data_path": "pbmc3k_processed",
                "sc_annot_col": "louvain",
                "organism": "human",
                "tissue": "pbmcs",
            },
        ),
    ]
    for dataset in my_datasets:
        yield dataset
