from functools import wraps
from typing import Callable, Tuple

import nico2_lib as n2l
import numpy as np
import scanpy as sc
from anndata.typing import AnnData

from spatial_transcriptomics_integration.utils import adata_dense_mut


def _create_spatial_loader(
    query_loader: Callable[[str], AnnData],
    query_ct_key: str,
    reference_loader: Callable[[str], AnnData],
    reference_ct_key: str,
) -> Callable[[str], Tuple[AnnData, AnnData, str, str]]:
    """Create a spatial query–reference dataset loader.

    The returned loader loads query and reference AnnData objects, densifies them,
    performs label transfer from the reference to the query, and returns a
    standardized (query, reference, cell-type keys) tuple.

    Args:
        query_loader: Function that loads the spatial query AnnData.
        query_ct_key: Cell type column name to store predictions in the query.
        reference_loader: Function that loads the reference AnnData.
        reference_ct_key: Cell type column name in the reference.

    Returns:
        A loader function mapping a data directory to
        (query, reference, query_ct_key, reference_ct_key).
    """

    def loader(dir: str) -> Tuple[AnnData, AnnData, str, str]:
        query = query_loader(dir)
        reference = reference_loader(dir)

        adata_dense_mut(query)
        adata_dense_mut(reference)

        query.obs[query_ct_key] = n2l.lt.label_transfer_tacco(
            query, reference, reference_ct_key
        )

        return query, reference, query_ct_key, reference_ct_key

    return loader


def _create_pseudospatial_loader(
    loader_func: Callable[[str], AnnData], ct_key: str
) -> Callable[[str], Tuple[AnnData, AnnData, str, str]]:
    """Create a pseudospatial query–reference dataset loader.

    The returned loader randomly splits a single AnnData object into query and
    reference halves. The query is restricted to the top 500 highly variable genes.

    Args:
        loader_func: Function that loads a single AnnData object.
        ct_key: Cell type column name shared by query and reference.

    Returns:
        A loader function mapping a data directory to
        (query, reference, ct_key, ct_key).
    """

    @wraps(loader_func)
    def split_loader(dir: str) -> Tuple[AnnData, AnnData, str, str]:
        adata = loader_func(dir)
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

DATASET_MAPPING: Dict[
    str, exp_runner.Variable[Callable[[str], Tuple[AnnData, AnnData, str, str]]]
] = {
    "mouse_small_intestine_spatial": exp_runner.Variable(
        create_spatial_loader(
            query_loader=n2l.dt.small_mouse_intestine_merfish,
            query_ct_key="annotation",
            reference_loader=n2l.dt.small_mouse_intestine_sc,
            reference_ct_key="cluster",
        ),
        {
            "dataset_id": 0,
            "dataset_name": "small_mouse_intestine_spatial",
            "organism": "mouse",
            "tissue": "small_intestine",
            "dataset_type": "spatial",
        },
    ),
    "mouse_small_intestine_pseudospatial": exp_runner.Variable(
        create_pseudospatial_loader(
            loader_func=n2l.dt.small_mouse_intestine_sc,
            ct_key="cluster",
        ),
        {
            "dataset_id": 1,
            "dataset_name": "small_mouse_intestine_pseudospatial",
            "organism": "mouse",
            "tissue": "small_intestine",
            "dataset_type": "pseudospatial",
        },
    ),
    "human_liver_spatial": exp_runner.Variable(
        create_spatial_loader(
            query_loader=lambda dir: n2l.dt.xenium_10x_loader(
                name="Xenium_V1_hLiver_nondiseased_section_FFPE", dir=dir
            ),
            query_ct_key="annot",
            reference_loader=n2l.dt.human_liver_cell_atlas,
            reference_ct_key="annot",
        ),
        {
            "dataset_id": 2,
            "dataset_name": "human_liver_spatial",
            "organism": "human",
            "tissue": "liver",
            "dataset_type": "spatial",
        },
    ),
    "human_liver_pseudospatial": exp_runner.Variable(
        create_pseudospatial_loader(
            loader_func=n2l.dt.human_liver_cell_atlas,
            ct_key="annot",
        ),
        {
            "dataset_id": 3,
            "dataset_name": "human_liver_pseudospatial",
            "organism": "human",
            "tissue": "liver",
            "dataset_type": "pseudospatial",
        },
    ),
}

__all__ = ["DATASET_MAPPING"]
