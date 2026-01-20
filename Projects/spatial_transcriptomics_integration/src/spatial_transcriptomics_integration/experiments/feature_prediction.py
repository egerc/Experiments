from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, List, Optional, Sequence, Tuple
from uuid import uuid4

import exp_runner
import nico2_lib as n2l
import numpy as np
import scanpy as sc
from anndata.typing import AnnData

from spatial_transcriptomics_integration.dataloaders import DATASET_MAPPING
from spatial_transcriptomics_integration.predictor_functions import PREDICTOR_MAPPING
from spatial_transcriptomics_integration.utils import adata_dense_mut


@dataclass
class _Input(exp_runner.VarProduct):
    datasets: Tuple[AnnData, AnnData]
    predictor: Callable[[AnnData, AnnData], AnnData]
    seed: Optional[int] = None
    predicted_genes_count: int = 10
    artifact_dir: Optional[str] = None


@exp_runner.runner()
def _gene_pred_benchmark(input: _Input) -> List[exp_runner.MetaData]:
    rng = np.random.default_rng(input.seed)
    query, reference = input.datasets
    shared_genes = np.intersect1d(query.var_names, reference.var_names).tolist()
    if input.predicted_genes_count > len(shared_genes):
        raise ValueError(
            "predicted_genes_count exceeds the number of shared genes between query "
            "and reference."
        )
    predicted_genes = rng.choice(
        shared_genes, size=input.predicted_genes_count, replace=False
    ).tolist()
    training_genes = np.setdiff1d(shared_genes, predicted_genes)
    query_recon = input.predictor(query[:, training_genes], reference[:, shared_genes])
    barcodes = query.obs_names
    artifact_folder = (
        Path(input.artifact_dir)
        if input.artifact_dir is not None
        else Path.cwd() / "artifacts"
    )
    artifact_folder.mkdir(parents=True, exist_ok=True)
    query_path = artifact_folder / f"{uuid4().hex}.h5ad"
    query_recon_path = artifact_folder / f"{uuid4().hex}.h5ad"
    query.write_h5ad(query_path)
    query_recon.write_h5ad(query_recon_path)

    res = []
    for barcode in barcodes:
        cell = query[barcode, predicted_genes].X
        cell_recon = query_recon[barcode, predicted_genes].X
        res.append(
            {
                "barcode": str(barcode),
                "query_path": str(query_path),
                "query_recon_path": str(query_recon_path),
                "mse": n2l.mt.mse_metric(cell, cell_recon),
                "pearsonr": n2l.mt.pearson_metric(cell, cell_recon),
                "spearmanr": n2l.mt.spearman_metric(cell, cell_recon),
                "cosine": n2l.mt.cosine_similarity_metric(cell, cell_recon),
                "exp_var_scikit": n2l.mt.explained_variance_metric(cell, cell_recon),
                "exp_var_dominic": n2l.mt.explained_variance_metric_v2(
                    cell, cell_recon
                ),
            }
        )
    return res


def _dataset_generator(
    dataset_keys: Sequence[str],
    dir: str,
    n_obs_ceiling: Optional[int] = None,
    include_full: bool = True,
    include_celltype_subsets: bool = True,
) -> Generator[exp_runner.Variable[Tuple[AnnData, AnnData]], None, None]:
    """
    Generates full datasets and then cell-type specific subsets for given dataset keys.

    For each dataset specified in `dataset_keys`, this generator first yields the full
    query and reference AnnData objects. Subsequently, for each cell type common to
    both the query and reference, it yields subsets of these AnnData objects,
    containing only cells of that specific common cell type.

    Args:
        dataset_keys: A list of strings, where each string is a key identifying a
            dataset in the global `DATASET_MAPPING`.
        dir: The directory path where data files are located.

    Yields:
        A tuple `(query_adata, reference_adata, query_ct_key, reference_ct_key)`.
        The yielded items are either the full query and reference AnnData objects,
        or AnnData objects subsetted to a common cell type, along with their
        respective cell type observation keys.

    Raises:
        ValueError: If a `dataset_key` is not found in `DATASET_MAPPING`.
    """
    for dataset_key in dataset_keys:
        if dataset_key not in DATASET_MAPPING:
            raise ValueError(
                f"Dataset key '{dataset_key}' not found in DATASET_MAPPING."
            )

        loader_variable = DATASET_MAPPING[dataset_key]
        query, reference, query_ct_key, reference_ct_key = loader_variable.value(
            dir=dir
        )
        if n_obs_ceiling is not None:
            n_obs_query = min(query.n_obs, n_obs_ceiling)
            n_obs_ref = min(reference.n_obs, n_obs_ceiling)
            sc.pp.subsample(query, n_obs=n_obs_query)
            sc.pp.subsample(reference, n_obs=n_obs_ref)
        adata_dense_mut(query)
        adata_dense_mut(reference)

        if include_full:
            yield exp_runner.Variable(
                (query, reference),
                metadata={
                    **loader_variable.metadata,
                    **{"celltype": "all", "reconstruction_strategy": "at_once"},
                },
            )

        query_cell_types = query.obs[query_ct_key].unique()
        reference_cell_types = reference.obs[reference_ct_key].unique()

        common_cell_types = sorted(
            list(set(query_cell_types) & set(reference_cell_types))
        )

        if include_celltype_subsets:
            for ct in common_cell_types:
                query_subset = query[query.obs[query_ct_key] == ct].copy()
                reference_subset = reference[
                    reference.obs[reference_ct_key] == ct
                ].copy()
                yield exp_runner.Variable(
                    (query_subset, reference_subset),
                    metadata={
                        **loader_variable.metadata,
                        **{"celltype": ct, "reconstruction_strategy": "per_celltype"},
                    },
                )


def _predictor_generator(predictor_keys: Optional[Sequence[str]] = None):
    keys = predictor_keys or PREDICTOR_MAPPING.keys()
    for key in keys:
        if key not in PREDICTOR_MAPPING:
            raise ValueError(f"Predictor key '{key}' not found in PREDICTOR_MAPPING.")
        yield PREDICTOR_MAPPING[key]


def gene_pred_benchmark(
    dataset_keys: Optional[Sequence[str]] = None,
    n_obs_ceiling: Optional[int] = None,
    n_samples: int = 20,
    dir: str = "./data",
    predictor_keys: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    predicted_genes_count: int = 10,
    include_full: bool = True,
    include_celltype_subsets: bool = True,
    artifact_dir: Optional[str] = None,
):
    """
    Run the gene prediction benchmark across datasets, predictors, and seeds.

    This benchmark withholds a random subset of shared genes for each dataset pair,
    reconstructs expression using each predictor, and computes per-cell metrics.
    Results are recorded via the exp_runner runner and h5ad artifacts are written
    for the query and reconstruction.

    Args:
        dataset_keys: Dataset keys to pull from `DATASET_MAPPING`. Defaults to a
            standard set of mouse/human spatial and pseudospatial datasets.
        n_obs_ceiling: Optional upper bound for subsampling cells per dataset.
        n_samples: Number of seeds to generate when `seeds` is not provided.
        dir: Directory containing input data files.
        predictor_keys: Predictor keys to pull from `PREDICTOR_MAPPING`. Defaults
            to all predictors.
        seeds: Explicit list of seeds to use (overrides `n_samples`).
        predicted_genes_count: Number of shared genes to withhold for evaluation.
        include_full: Include full (all-cell) dataset pairs.
        include_celltype_subsets: Include per-cell-type subsets for common cell types.
        artifact_dir: Directory for writing h5ad artifacts. Defaults to `./artifacts`
            under the current working directory.
    """
    dataset_keys = dataset_keys or [
        "mouse_small_intestine_spatial",
        "mouse_small_intestine_pseudospatial",
        "human_liver_spatial",
        "human_liver_pseudospatial",
    ]

    seed_values = list(seeds) if seeds is not None else list(range(0, n_samples))
    seed_generator = (
        exp_runner.Variable(value, {"seed_id": i, "seed_value": value})
        for i, value in enumerate(seed_values)
    )
    inputs = _Input.generate_from(
        (
            _dataset_generator(
                dataset_keys,
                dir,
                n_obs_ceiling=n_obs_ceiling,
                include_full=include_full,
                include_celltype_subsets=include_celltype_subsets,
            ),
            _predictor_generator(predictor_keys),
            seed_generator,
            (exp_runner.Variable(predicted_genes_count, {}),),
            (exp_runner.Variable(artifact_dir, {}),),
        )
    )
    _gene_pred_benchmark(inputs)
