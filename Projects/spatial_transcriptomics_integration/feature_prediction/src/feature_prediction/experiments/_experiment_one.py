from dataclasses import dataclass
from functools import partial
from typing import Callable, Generator, List, Optional, Sequence, Tuple

import exp_runner
import nico2_lib as n2l
import numpy as np
import scanpy as sc
from anndata.typing import AnnData

from feature_prediction import DATASET_MAPPING
from feature_prediction._predictors import make_shared_gene_predictor
from feature_prediction.utils import adata_dense_mut, iterate_shared_cells


@dataclass
class _Input(exp_runner.VarProduct):
    datasets: Tuple[AnnData, AnnData]
    predictor: Callable[[AnnData, AnnData], AnnData]
    seed: Optional[int] = None


@exp_runner.runner()
def _gene_pred_benchmark(input: _Input) -> List[exp_runner.MetaData]:
    print(input)
    rng = np.random.default_rng(input.seed)
    query, reference = input.datasets
    shared_genes = np.intersect1d(query.var_names, reference.var_names).tolist()
    predicted_genes = rng.choice(shared_genes, size=10, replace=False).tolist()
    training_genes = np.setdiff1d(shared_genes, predicted_genes)
    query_recon = input.predictor(query[:, training_genes], reference[:, shared_genes])
    return [
        {
            "mse": n2l.mt.mse_metric(cell, cell_recon),
            "pearsonr": n2l.mt.pearson_metric(cell, cell_recon),
            "spearmanr": n2l.mt.spearman_metric(cell, cell_recon),
            "cosine": n2l.mt.cosine_similarity_metric(cell, cell_recon),
            "exp_var_scikit": n2l.mt.explained_variance_metric(cell, cell_recon),
            "exp_var_dominic": n2l.mt.explained_variance_metric_v2(cell, cell_recon),
        }
        for cell, cell_recon in iterate_shared_cells(
            query[:, predicted_genes], query_recon[:, predicted_genes]
        )
    ]


def _dataset_generator(
    dataset_keys: Sequence[str], dir: str, n_obs_ceiling: Optional[int] = None
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
        query, reference, query_ct_key, reference_ct_key = loader_variable.value(dir)
        if n_obs_ceiling is not None:
            n_obs_query = min(query.n_obs, n_obs_ceiling)
            n_obs_ref = min(reference.n_obs, n_obs_ceiling)
            sc.pp.subsample(query, n_obs=n_obs_query)
            sc.pp.subsample(reference, n_obs=n_obs_ref)
        adata_dense_mut(query)
        adata_dense_mut(reference)

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

        for ct in common_cell_types:
            query_subset = query[query.obs[query_ct_key] == ct].copy()
            reference_subset = reference[reference.obs[reference_ct_key] == ct].copy()
            yield exp_runner.Variable(
                (query_subset, reference_subset),
                metadata={
                    **loader_variable.metadata,
                    **{"celltype": ct, "reconstruction_strategy": "per_celltype"},
                },
            )


def _predictor_generator():
    predictors = [
        exp_runner.Variable(
            make_shared_gene_predictor(n2l.pd.NmfPredictor, {"n_components": 3}),
            {
                "predictor_id": 0,
                "predictor_name": "nmf_3",
                "architecture": "NMF",
                "n_components": 3,
            },
        ),
        exp_runner.Variable(
            make_shared_gene_predictor(n2l.pd.NmfPredictor, {"n_components": 8}),
            {
                "predictor_id": 1,
                "predictor_name": "nmf_8",
                "architecture": "NMF",
                "n_components": 8,
            },
        ),
        exp_runner.Variable(
            make_shared_gene_predictor(n2l.pd.NmfPredictor, {"n_components": "auto"}),
            {
                "predictor_id": 2,
                "predictor_name": "nmf_auto",
                "architecture": "NMF",
                "n_components": 8,
            },
        ),
        exp_runner.Variable(
            make_shared_gene_predictor(n2l.pd.TangramPredictor),
            {"predictor_id": 3, "predictor_name": "tangram", "architecture": "Tangram"},
        ),
        exp_runner.Variable(
            make_shared_gene_predictor(
                partial(
                    n2l.pd.VaePredictor,
                    vae_cls=n2l.pd.models.VAE,
                    vae_kwargs={
                        "hidden_features_in": 128,
                        "hidden_features_out": 128,
                        "latent_features": 3,
                        "lr": 1e-3,
                    },
                    devices=1,
                )
            ),
            {
                "predictor_id": 4,
                "predictor_name": "VAE_3",
                "architecture": "VAE",
                "hidden_features_in": 128,
                "hidden_features_out": 128,
                "latent_features": 3,
            },
        ),
    ]
    for predictor in predictors:
        yield predictor


def gene_pred_benchmark(
    dataset_keys: Optional[Sequence[str]] = None,
    n_obs_ceiling: Optional[int] = None,
    n_samples: int = 20,
    dir: str = "./data",
):
    dataset_keys = dataset_keys or [
        "mouse_small_intestine_spatial",
        "mouse_small_intestine_pseudospatial",
        # "human_liver_spatial",
        # "human_liver_pseudospatial",
    ]

    seed_generator = (
        exp_runner.Variable(value, {"seed_id": i, "seed_value": value})
        for i, value in enumerate(range(0, n_samples))
    )
    inputs = _Input.generate_from(
        (
            _dataset_generator(dataset_keys, dir, n_obs_ceiling),
            _predictor_generator(),
            seed_generator,
        )
    )
    _gene_pred_benchmark(inputs)
