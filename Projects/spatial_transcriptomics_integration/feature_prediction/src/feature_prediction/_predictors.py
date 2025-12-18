"""
_predictors.py

Defines predictor functions for single-cell feature prediction.

Includes implementations of predictors, wrappers for cross-validation,
and a generator to iterate over available predictors.
"""

from functools import partial, wraps
from typing import Any, Callable, Dict, Generator, Type, Union

import numpy as np
import anndata as ad  # type: ignore
from anndata.typing import AnnData  # type: ignore
from exp_runner import Variable  # type: ignore
from nico2_lib import predictors  # type: ignore
import scanpy as sc  # type: ignore

from feature_prediction import typing
from feature_prediction.utils import adata_dense_mut
from sklearn.model_selection import KFold


def make_shared_gene_predictor(
    PredictorCls: Type[predictors.PredictorProtocol],
    predictor_kwargs: Dict[str, Any] | None = None,
) -> Callable[[AnnData, AnnData], AnnData]:
    """
    Factory for predictors that:
      - train on reference shared genes
      - predict reference-only genes in query

    Assumes PredictorCls has:
        .fit(X_shared_ref, X_ref_only)
        .predict(X_shared_query)
    """
    predictor_kwargs = predictor_kwargs or {}

    def predictor(query: AnnData, reference: AnnData) -> AnnData:
        shared_genes = np.intersect1d(query.var_names, reference.var_names)
        ref_only_genes = np.setdiff1d(reference.var_names, shared_genes)

        model = PredictorCls(**predictor_kwargs)

        X_ref_shared = reference[:, shared_genes].X
        X_ref_only = reference[:, ref_only_genes].X
        X_query_shared = query[:, shared_genes].X

        X_recon = model.fit(X_ref_shared, X_ref_only).predict(X_query_shared)

        return ad.AnnData(
            X=X_recon,
            obs=query.obs.copy(),
            var=reference[:, ref_only_genes].var.copy(),
        )

    return predictor


def _nmf_predictor(query: AnnData, reference: AnnData) -> AnnData:
    """
    Basic NMF-based predictor for reconstructing query cells from reference data.

    Parameters
    ----------
    query : AnnData
        Query AnnData with cells to reconstruct.
    reference : AnnData
        Reference AnnData used to predict missing genes.

    Returns
    -------
    AnnData
        Reconstructed query AnnData with predicted gene expression.
    """
    shared_genes = np.intersect1d(query.var_names, reference.var_names)
    sc_only_genes = np.setdiff1d(reference.var_names, shared_genes)
    recon_counts = (
        predictors.NmfPredictor(n_components=3)
        .fit(reference[:, shared_genes].X, reference[:, sc_only_genes].X)
        .predict(query[:, shared_genes].X)
    )
    return ad.AnnData(
        X=recon_counts, obs=query.obs, var=reference[:, sc_only_genes].var
    )


def genewise_cv_predictor(
    func: Callable[[AnnData, AnnData], AnnData],
    n_splits: int = 5,
) -> Callable[[AnnData, AnnData], AnnData]:
    """
    Wrap a gene-wise predictor with K-fold cross-validation over shared genes.

    Parameters
    ----------
    func : Callable[[AnnData, AnnData], AnnData]
        Predictor function that takes query and reference AnnData.
    n_splits : int
        Number of folds for K-fold cross-validation.

    Returns
    -------
    Callable[[AnnData, AnnData], AnnData]
        Predictor function with cross-validation applied.
    """

    @wraps(func)
    def predictor(query: AnnData, reference: AnnData) -> AnnData:
        adata_dense_mut(query)
        adata_dense_mut(reference)

        shared_genes = np.intersect1d(query.var_names, reference.var_names)  # type: ignore
        cv = KFold(n_splits=n_splits)

        folds = [
            func(
                query[:, shared_genes][:, train_idx],
                reference[:, shared_genes],
            )
            for train_idx, _ in cv.split(shared_genes)
        ]

        return sc.concat(  # type: ignore
            folds,
            axis="var",
            join="inner",
        )

    return predictor


tangram_predictor = genewise_cv_predictor(
    make_shared_gene_predictor(predictors.TangramPredictor)
)


def predictor_generator() -> Generator[Variable[typing.Predictor], Any, None]:
    """
    Generate available predictors wrapped in Variable objects for experiments.

    Yields
    ------
    Variable[typing.Predictor]
        Predictor functions with metadata.
    """
    nmf_predictor = genewise_cv_predictor(
        make_shared_gene_predictor(predictors.NmfPredictor, {"n_components": 3})
    )
    tangram_predictor = genewise_cv_predictor(
        make_shared_gene_predictor(predictors.TangramPredictor)
    )
    lvae_predictor = genewise_cv_predictor(
        make_shared_gene_predictor(
            partial(
                predictors.VaePredictor,
                vae_cls=predictors.models.LVAE,
                vae_kwargs={"latent_features": 64, "lr": 1e-3},
                devices=1,
            )
        )
    )
    ldvae_predictor = genewise_cv_predictor(
        make_shared_gene_predictor(
            partial(
                predictors.VaePredictor,
                vae_cls=predictors.models.LDVAE,
                vae_kwargs={
                    "hidden_features_in": 1024,
                    "latent_features": 64,
                    "lr": 1e-3,
                },
                devices=1,
            )
        )
    )
    levae_predictor = genewise_cv_predictor(
        make_shared_gene_predictor(
            partial(
                predictors.VaePredictor,
                vae_cls=predictors.models.LEVAE,
                vae_kwargs={
                    "hidden_features_out": 1024,
                    "latent_features": 64,
                    "lr": 1e-3,
                },
                devices=1,
            )
        )
    )
    vae_predictor = genewise_cv_predictor(
        make_shared_gene_predictor(
            partial(
                predictors.VaePredictor,
                vae_cls=predictors.models.VAE,
                vae_kwargs={
                    "hidden_features_in": 1024,
                    "hidden_features_out": 1024,
                    "latent_features": 64,
                    "lr": 1e-3,
                },
                devices=1,
            )
        )
    )
    predictors_list = [
        Variable(
            nmf_predictor,
            {"predictor_name": "nmf_predictor", "n_components": 3},
        ),
        Variable(
            tangram_predictor,
            {"predictor_name": "tangram_predictor"},
        ),
        Variable(lvae_predictor, {"predictor_name": "LVAE"}),
        Variable(ldvae_predictor, {"predictor_name": "LDVAE"}),
        Variable(levae_predictor, {"predictor_name": "LEVAE"}),
        Variable(vae_predictor, {"predictor_name": "VAE"}),
    ]
    for predictor in predictors_list:
        yield predictor
