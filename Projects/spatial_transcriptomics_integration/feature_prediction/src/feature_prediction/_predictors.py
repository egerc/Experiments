from typing import Any, Callable, Generator

import numpy as np
import anndata as ad
from anndata.typing import AnnData
from exp_runner import Variable
from nico2_lib import predictors
import scanpy as sc

from feature_prediction import typing
from sklearn.model_selection import KFold


def _nmf_predictor(query: AnnData, reference: AnnData) -> AnnData:
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
    Wrap a gene-wise predictor with K-fold CV over shared genes.
    """

    def predictor(query: AnnData, reference: AnnData) -> AnnData:
        shared_genes = np.intersect1d(query.var_names, reference.var_names)
        cv = KFold(n_splits=n_splits)

        folds = [
            func(
                query[:, shared_genes][:, train_idx],
                reference[:, shared_genes],
            )
            for train_idx, _ in cv.split(shared_genes)
        ]

        return sc.concat(
            folds,
            axis="var",
            join="inner",
        )

    return predictor


def predictor_generator() -> Generator[Variable[typing.Predictor], Any, None]:
    predictors = [
        Variable(
            genewise_cv_predictor(_nmf_predictor),
            {"name": "nmf_predictor", "n_components": 3},
        )
    ]
    for predictor in predictors:
        yield predictor
