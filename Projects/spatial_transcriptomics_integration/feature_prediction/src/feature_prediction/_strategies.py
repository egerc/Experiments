from itertools import product
from typing import Any, Callable, Generator, List

from exp_runner import Variable
from anndata.typing import AnnData
import scanpy as sc
from numpy.typing import NDArray
from numpy import number

from feature_prediction import typing


def _at_once(adata: AnnData) -> List[AnnData]:
    return [adata]


def _stratified_folds(adata: AnnData) -> List[AnnData]: ...


def _per_celltype(adata: AnnData) -> List[AnnData]: ...


def strategy_generator() -> Generator[Variable[typing.Strategy], Any, None]:
    def wrapper(
        func: Callable[[AnnData], List[AnnData]],
    ) -> Callable[[AnnData, AnnData, typing.Predictor], AnnData]:
        def wrapped(
            query: AnnData, reference: AnnData, predictor: typing.Predictor
        ) -> AnnData:
            recon = sc.concat(
                [
                    predictor(query_subset, reference_subset)
                    for query_subset, reference_subset in product(
                        func(query), func(reference)
                    )
                ]
            )
            return recon

        return wrapped

    strategies = [
        Variable(wrapper(_at_once), {"reconstruction_strategy": "all at once"}),
    ]
    for strategy in strategies:
        yield strategy
