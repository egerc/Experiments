
from itertools import product
from typing import Any, Callable, Generator, List

from exp_runner import Variable
from anndata.typing import AnnData
from numpy.typing import NDArray
from numpy import number

from feature_prediction import typing

def _at_once(adata: AnnData) -> List[AnnData]: ...
def _stratified_folds(adata: AnnData) -> List[AnnData]: ...
def _per_celltype(adata: AnnData) -> List[AnnData]: ...

def strategy_generator() -> Generator[Variable[typing.Strategy], Any, None]:
    def _wrapper(func: Callable[[AnnData], List[AnnData]]) -> Callable[[AnnData, AnnData, typing.Predictor], NDArray[number]]:
        def wrapped(query: AnnData, reference: AnnData, predictor: typing.Predictor) -> NDArray[number[Any, int | float | complex]]:
            for query_subset, reference_subset in product(func(query), func(reference)):
                recon = predictor(query_subset.X, reference.subset.X)
            return recon
        return wrapped
    
    strategies = [
        Variable(_wrapper(_at_once), {"reconstruction_strategy": "all at once"}),
        Variable(_wrapper(_stratified_folds), {"reconstruction_strategy": "along stratified folds"}),
        Variable(_wrapper(_per_celltype), {"reconstruction_strategy": "per celltype"}),
    ]
    for strategy in strategies:
        yield strategy

