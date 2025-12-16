from dataclasses import dataclass
from typing import Callable, List, Tuple
import itertools
from numpy.typing import NDArray
from numpy import number
from anndata.typing import AnnData
from exp_runner import VarProduct, runner_pickled, MetaData
from nico2_lib.metrics import (
    pearson_metric,
    spearman_metric,
    cosine_similarity_metric,
    explained_variance_metric,
    explained_variance_metric_v2,
    mse_metric,
)

from feature_prediction import typing, dataset_generator, predictor_generator


@dataclass
class Input(VarProduct):
    dataset: typing.Dataset
    predictor: typing.Predictor
    strategy: typing.Strategy


def _to_macro(metric) -> Callable: ...


def _to_balanced(metric) -> Callable: ...


@runner_pickled()
def experiment(input: Input) -> Tuple[AnnData, NDArray[number]]:
    query, reference = input.dataset()
    query_recon = input.strategy(query, reference, input.predictor)
    return query, query_recon


def main():
    inputs = Input.generate_from((dataset_generator("./data"), predictor_generator()))
    experiment(inputs)
