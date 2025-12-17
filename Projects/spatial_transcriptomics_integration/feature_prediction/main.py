from dataclasses import dataclass
from typing import Callable, List, Tuple
import itertools
from numpy.typing import NDArray
from numpy import number
from anndata.typing import AnnData
from exp_runner import VarProduct, runner_pickled


from feature_prediction import (
    typing,
    dataset_generator,
    predictor_generator,
    strategy_generator,
)


@dataclass
class Input(VarProduct):
    dataset: typing.Dataset
    predictor: typing.Predictor
    strategy: typing.Strategy


@runner_pickled(output_dir="./output", artifacts_subdir="../artifacts")
def experiment(input: Input) -> AnnData:
    query, reference = input.dataset()
    query_recon = input.strategy(query, reference, input.predictor)
    return query_recon


def main():
    inputs = Input.generate_from(
        (dataset_generator("../data"), predictor_generator(), strategy_generator())
    )
    experiment(inputs)
