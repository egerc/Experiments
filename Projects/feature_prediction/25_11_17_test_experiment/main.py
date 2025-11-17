from dataclasses import dataclass
from itertools import product
from typing import Callable
from anndata import AnnData
from exp_runner import MetaData, Variable, runner
import scanpy as sc
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from spatial_transcript_inference.nmf_utils import nmf_predictor


@dataclass
class Dataset:
    adata: AnnData
    tissue: str


@dataclass
class Input:
    predictor: Callable
    dataset: Dataset


variables = (
    Variable(Input(predictor, dataset), metadata={"func": predictor.__name__, "tissue": dataset.tissue})
    for predictor, dataset in product(
        [nmf_predictor], [Dataset(sc.datasets.pbmc3k(), "human")]
    )
)


@runner()
def experiment(input: Input):
    X = input.dataset.adata.X
    X_pred = input.predictor(X)
    mse = mean_absolute_error(X, X_pred)
    return [{"mse": mse}]


def main():
    experiment(variables)


if __name__ == "__main__":
    main()
