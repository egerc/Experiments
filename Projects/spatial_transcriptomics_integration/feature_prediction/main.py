from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple
import itertools
from numpy.typing import NDArray
from numpy import number
from anndata.typing import AnnData
from exp_runner import VarProduct, runner_pickled
from exp_runner.runner import get_timestamp


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


def h5ad_saver(adata: AnnData, directory: Path) -> str:
    """
    Default saver: write AnnData to a timestamped .h5ad file.
    Returns the file path as a string.
    """
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{get_timestamp()}.h5ad"
    path = directory / filename
    adata.write_h5ad(path)
    return str(path)

@runner_pickled(output_dir="./output", saver=h5ad_saver)
def experiment(input: Input) -> AnnData:
    print(input.dataset, input.predictor, input.strategy)
    query, reference, query_ct_key, reference_ct_key = input.dataset()
    query_recon = input.strategy(query, reference, input.predictor, query_ct_key, reference_ct_key)
    query_recon.obs = query.obs
    return query_recon


def main():
    inputs = Input.generate_from(
        (dataset_generator("../data"), predictor_generator(), strategy_generator())
    )
    experiment(inputs)

if __name__ == "__main__":
    main()
