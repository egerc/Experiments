from dataclasses import dataclass
from pathlib import Path

import scanpy as sc
from anndata.typing import AnnData
from exp_runner import VarProduct, runner_pickled
from exp_runner.runner import get_timestamp
from numpy import number

from feature_prediction import (
    dataset_generator,
    predictor_generator,
    typing,
)
from feature_prediction.utils import adata_dense_mut


@dataclass
class Input(VarProduct):
    dataset: typing.Dataset
    predictor: typing.Predictor


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


@runner_pickled(output_dir="./output", saver=h5ad_saver, format="csv")
def experiment(input: Input) -> AnnData:
    print(input.dataset, input.predictor)
    query, reference, query_ct_key, reference_ct_key = input.dataset()
    n_obs_ceiling = 2500
    n_obs_query = min(query.n_obs, n_obs_ceiling)
    n_obs_ref = min(reference.n_obs, n_obs_ceiling)
    sc.pp.subsample(query, n_obs=n_obs_query)
    sc.pp.subsample(reference, n_obs=n_obs_ref)
    adata_dense_mut(query)
    adata_dense_mut(reference)
    query_recon = input.predictor(query, reference, query_ct_key, reference_ct_key)
    query_recon.obs = query_recon.obs.join(query.obs)
    print(query_recon)
    return query_recon


def main():
    inputs = Input.generate_from((dataset_generator("../data"), predictor_generator()))
    experiment(inputs)


if __name__ == "__main__":
    main()
