from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Optional, Sequence

import scanpy as sc
from exp_runner import MetaData, VarProduct, runner
from spatial_transcriptomics_integration.datasets import (
    Dataset,
    label_transfer_dataset_generator,
)
from spatial_transcriptomics_integration.label_transfer import (
    method_generator,
    method_type,
)


@dataclass
class Input(VarProduct):
    method: method_type
    dataset: Callable[[], Dataset]


@runner(format="parquet")
def experiment(input: Input) -> List[MetaData]:
    dataset = input.dataset()
    n_obs = min(dataset.query.n_obs, dataset.reference.n_obs, 5000)
    subsample = partial(sc.pp.subsample, n_obs=n_obs)
    subsample(dataset.query)
    subsample(dataset.reference)
    labels = input.method(dataset.query, dataset.reference, dataset.reference_ct_key)
    return [
        {
            "label": label,
            "barcode": barcode,
            "x": x,
            "y": y,
            "ground_truth": ground_truth,
        }
        for label, barcode, (x, y), ground_truth in zip(
            labels,
            dataset.query_barcodes,
            dataset.query_coordinates,
            dataset.query_lables,
        )
    ]


def label_transfer_benchmark(
    dataset_keys: Optional[Sequence[str]] = None,
    dir: str = "./data",
) -> None:
    variables = (
        method_generator(),
        label_transfer_dataset_generator(dir, dataset_keys=dataset_keys),
    )
    inputs = Input.generate_from(variables)
    experiment(inputs)
