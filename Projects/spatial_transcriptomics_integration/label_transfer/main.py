from dataclasses import dataclass
from functools import partial
from typing import Callable, List
from joblib import Memory
from exp_runner import MetaData, VarProduct, runner

import scanpy as sc

from label_transfer import Dataset, method_type, method_generator, dataset_generator

memory = Memory("./cache", verbose=0)


@dataclass
class Input(VarProduct):
    method: method_type
    dataset: Callable[[], Dataset]


@runner(format="parquet")
def experiment(input: Input) -> List[MetaData]:
    print("test")
    dataset = input.dataset()
    subsample = partial(sc.pp.subsample, n_obs=500)
    #subsample(dataset.query)
    #subsample(dataset.reference)
    print(input.method.__name__)
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


def main():
    variables = (method_generator(), dataset_generator("./data"))
    inputs = Input.generate_from(variables)
    experiment(inputs)


if __name__ == "__main__":
    main()
