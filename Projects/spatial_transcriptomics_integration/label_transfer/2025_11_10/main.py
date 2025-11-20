from .typing import method_type, dataset_type
from .utils import Dataset
from .methods import nico_transfer, nmf_transfer
from dataclasses import dataclass
from typing import Any, Generator, List
from exp_runner import MetaData, VarProduct, Variable, runner


@dataclass
class Input(VarProduct):
    method: method_type
    dataset: Dataset


def dataset_generator() -> Generator[Variable[dataset_type], Any, None]: ...


def method_generator() -> Generator[Variable[method_type], Any, None]:
    funcs = [nmf_transfer, nico_transfer]
    methods = [Variable(func, {"name": func.__name__}) for func in funcs]
    for method in methods:
        yield method


@runner()
def experiment(input: Input) -> List[MetaData]:
    labels = input.method(input.dataset.query, input.dataset.reference, input.dataset.reference_ct_key)
    barcodes = input.dataset.get_query_barcodes()
    coords = input.dataset.get_query_coordinates()
    return [
        {"label": label, "barcode": barcode, "x": x, "y": y}
        for label, barcode, (x, y) in zip(labels, barcodes, coords)
    ]


def main():
    variables = (method_generator(), dataset_generator())
    inputs = Input.generate_from(variables)
    experiment(inputs)


if __name__ == "__main__":
    main()
