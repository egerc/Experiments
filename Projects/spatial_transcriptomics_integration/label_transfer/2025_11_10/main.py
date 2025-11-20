from .typing import method_type, dataset_type
from .utils import get_coordinates
from .methods import nico_transfer, nmf_transfer
from dataclasses import dataclass
from typing import Any, Callable, Generator, List, Tuple
from anndata import AnnData
from exp_runner import MetaData, VarProduct, Variable, runner


@dataclass
class Input(VarProduct):
    method: method_type
    dataset: dataset_type


def dataset_generator() -> Generator[Variable[dataset_type], Any, None]: ...


def method_generator() -> Generator[Variable[method_type], Any, None]:
    methods = [Variable(nmf_transfer, {}), Variable(nico_transfer, {})]
    for method in methods:
        yield method


@runner()
def experiment(input: Input) -> List[MetaData]:
    query, reference = input.dataset
    labels = input.method(query, reference)
    barcodes = list(query.obs.index)
    x_coords, y_coords = get_coordinates(query)
    return [
        {"label": label, "barcode": barcode, "x": x, "y": y}
        for label, barcode, x, y in zip(labels, barcodes, x_coords, y_coords)
    ]


def main():
    variables = (dataset_generator(), method_generator())
    inputs = Input.generate_from(variables)
    experiment(inputs)


if __name__ == "__main__":
    main()
