from dataclasses import dataclass
from typing import Callable, List
from exp_runner import MetaData, VarProduct, runner

from label_transfer import Dataset, method_type, method_generator, dataset_generator



@dataclass
class Input(VarProduct):
    method: method_type
    dataset: Callable[[], Dataset]


@runner(head=5)
def experiment(input: Input) -> List[MetaData]:
    dataset = input.dataset()
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
    variables = (method_generator(), dataset_generator())
    inputs = Input.generate_from(variables)
    experiment(inputs)


if __name__ == "__main__":
    main()
