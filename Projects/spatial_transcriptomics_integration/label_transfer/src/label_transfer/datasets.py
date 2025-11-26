from functools import partial
from typing import Any, Callable, Generator

from anndata import AnnData
from exp_runner import Variable
from label_transfer.utils import Dataset
from nico2_lib.datasets import xenium_10x_loader

dataset_pairings = [
    ()
]

def dataset_generator() -> Generator[Variable[Callable[[], Dataset]], Any, None]:
    def human_liver_10x() -> Dataset:
        query = xenium_10x_loader("Xenium_V1_hLiver_nondiseased_section_FFPE")
        query_ct_key: str = ...
        reference: AnnData = ...
        reference_ct_key: str = ...
        return Dataset(
            query, reference, query_ct_key, reference_ct_key
        )