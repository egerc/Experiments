from dataclasses import dataclass
from typing import Any, Callable, Generator, List, Optional, Tuple, Union

from anndata import AnnData
from exp_runner import Variable
from nico2_lib.datasets import xenium_10x_loader, human_liver_cell_atlas


def get_coordinates(adata: AnnData) -> List[Tuple[float, float]]:
    """
    Extract 2D spatial coordinates from an AnnData object.
    """
    if "spatial" not in adata.obsm:
        raise ValueError("No spatial coordinates found in adata.obsm['spatial']")

    coords_array = adata.obsm["spatial"]
    return [(float(x), float(y)) for x, y in coords_array]


def get_barcodes(adata: AnnData) -> List[str]:
    """
    Extract barcodes (spot IDs) from an AnnData object.
    """
    return list(adata.obs_names)


def get_labels(adata: AnnData, key: str) -> List[str]:
    values = adata.obs[key]
    try:
        return list(values.astype(str))
    except Exception as e:
        raise ValueError(
            f"Could not convert values in adata.obs['{key}'] to strings."
        ) from e


@dataclass
class Dataset:
    reference: AnnData
    reference_ct_key: str
    query: AnnData
    query_ct_key: Optional[str] = None

    @property
    def query_coordinates(self) -> List[Tuple[float, float]]:
        return get_coordinates(self.query)

    @property
    def query_barcodes(self) -> List[str]:
        return get_barcodes(self.query)

    @property
    def query_lables(self) -> Union[List[str], List[None]]:
        if self.query_ct_key:
            return get_labels(self.query, self.query_ct_key)
        else:
            labels = [None for cell in self.query.obs_names]
            return labels


def dataset_generator(
    dir: str,
) -> Generator[Variable[Callable[[], Dataset]], Any, None]:
    def human_liver_10x() -> Dataset:
        query = xenium_10x_loader("Xenium_V1_hLiver_nondiseased_section_FFPE", dir=dir)
        reference: AnnData = human_liver_cell_atlas(dir=dir)
        reference_ct_key: str = "annot"
        return Dataset(reference, reference_ct_key, query)

    variables = [
        Variable(
            human_liver_10x,
            {
                "query_repository": "10x",
                "query_id": "Xenium_V1_hLiver_nondiseased_section_FFPE",
                "reference_repository": "livercellatlas",
                "reference_id": "Liver Cell Atlas: Human, All Liver Cells",
            },
        )
    ]
    for variable in variables:
        yield variable
