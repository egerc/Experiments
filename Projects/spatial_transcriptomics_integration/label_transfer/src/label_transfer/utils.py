from dataclasses import dataclass
from typing import List, Tuple
from anndata.typing import AnnData


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
    query: AnnData
    reference: AnnData
    query_ct_key: str
    reference_ct_key: str

    @property
    def query_coordinates(self) -> List[Tuple[float, float]]:
        return get_coordinates(self.query)

    @property
    def query_barcodes(self) -> List[str]:
        return get_barcodes(self.query)
    
    @property
    def query_lables(self) -> List[str]:
        return get_labels(self.query, self.query_ct_key)
