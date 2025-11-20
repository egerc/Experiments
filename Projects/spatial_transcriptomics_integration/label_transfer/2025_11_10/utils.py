from dataclasses import dataclass
from typing import List, Tuple
from anndata.typing import AnnData
from numpy.typing import NDArray
from numpy import floating, str_


@dataclass
class Dataset:
    query: AnnData
    reference: AnnData
    query_ct_key: str
    reference_ct_key: str

    def get_query_coordinates(self) -> List[Tuple[float, float]]:
        """
        Extract 2D spatial coordinates from an AnnData object.
        """
        if "spatial" not in self.query.obsm:
            raise ValueError("No spatial coordinates found in adata.obsm['spatial']")

        coords_array = self.query.obsm["spatial"]
        return [(float(x), float(y)) for x, y in coords_array]

    def get_query_barcodes(self) -> List[str]:
        """
        Extract barcodes (spot IDs) from an AnnData object.
        """
        return list(self.query.obs_names)