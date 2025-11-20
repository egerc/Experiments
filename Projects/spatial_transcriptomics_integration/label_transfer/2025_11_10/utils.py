from typing import Tuple
from anndata.typing import AnnData
from numpy.typing import NDArray
from numpy import floating


def get_coordinates(adata: AnnData) -> Tuple[NDArray[floating], NDArray[floating]]: ...