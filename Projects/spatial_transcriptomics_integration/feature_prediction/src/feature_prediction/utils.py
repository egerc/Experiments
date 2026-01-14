from typing import Generator, Sequence

import numpy as np
from anndata.typing import AnnData  # type:ignore
from numpy import number
from numpy.typing import NDArray


def adata_dense_mut(adata: AnnData) -> None:
    if hasattr(adata.X, "toarray"):  # type: ignore
        adata.X = adata.X.toarray()  # type: ignore


def iterate_shared_cells(
    adata: AnnData, adata_recon: AnnData
) -> Generator[Sequence[NDArray[number]]]:
    shared_cells = np.intersect1d(adata.obs_names, adata_recon.obs_names)
    adata_dense_mut(adata)
    adata_dense_mut(adata_recon)
    for cell, cell_recon in zip(
        adata[shared_cells].X,
        adata_recon[shared_cells].X,
    ):
        yield (cell, cell_recon)
