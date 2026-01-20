from anndata.typing import AnnData


def adata_dense_mut(adata: AnnData) -> None:
    if hasattr(adata.X, "toarray"):  # type: ignore
        adata.X = adata.X.toarray()  # type: ignore
