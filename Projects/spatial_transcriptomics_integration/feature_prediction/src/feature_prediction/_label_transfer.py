from typing import Callable, List
from anndata.typing import AnnData

import numpy as np
from numpy import number
from numpy.typing import NDArray
from sklearn.decomposition import NMF


def cosine_similarity_row(A: NDArray[number], B: NDArray[number]) -> NDArray[number]:
    """Compute cosine similarity between rows of A and corresponding rows of B."""
    dot_products = np.sum(A * B, axis=1)
    norms_A = np.linalg.norm(A, axis=1)
    norms_B = np.linalg.norm(B, axis=1)
    denom = norms_A * norms_B
    return np.divide(
        dot_products,
        denom,
        out=np.zeros_like(dot_products, dtype=float),
        where=denom != 0,
    )


def _fit_nmf_and_score(
    query: NDArray[number],
    reference: NDArray[number],
    nmf: NMF,
    score_func: Callable[
        [NDArray[number], NDArray[number]], NDArray[number]
    ] = cosine_similarity_row,
) -> NDArray[number]:
    """Fit NMF on reference, reconstruct query, return cosine scores."""
    nmf.fit(reference)
    H = nmf.components_
    W = nmf.transform(query.astype(H.dtype))
    reconstruction = W @ H
    return score_func(query, reconstruction)


def _nmf_label_transfer(
    query: NDArray[number],
    reference: NDArray[number],
    labels: NDArray[np.str_],
    n_components: int = 3,
    max_iter: int = 5000,
) -> NDArray[np.str_]:
    """
    Assign labels to query cells by maximizing NMF-based cosine similarity.

    Parameters
    ----------
    query : np.ndarray
        Query cell expression matrix (cells x genes).
    reference : np.ndarray
        Reference cell expression matrix (cells x genes).
    labels : np.ndarray
        Cell-type labels for reference cells.
    n_components : int
        Number of NMF components.
    max_iter : int
        Maximum iterations for NMF.
    """
    celltypes = np.unique(labels)

    similarities: List[NDArray[number]] = []
    for ct in celltypes:
        reference_mask = labels == ct
        scores = _fit_nmf_and_score(
            query,
            reference[reference_mask],
            NMF(n_components=n_components, max_iter=max_iter),
        )
        similarities.append(scores)

    similarities = np.vstack(similarities)
    return celltypes[np.argmax(similarities, axis=0)]


def nmf_transfer(
    query: AnnData,
    reference: AnnData,
    reference_ct_key: str,
    n_components: int = 3,
    max_iter: int = 5000,
) -> NDArray[np.str_]:
    """
    Transfer cell-type labels from reference AnnData to query AnnData using NMF.

    Parameters
    ----------
    query : AnnData
        Query dataset.
    reference : AnnData
        Reference dataset.
    reference_ct_key : str
        Key in reference.obs containing cell-type labels.
    query_ct_key : str, optional
        If provided, will also add annotations to query.obs under this key.
    n_components : int
        Number of NMF components.
    max_iter : int
        Maximum iterations for NMF.
    """
    feature_names = np.intersect1d(query.var_names, reference.var_names)

    query_counts = query[:, feature_names].X
    reference_counts = reference[:, feature_names].X
    if not isinstance(query_counts, np.ndarray):
        query_counts = query_counts.toarray()
    if not isinstance(reference_counts, np.ndarray):
        reference_counts = reference_counts.toarray()

    reference_labels = np.array(reference.obs[reference_ct_key].values, dtype=str)

    query_labels = _nmf_label_transfer(
        query_counts,
        reference_counts,
        reference_labels,
        n_components=n_components,
        max_iter=max_iter,
    )

    return query_labels
