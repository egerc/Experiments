from typing import Any, Callable, Generator, List

from exp_runner import Variable
from anndata.typing import AnnData
import numpy as np
from numpy import number
from numpy.typing import NDArray
import scvi
from sklearn.decomposition import NMF, non_negative_factorization
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import scanpy as sc
import tangram as tg

from label_transfer.typing import dataset_labels, method_type
from label_transfer.datasets import get_labels


def nmf_transfer_old(
    query: AnnData, reference: AnnData, reference_ct_key: str
) -> dataset_labels:
    max_iter = 200
    reference_labels = reference.obs[reference_ct_key].values
    unique_labels = sorted(list(set(reference_labels)))

    n_query = query.shape[0]
    n_labels = len(unique_labels)
    mse_matrix = np.zeros((n_query, n_labels))

    shared_genes = np.intersect1d(reference.var_names, query.var_names)
    ref_idx = [np.where(reference.var_names == g)[0][0] for g in shared_genes]
    query_idx = [np.where(query.var_names == g)[0][0] for g in shared_genes]
    X_query_shared = query.X[:, query_idx]

    for j, label in enumerate(unique_labels):
        label_idx = np.where(reference_labels == label)[0]
        X_label = reference.X[label_idx]  # [:, ref_idx]

        # Fit NMF on label-specific reference
        nmf_model = NMF(
            n_components=3, max_iter=max_iter, init="random", random_state=0
        )
        _ = nmf_model.fit_transform(X_label)
        H_ref = nmf_model.components_.astype(X_query_shared.dtype)

        # Project query with frozen H_ref
        W_query, H_query, _ = non_negative_factorization(
            X_query_shared,
            H=H_ref[:, ref_idx],
            init="custom",
            update_H=False,
            max_iter=max_iter,
        )
        query_pred = W_query @ H_query

        for i in range(n_query):
            q_i = X_query_shared[i]
            if hasattr(q_i, "toarray"):
                q_i = q_i.toarray().ravel()
            elif hasattr(q_i, "A"):
                q_i = q_i.A.ravel()
            else:
                q_i = np.asarray(q_i).ravel()

            query_pred_i = query_pred[i]
            if hasattr(query_pred_i, "toarray"):
                query_pred_i = query_pred_i.toarray().ravel()
            elif hasattr(query_pred_i, "A"):
                query_pred_i = query_pred_i.A.ravel()
            else:
                query_pred_i = np.asarray(query_pred_i).ravel()

            mse_matrix[i, j] = cosine_similarity(
                q_i.reshape(1, -1), query_pred_i.reshape(1, -1)
            )[0, 0]

    best_label_idx = np.argmin(mse_matrix, axis=1)
    predicted_labels = [unique_labels[k] for k in best_label_idx]
    return predicted_labels

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

    query_counts = query[:, feature_names].X.toarray()
    reference_counts = reference[:, feature_names].X.toarray()

    reference_labels = np.array(reference.obs[reference_ct_key].values, dtype=str)

    query_labels = _nmf_label_transfer(
        query_counts,
        reference_counts,
        reference_labels,
        n_components=n_components,
        max_iter=max_iter,
    )

    return query_labels

def nico_transfer(
    query: AnnData, reference: AnnData, reference_ct_key: str
) -> dataset_labels: ...
def tangram_transfer(
    query: AnnData, reference: AnnData, reference_ct_key: str
) -> dataset_labels:
    """
    Transfer labels from reference to query using Tangram.

    Tangram maps reference single-cell cell types onto the query
    (spatial) transcriptomics dataset. We assign each query cell the
    cell type with maximal mapped probability.

    Parameters
    ----------
    query : AnnData
        Query dataset (typically spatial).
    reference : AnnData
        Reference scRNA-seq dataset with cell-type labels.
    reference_ct_key : str
        Column in reference.obs containing the cell-type annotations.

    Returns
    -------
    dataset_labels
        List of predicted labels for each query cell.
    """

    tg.pp_adatas(reference, query)

    adata_map = tg.map_cells_to_space(
        adata_sc=reference,
        adata_sp=query,
        mode="cells",
    )

    tg.project_cell_annotations(
        adata_map=adata_map,
        adata_sp=query,
        annotation=reference_ct_key
    )

    predicted_labels = query.obsm["tangram_ct_pred"].idxmax(axis=1).values.tolist()

    return predicted_labels

def scvi_transfer(
    query: AnnData, reference: AnnData, reference_ct_key: str
) -> dataset_labels:
    """
    Predict query cell types using scANVI based on a labeled reference.

    Parameters
    ----------
    query : AnnData
        The query dataset (unlabeled).
    reference : AnnData
        The reference dataset with known cell-type labels.
    reference_ct_key : str
        Column in reference.obs containing cell-type labels.

    Returns
    -------
    dataset_labels
        List of predicted labels for each cell in query.
    """

    max_epochs = 200
    scvi.model.SCVI.setup_anndata(reference, labels_key=reference_ct_key)
    vae = scvi.model.SCVI(reference)
    vae.train(max_epochs=max_epochs)
    scanvi = scvi.model.SCANVI.from_scvi_model(
        vae, labels_key=reference_ct_key, unlabeled_category="Unknown"
    )
    scanvi.train()
    scvi.model.SCANVI.prepare_query_anndata(query, scanvi)
    query_model = scvi.model.SCANVI.load_query_data(query, scanvi)
    predicted_labels = query_model.predict(query)
    return predicted_labels.tolist()


def method_generator() -> Generator[Variable[method_type], Any, None]:
    methods = [
        Variable(nmf_transfer, {"method_name": "nmf"}),
        #Variable(nmf_transfer_old, {"method_name": "nmf_old"}),
        #Variable(scvi_transfer, {"method_name": "scvi"}),
        #Variable(tangram_transfer, {"method_name": "tangram"}),
    ]
    for method in methods:
        yield method
