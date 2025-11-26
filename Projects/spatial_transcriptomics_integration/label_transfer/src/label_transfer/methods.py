from typing import Any, Generator, List

from exp_runner import Variable
from anndata.typing import AnnData
import numpy as np
import scvi
from sklearn.decomposition import NMF, non_negative_factorization
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import scanpy as sc
import tangram as tg

from label_transfer.typing import dataset_labels, method_type
from label_transfer.datasets import get_labels


def nmf_transfer(
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

    scvi.model.SCVI.setup_anndata(reference, labels_key=reference_ct_key)
    vae = scvi.model.SCVI(reference)
    vae.train(max_epochs=20)
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
        Variable(scvi_transfer, {"method_name": "scvi"}),
        Variable(tangram_transfer, {"method_name": "tangram"}),
    ]
    for method in methods:
        yield method
