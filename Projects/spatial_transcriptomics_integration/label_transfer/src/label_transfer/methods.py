from typing import Any, Generator, List

from exp_runner import Variable
from anndata.typing import AnnData
import numpy as np
from sklearn.decomposition import NMF, non_negative_factorization
from sklearn.metrics import mean_squared_error

from label_transfer.typing import dataset_labels, method_type
from label_transfer.datasets import get_labels


def nmf_transfer(
    query: AnnData, reference: AnnData, reference_ct_key: str
) -> List[str]:
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
        X_label = reference.X[label_idx][:, ref_idx]

        _, H_ref, _ = non_negative_factorization(
            X_label, n_components=3, init="random", random_state=0
        )

        H_ref = H_ref.astype(X_query_shared.dtype)
        W_query, _, _ = non_negative_factorization(
            X_query_shared,
            H=H_ref,
            n_components=H_ref.shape[0],
            init="custom",
            update_H=False,
            random_state=0,
        )
        query_pred = W_query @ H_ref

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

    mse_matrix[i, j] = mean_squared_error(q_i, query_pred_i)
    best_label_idx = np.argmin(mse_matrix, axis=1)
    predicted_labels = [unique_labels[k] for k in best_label_idx]
    return predicted_labels


def nico_transfer(
    query: AnnData, reference: AnnData, reference_ct_key: str
) -> dataset_labels: ...
def tangram_transfer(query: AnnData, reference: AnnData) -> dataset_labels: ...
def scvi_transfer(query: AnnData, reference: AnnData) -> dataset_labels: ...


def method_generator() -> Generator[Variable[method_type], Any, None]:
    methods = [Variable(nmf_transfer, {"name": "nmf"})]
    for method in methods:
        yield method
