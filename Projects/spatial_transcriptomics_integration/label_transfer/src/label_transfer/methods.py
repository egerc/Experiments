from typing import Any, Generator, List

from exp_runner import Variable
from NiCo import Annotations
from anndata.typing import AnnData
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error

from .typing import dataset_labels, method_type
from .utils import get_labels


def nmf_transfer(query: AnnData, reference: AnnData, reference_ct_key: str) -> dataset_labels:
    reference_labels = get_labels(reference, reference_ct_key)
    unique_labels = sorted(list(set(reference_labels)))

    n_query = query.shape[0]
    n_labels = len(unique_labels)
    mse_matrix = np.zeros((n_query, n_labels))
    for j, label in enumerate(unique_labels):
        label_idx = np.where(np.array(reference_labels) == label)[0]
        label_data = reference[label_idx]
        nmf = NMF(n_components=3)
        nmf.fit(label_data.X)
        query_embeddings = nmf.transform(query.X)
        query_pred = nmf.inverse_transform(query_embeddings)
        for i in range(n_query):
            mse_matrix[i, j] = mean_squared_error(
                query.X[i].A.ravel() if hasattr(query.X[i], "A") else query.X[i],
                query_pred[i]
            )
    best_label_idx: List[int] = list(np.argmin(mse_matrix, axis=1))
    predicted_labels = [unique_labels[k] for k in best_label_idx]
    return predicted_labels

def nico_transfer(query: AnnData, reference: AnnData, reference_ct_key: str) -> dataset_labels: ...
def tangram_transfer(query: AnnData, reference: AnnData) -> dataset_labels: ...
def scvi_transfer(query: AnnData, reference: AnnData) -> dataset_labels: ...

def method_generator() -> Generator[Variable[method_type], Any, None]:
    methods = [
        Variable(nmf_transfer, {"name": "nmf"})
    ]
    for method in methods:
        yield method