from pathlib import Path
import shutil
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
from NiCo import Annotations as ann

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
) -> dataset_labels:
    nico_containment_path = Path("./data/nico_artifacts")
    nico_containment_path.mkdir(exist_ok=True)
    Original_counts = reference.copy()
    Original_counts.raw = Original_counts.copy()
    sc.pp.normalize_total(Original_counts)
    sc.pp.log1p(Original_counts)
    sc.tl.pca(Original_counts)
    sc.pp.neighbors(Original_counts)
    index_sp, index_sc = ann.find_index(
        query.var_names.to_numpy(), reference.var_names.to_numpy()
    )

    ad_seq_common = reference[:, index_sc].copy()
    # ad_seq_common = reference.copy()
    ad_spatial_common = query[:, index_sp].copy()
    ad_seq_common.raw = ad_seq_common.copy()
    ad_spatial_common.raw = ad_spatial_common.copy()
    sc.experimental.pp.normalize_pearson_residuals(ad_seq_common, inplace=True)
    ad_seq_common.X[ad_seq_common.X < 0] = 0
    ad_seq_common.X = np.nan_to_num(ad_seq_common.X)
    sc.experimental.pp.normalize_pearson_residuals(ad_spatial_common, inplace=True)
    ad_spatial_common.X[ad_spatial_common.X < 0] = 0
    ad_spatial_common.X = np.nan_to_num(ad_spatial_common.X)

    ref_path = nico_containment_path / "inputRef"
    query_path = nico_containment_path / "inputQuery"
    ref_path.mkdir(exist_ok=True)
    query_path.mkdir(exist_ok=True)
    sc_sct_anndata_filename = ref_path / "sct_singleCell.h5ad"
    spatial_sct_anndata_filename = query_path / "sct_spatial.h5ad"
    sct_full_anndata_filename = ref_path / "Original_counts.h5ad"
    output_nico_dir = nico_containment_path / "nico_out"
    output_nico_dir.mkdir(exist_ok=True)
    ad_seq_common.write_h5ad(sc_sct_anndata_filename)
    Original_counts.write_h5ad(sct_full_anndata_filename)
    sc.pp.pca(ad_spatial_common)
    sc.pp.neighbors(ad_spatial_common, n_pcs=30)
    sc.tl.leiden(ad_spatial_common)
    ad_spatial_common.write_h5ad(spatial_sct_anndata_filename)
    anchors_and_neighbors_info = ann.find_anchor_cells_between_ref_and_query(
        refpath=f"{str(ref_path)}/",
        quepath=f"{str(query_path)}/",
        output_nico_dir=f"{str(output_nico_dir)}/",
    )

    output_info = ann.nico_based_annotation(
        anchors_and_neighbors_info,
        guiding_spatial_cluster_resolution_tag="leiden",
        ref_cluster_tag=reference_ct_key,
    )
    shutil.rmtree(nico_containment_path)
    annotation = output_info.nico_cluster
    return annotation


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
        adata_map=adata_map, adata_sp=query, annotation=reference_ct_key
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
    unlabeled_category = "Unknown"
    batch_label = "batch"
    query.obs["celltype"] = unlabeled_category
    reference.obs["celltype"] = reference.obs[reference_ct_key]
    adata = sc.concat([query, reference], label=batch_label)
    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_label)
    vae = scvi.model.SCVI(adata)
    vae.train(max_epochs=max_epochs)
    scanvi_prediction_key = "C_scanvi"
    scanvi = scvi.model.SCANVI.from_scvi_model(
        vae, adata=adata, labels_key="celltype", unlabeled_category=unlabeled_category
    )
    scanvi.train()
    adata.obs[scanvi_prediction_key] = scanvi.predict(adata)
    query.obs = query.obs.join(adata.obs[scanvi_prediction_key], how="left")
    predicted_labels = list(query.obs[scanvi_prediction_key])
    return predicted_labels


def method_generator() -> Generator[Variable[method_type], Any, None]:
    methods = [
        Variable(nmf_transfer, {"method_name": "nmf"}),
        # Variable(nmf_transfer_old, {"method_name": "nmf_old"}),
        Variable(scvi_transfer, {"method_name": "scvi"}),
        Variable(nico_transfer, {"method_name": "nico_old"}),
        # Variable(tangram_transfer, {"method_name": "tangram"}),
    ]
    for method in methods:
        yield method
