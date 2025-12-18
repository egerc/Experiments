#!/usr/bin/env python3

from curses.ascii import isspace
from typing import Any, Dict, List, Optional
from itertools import chain
import argparse
from pathlib import Path

import polars as pl
import scanpy as sc
import numpy as np

from nico2_lib.metrics import (
    mse_metric,
    explained_variance_metric,
    explained_variance_metric_v2,
    pearson_metric,
    spearman_metric,
    cosine_similarity_metric,
)
from feature_prediction import utils
from scipy import sparse


def compute_cellwise_metrics(
    sc_data_path: str,
    artifact_path: str,
    spatial_data_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    original_adata_path = (
        f"../data/{spatial_data_path}/{spatial_data_path}.h5ad"
        if spatial_data_path
        else f"../data/{sc_data_path}/{sc_data_path}.h5ad"
    )

    adata = sc.read_h5ad(original_adata_path)
    adata_recon = sc.read_h5ad(artifact_path)

    shared_barcodes = np.intersect1d(adata.obs_names, adata_recon.obs_names)
    shared_genes = np.intersect1d(adata.var_names, adata_recon.var_names)

    adata = adata[shared_barcodes, shared_genes].copy()
    adata_recon = adata_recon[shared_barcodes, shared_genes].copy()

    if sparse.issparse(adata.X):
        adata.X = adata.X.toarray()
    if sparse.issparse(adata_recon.X):
        adata_recon.X = adata_recon.X.toarray()

    return [
        {
            "barcode": barcode,
            "mse": float(mse_metric(x, x_recon)),
            "exp_var_dominic": float(explained_variance_metric_v2(x, x_recon)),
            "exp_var_scikit": explained_variance_metric(x, x_recon),
            "pearsonr": float(pearson_metric(x, x_recon)),
            "spearmanr": float(spearman_metric(x, x_recon)),
            "cosine_sim": float(cosine_similarity_metric(x, x_recon)),
        }
        for barcode, x, x_recon in zip(shared_barcodes, adata.X, adata_recon.X)
    ]


def expand_entry_with_cell_metrics(
    entry: Dict[str, Any],
) -> List[Dict[str, Any]]:
    print(entry)
    cell_metrics = compute_cellwise_metrics(
        sc_data_path=entry["sc_data_path"],
        artifact_path=entry["artifact_path"],
        spatial_data_path=entry.get("spatial_data_path"),
    )

    return [
        {
            **entry,
            **cell_entry,
        }
        for cell_entry in cell_metrics
    ]


def expand_df_with_cell_metrics(df: pl.DataFrame) -> pl.DataFrame:
    expanded_rows = chain.from_iterable(
        expand_entry_with_cell_metrics(entry) for entry in df.to_dicts()
    )
    return pl.DataFrame(expanded_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand a metadata CSV into per-cell metric rows using AnnData files."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input CSV containing dataset metadata",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <input>_cell_metrics.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path: Path = args.input
    output_path: Path = (
        args.output
        if args.output is not None
        else input_path.with_name(f"{input_path.stem}_cell_metrics.csv")
    )

    df = pl.read_csv(input_path)
    df = expand_df_with_cell_metrics(df)
    df.write_csv(output_path)

    print(f"Wrote expanded dataframe to: {output_path}")


if __name__ == "__main__":
    main()
