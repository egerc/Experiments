"""
_strategies.py

Defines reconstruction strategies for single-cell feature prediction.
Includes functions to generate and apply strategies to query and reference
AnnData objects, and a generator to iterate over available strategies.
"""

from functools import wraps
from typing import Callable, Generator, List, Tuple

import numpy as np
import scanpy as sc  # type: ignore
from anndata import AnnData  # type: ignore
from exp_runner import Variable  # type: ignore

from feature_prediction import typing


def stratified_predictor_factory(
    base_predictor: Callable[[AnnData, AnnData], AnnData],
    strategy: Callable[[AnnData, AnnData, str, str], List[Tuple[AnnData, AnnData]]],
) -> Callable[[AnnData, AnnData, str, str], AnnData]:
    def stratified_predictor(
        query: AnnData, reference: AnnData, query_ct_key: str, ref_ct_key: str
    ) -> AnnData:
        og_adatas = strategy(query, reference, query_ct_key, ref_ct_key)
        reconstructed_adatas = [
            base_predictor(query_subset, ref_subset)
            for query_subset, ref_subset in og_adatas
        ]
        return sc.concat(reconstructed_adatas)

    return stratified_predictor


def all_at_once(
    query: AnnData, reference: AnnData, query_ct_key: str, ref_ct_key: str
) -> List[Tuple[AnnData, AnnData]]:
    """
    Strategy that uses the entire dataset at once for reconstruction.

    This strategy does not split the data by cell type or any other criteria.
    It returns a single tuple containing the full query and full reference.

    Parameters
    ----------
    query : AnnData
        The query AnnData object containing cells to reconstruct.
    reference : AnnData
        The reference AnnData object containing cells used for prediction.
    query_ct_key : str
        The column in `query.obs` representing cell types (unused in this strategy).
    ref_ct_key : str
        The column in `reference.obs` representing cell types (unused in this strategy).

    Returns
    -------
    List[Tuple[AnnData, AnnData]]
        A list containing a single tuple of (query, reference).
    """
    return [(query, reference)]


def per_celltype(
    query: AnnData, reference: AnnData, query_ct_key: str, ref_ct_key: str
) -> List[Tuple[AnnData, AnnData]]:
    """
    Strategy that splits the query and reference datasets by shared cell types.

    Each query subset corresponding to a cell type is paired with the matching
    reference subset of the same cell type.

    Parameters
    ----------
    query : AnnData
        The query AnnData object containing cells to reconstruct.
    reference : AnnData
        The reference AnnData object containing cells used for prediction.
    query_ct_key : str
        The column in `query.obs` representing cell types.
    ref_ct_key : str
        The column in `reference.obs` representing cell types.

    Returns
    -------
    List[Tuple[AnnData, AnnData]]
        A list of tuples, each containing (query_subset, reference_subset) for a shared cell type.
    """
    pairs: List[Tuple[AnnData, AnnData]] = []
    shared_celltypes = np.intersect1d(
        query.obs[query_ct_key].unique(), reference.obs[ref_ct_key].unique()
    )
    for ct in shared_celltypes:
        q_chunk = query[query.obs[query_ct_key] == ct]
        r_chunk = reference[reference.obs[ref_ct_key] == ct]
        pairs.append((q_chunk, r_chunk))
    return pairs


def _wrapper(
    strategy_func: Callable[
        [AnnData, AnnData, str, str], List[Tuple[AnnData, AnnData]]
    ],
) -> Callable[[AnnData, AnnData, typing.Predictor, str, str], AnnData]:
    """
    Wraps a reconstruction strategy function to apply a predictor over dataset chunks.

    The wrapper takes a strategy function that splits query and reference datasets
    into chunks (or uses the full dataset) and applies the provided predictor
    to each pair of chunks. The reconstructed results are concatenated back
    into a single AnnData object.

    Parameters
    ----------
    strategy_func : Callable[[AnnData, AnnData, str, str], List[Tuple[AnnData, AnnData]]]
        A function that takes query and reference AnnData objects and their respective
        cell type keys, returning a list of (query_chunk, reference_chunk) pairs.

    Returns
    -------
    Callable[[AnnData, AnnData, typing.Predictor, str, str], AnnData]
        A function that takes query and reference datasets, a predictor, and the
        cell type keys, returning a reconstructed AnnData object.
    """

    @wraps(strategy_func)
    def wrapped(
        query: AnnData,
        reference: AnnData,
        predictor: typing.Predictor,
        query_ct_key: str,
        reference_ct_key: str,
    ) -> AnnData:
        chunk_pairs = strategy_func(query, reference, query_ct_key, reference_ct_key)
        recon_chunks = [predictor(q, r) for q, r in chunk_pairs]
        recon = sc.concat(recon_chunks, axis="obs", join="inner")  # type: ignore
        return recon

    return wrapped


def strategy_generator() -> Generator[Variable[typing.Strategy], None, None]:
    """
    Generates available reconstruction strategies wrapped for prediction.

    Each strategy is returned as a `Variable` object with a human-readable
    description of the reconstruction strategy.

    Yields
    ------
    Variable[typing.Strategy]
        A Variable containing a wrapped reconstruction strategy and its metadata.
    """
    strategies = [
        Variable(_wrapper(all_at_once), {"reconstruction_strategy": "all at once"}),
        Variable(_wrapper(per_celltype), {"reconstruction_strategy": "per_celltype"}),
    ]
    for strategy in strategies:
        yield strategy
