from typing import Callable, Generator, List, Tuple
import scanpy as sc
from anndata import AnnData
import numpy as np

from exp_runner import Variable
from feature_prediction import typing
from sklearn.model_selection import StratifiedKFold




def all_at_once(
    query: AnnData, reference: AnnData, query_ct_key: str, ref_ct_key: str
) -> List[Tuple[AnnData, AnnData]]:
    return [(query, reference)]


def per_celltype(
    query: AnnData, reference: AnnData, query_ct_key: str, ref_ct_key: str
) -> List[Tuple[AnnData, AnnData]]:
    pairs = []
    shared_celltypes = np.intersect1d(
        query.obs[query_ct_key].unique(), reference.obs[ref_ct_key].unique()
    )
    for ct in shared_celltypes:
        q_chunk = query[query.obs[query_ct_key] == ct]
        r_chunk = reference[reference.obs[ref_ct_key] == ct]
        pairs.append((q_chunk, r_chunk))
    return pairs




def _wrapper(
    strategy_func: Callable[[AnnData, AnnData, str, str], List[Tuple[AnnData, AnnData]]],
) -> Callable[[AnnData, AnnData, typing.Predictor, str, str], AnnData]:
    def wrapped(
        query: AnnData,
        reference: AnnData,
        predictor: typing.Predictor,
        query_ct_key: str,
        reference_ct_key: str,
    ) -> AnnData:
        chunk_pairs = strategy_func(query, reference, query_ct_key, reference_ct_key)

        recon_chunks = [predictor(q, r) for q, r in chunk_pairs]

        recon = sc.concat(recon_chunks, axis="obs", join="inner")
        return recon

    return wrapped




def strategy_generator() -> Generator[Variable[typing.Strategy], None, None]:
    strategies = [
        Variable(_wrapper(all_at_once), {"reconstruction_strategy": "all at once"}),
        Variable(_wrapper(per_celltype), {"reconstruction_strategy": "per_celltype"}),
    ]
    for strategy in strategies:
        yield strategy
