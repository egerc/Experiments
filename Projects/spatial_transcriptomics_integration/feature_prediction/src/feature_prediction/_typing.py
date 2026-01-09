from typing import Callable, Tuple

from anndata.typing import AnnData  # type: ignore

type Dataset = Callable[[], Tuple[AnnData, AnnData, str, str]]
type Predictor = Callable[[AnnData, AnnData, str, str], AnnData]
type Strategy = Callable[[AnnData, AnnData, Predictor, str, str], AnnData]
