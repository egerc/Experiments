from typing import Callable, Tuple
from anndata.typing import AnnData

type Dataset = Callable[[], Tuple[AnnData, AnnData, str, str]]
type Predictor = Callable[[AnnData, AnnData], AnnData]
type Strategy = Callable[[AnnData, AnnData, Predictor, str, str], AnnData]
