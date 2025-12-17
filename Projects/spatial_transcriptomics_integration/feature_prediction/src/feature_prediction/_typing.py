from typing import Callable, Tuple
from anndata.typing import AnnData
from numpy.typing import NDArray
from numpy import number

type Dataset = Callable[[], Tuple[AnnData, AnnData]]
type Predictor = Callable[[AnnData, AnnData], AnnData]
type Strategy = Callable[[AnnData, AnnData, Predictor], AnnData]
