from typing import Callable, Tuple

from anndata.typing import AnnData
from numpy.typing import NDArray
from numpy import str_

type dataset_type = Tuple[AnnData, AnnData]
type dataset_labels = NDArray[str_]
type method_type = Callable[[AnnData, AnnData], dataset_labels]