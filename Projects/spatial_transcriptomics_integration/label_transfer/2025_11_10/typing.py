from typing import Callable, List, Tuple

from anndata.typing import AnnData
from numpy.typing import NDArray
from numpy import str_

type dataset_type = Tuple[AnnData, AnnData]
type dataset_labels = List[str]
type method_type = Callable[[AnnData, AnnData, str], dataset_labels]