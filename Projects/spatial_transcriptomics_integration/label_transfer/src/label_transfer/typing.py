from typing import Callable, List, Optional, Tuple

from anndata.typing import AnnData

type dataset_type = Tuple[AnnData, AnnData]
type dataset_labels = List[Optional[str]]
type method_type = Callable[[AnnData, AnnData, str], dataset_labels]