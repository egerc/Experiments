from typing import Any, Callable, Generator

from exp_runner import Variable
from .utils import Dataset



def dataset_generator() -> Generator[Variable[Callable[[], Dataset]], Any, None]: ...