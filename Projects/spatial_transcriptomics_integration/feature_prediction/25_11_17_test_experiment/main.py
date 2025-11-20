from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generator,
    List,
    Tuple,
)
from anndata import AnnData
from exp_runner import MetaData, Variable, runner, VarProduct
from numpy import number
import scanpy as sc
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from spatial_transcript_inference.nmf_utils import nmf_predictor
import numpy as np
from numpy.typing import NDArray


"""
entity: predictor | dataset

predictor_metadata: predictor_name
dataset_metadata: split_id | tissue | organism |

measurements: mse | mae | pearsonr | spearmanr | explained_variability
measurements_metadata
"""

type predictor_type = Callable[[NDArray[number], NDArray[number]], NDArray[number]]
type dataset_type = Tuple[NDArray[number], NDArray[number], NDArray[number]]
        


@dataclass
class Input(VarProduct):
    predictor: predictor_type
    dataset: dataset_type



@runner()
def experiment(input: Input) -> List[MetaData]:
    predictor = input.predictor
    X_train, X_test, X_true = input.dataset
    X_pred = predictor(X_test, X_train)
    mse = float(mean_absolute_error(X_true, X_pred))
    return [{"mse": mse}]

def main():
    predictors = [Variable(nmf_predictor, {"name": "nmf_predictor"})]
    datasets = (Variable((np.array([1]), np.array(1), np.array(1)), {"tissue": None}) for i in range(2))

    inputs = Input.generate_from((predictors, datasets))

    experiment(inputs)


if __name__ == "__main__":
    main()
