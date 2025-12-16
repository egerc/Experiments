from typing import Any, Generator

from exp_runner import Variable

from feature_prediction import typing


def predictor_generator() -> Generator[Variable[typing.Predictor], Any, None]: ...
    def nmf_predictor(): ...