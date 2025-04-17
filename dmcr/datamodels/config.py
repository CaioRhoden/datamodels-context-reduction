from dataclasses import dataclass
from dmcr.models import BaseLLM
from dmcr.evaluators import BaseEvaluator
import numpy as np
import pandas as pd


@dataclass
class DatamodelConfig:

    k: int
    num_models: int
    datamodels_path: str | None


@dataclass
class MemMapConfig:

    filename: str
    dtype: type
    shape: tuple
    mode: str


@dataclass
class LogConfig:

    project: str
    dir: str
    id: str
    name: str
    config: dict
    tags: list



