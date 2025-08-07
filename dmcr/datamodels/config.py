from dataclasses import dataclass
from dmcr.models import BaseLLM
from dmcr.evaluators import BaseReferenceEvaluator
import numpy as np
import pandas as pd


@dataclass
class DatamodelConfig:

    k: int
    num_models: int
    datamodels_path: str | None

@dataclass 
class DatamodelIndexBasedConfig(DatamodelConfig):
    train_set_path: str
    test_set_path: str


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



