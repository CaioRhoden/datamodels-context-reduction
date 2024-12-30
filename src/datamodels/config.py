from dataclasses import dataclass
from src.llms import BaseLLM
from src.evaluator import BaseEvaluator
import numpy as np
import pandas as pd


@dataclass
class DatamodelConfig:

    k: int
    num_models: int
    datamodels_path: str | None
    train_collections_idx: np.ndarray | None
    test_collections_idx: np.ndarray | None
    test_set: pd.DataFrame | None
    train_set: pd.DataFrame | None
    instructions: dict | None
    llm: BaseLLM | None
    evaluator: BaseEvaluator | None

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



