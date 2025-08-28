from dataclasses import dataclass
import polars as pl
import numpy as np

@dataclass
class DatamodelsPipelineData:
    train_collections_idx: np.ndarray
    test_collections_idx: np.ndarray
    train_set: pl.DataFrame
    test_set: pl.DataFrame
    datamodels_path: str
