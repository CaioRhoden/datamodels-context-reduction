#bultin
from abc import ABC, abstractmethod
import h5py
import json

#custom
from src.datamodels.config import DatamodelConfig

#external
import pandas as pd


class BaseDatamodelsPipeline(ABC):

    @abstractmethod
    def __init__(self, config: DatamodelConfig) -> None:
        pass
       

    @abstractmethod
    def __str__(self):
        return f"BaseDatamodelsPipeline(k={self.k}, num_models={self.num_models}, datamodels_path={self.datamodels_path})"
    
    @abstractmethod
    def create_pre_collections(self):
        pass

    @abstractmethod
    def create_collections(self):
        pass

    @abstractmethod
    def train_datamodels(self):
        pass


