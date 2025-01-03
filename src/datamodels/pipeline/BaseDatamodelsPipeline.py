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

        """
        Initializes a new instance of the BaseDatamodelsPipeline class.
        It's expected the following files in the datamodels path:
        - train_collection.h5
        - test_collection.h5
        - train_set.feather
        - test_set.feather
        - instructions.json

        Parameters:
            config (DatamodelConfig): The configuration for the datamodels.

        Returns:
            None
        """
        self.k = config.k
        self.num_models = config.num_models
        self.datamodels_path = config.datamodels_path

        ## Initialize collections index
        print("Initializing collections index")

        with h5py.File(f"{self.datamodels_path}/train_collection.h5", "r") as f:
            self.train_collections_idx = f["train_collection"][()]
        print("Loaded train collection index")

        with h5py.File(f"{self.datamodels_path}/test_collection.h5", "r") as f:
            self.test_collections_idx = f["test_collection"][()]
        print("Loaded test collection index")
        

        ## Initialize dataframes
        print("Initializing dataframes")

        self.train_set = pd.read_csv(f"{self.datamodels_path}/train_set.feather")
        print("Loaded train set")

        self.test_set = pd.read_csv(f"{self.datamodels_path}/test_set.feather")
        print("Loaded test set")


        ## Initialize instructions
        print("Initializing instructions")

        with open(f"{self.datamodels_path}/instructions.json", "r") as f:
            self.instructions = json.load(f)
        print("Loaded instructions")
        

    @abstractmethod
    def __str__(self):
        return f"BaseDatamodelsPipeline(k={self.k}, num_models={self.num_models}, datamodels_path={self.datamodels_path})"

