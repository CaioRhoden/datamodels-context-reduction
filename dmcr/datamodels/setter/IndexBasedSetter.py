from dmcr.datamodels.setter.BaseSetter import BaseSetter
from dmcr.datamodels.setter.SetterConfig import IndexBasedSetterConfig
import polars as pl
import h5py
import numpy as np 
import os

class IndexBasedSetter(BaseSetter):
    # TODO: Implement seed for the choice method
    def __init__(self,
                 config: IndexBasedSetterConfig
                 ) -> None:
        
        self.config = config

    def set(self) -> None:

        
        
        """
        This function reads a CSV file containing the training dataset, randomly selects indices to create collections, and 
        then splits these indices into train and test collections. The collections are saved as HDF5 files.

        Parameters:
            train_set_path (str): The file path to the training dataset in CSV format.
            save_path (str): The directory path where the train and test collection indices will be saved.
            train_samples (int): The number of sample indices to generate for the collections.
            test_per (float): The proportion of samples to allocate to the test collection, 
                            represented as a float between 0 and 1.

        Returns:
            None
        """
        
        if not os.path.exists(f"{self.config.save_path}"):
            raise Exception("Save path does not exist for IndexBasedSetter")
        
        if (
            self.config.size_index <= 0 or
            self.config.k <= 0 or
            self.config.train_samples <= 0 or
            self.config.test_samples <= 0
        ):
            raise Exception("Invalid parameters for IndexBasedSetterConfig, integer parameters should be positive")

        train_random_indices = [np.random.randint(self.config.size_index, size=self.config.k) for _ in range(self.config.train_samples)]
        train_indices_array = np.array(train_random_indices)
        test_random_indices = [np.random.randint(self.config.size_index, size=self.config.k) for _ in range(self.config.test_samples)]
        test_indices_array = np.array(test_random_indices)


        with h5py.File(f"{self.config.save_path}/train_collection.h5", 'w') as hf:
            hf.create_dataset('train_collection', data=train_indices_array)
        
        with h5py.File(f"{self.config.save_path}/test_collection.h5", 'w') as hf:
            hf.create_dataset('test_collection', data=test_indices_array)
    