from dmcr.datamodels.setter.BaseSetter import BaseSetter
from dmcr.datamodels.setter.SetterConfig import NaiveSetterConfig
import polars as pl
import h5py
import numpy as np 


class NaiveSetter(BaseSetter):
    # TODO: Implement seed for the choice method
    def __init__(self,
                load_path: str,
                save_path: str,
                k: int,
                index_col: str,
                train_samples: int,
                test_samples: int
                
                ) -> None:
        
        self.config = NaiveSetterConfig(
            load_path=load_path,
            save_path=save_path,
            k=k,
            train_samples=train_samples,
            test_samples=test_samples,
            index_col=index_col
        )

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
        if self.config.load_path.endswith(".csv"): 
            train_set = pl.read_csv(self.config.load_path)

        elif self.config.load_path.endswith(".feather"):
            train_set = pl.read_ipc(self.config.load_path)

        else:
            raise ValueError(f"Unsupported file format: {self.config.load_path}")

        train_random_indices = [np.random.choice(train_set.select(self.config.index_col).to_numpy().squeeze(), self.config.k, replace=False) for _ in range(self.config.train_samples)]
        train_indices_array = np.array(train_random_indices)
        test_random_indices = [np.random.choice(train_set.select(self.config.index_col).to_numpy().squeeze(), self.config.k, replace=False) for _ in range(self.config.test_samples)]
        test_indices_array = np.array(test_random_indices)


        with h5py.File(f"{self.config.save_path}/train_collection.h5", 'w') as hf:
            hf.create_dataset('train_collection', data=train_indices_array)
        
        with h5py.File(f"{self.config.save_path}/test_collection.h5", 'w') as hf:
            hf.create_dataset('test_collection', data=test_indices_array)
    