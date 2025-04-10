from dmcr.datamodels.setter.BaseSetter import BaseSetter
from dmcr.datamodels.setter.SetterConfig import NaiveSetterConfig
import pandas as pd
import h5py
import numpy as np 


class NaiveSetter(BaseSetter):

    def __init__(self,
                load_path: str,
                save_path: str,
                k: int,
                n_samples: int,
                test_samples: int
                 
                ) -> None:
        
        self.config = NaiveSetterConfig(
            load_path=load_path,
            save_path=save_path,
            k=k,
            n_samples=n_samples,
            test_samples=test_samples
        )

    def set(self) -> None:

        
        
        """
        This function reads a CSV file containing the training dataset, randomly selects indices to create collections, and 
        then splits these indices into train and test collections. The collections are saved as HDF5 files.

        Parameters:
            train_set_path (str): The file path to the training dataset in CSV format.
            save_path (str): The directory path where the train and test collection indices will be saved.
            n_samples (int): The number of sample indices to generate for the collections.
            test_per (float): The proportion of samples to allocate to the test collection, 
                            represented as a float between 0 and 1.

        Returns:
            None
        """
        
        train_set = pd.read_csv(self.config.load_path)

        random_indices = [np.random.choice(train_set.index, self.k, replace=False) for _ in range(self.config.n_samples)]
        random_indices_array = np.array(random_indices)


        test_collection = random_indices_array[:int(self.config.test_samples)]
        train_collection = random_indices_array[int(self.config.test_samples):]

        with h5py.File(f"{self.config.save_path}/train_collection.h5", 'w') as hf:
            hf.create_dataset('train_collection', data=train_collection)
        
        with h5py.File(f"{self.config.save_path}/test_collection.h5", 'w') as hf:
            hf.create_dataset('test_collection', data=test_collection)
    