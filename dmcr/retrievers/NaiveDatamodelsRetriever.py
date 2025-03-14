from dmcr.retrievers import BaseRetriever
import pandas as pd
import h5py
import numpy as np 


class NaiveDatamodelsRetriever(BaseRetriever):

    def __init__(self,
                 k: int,
                 
                ) -> None:
        
        self.k = k

    def create_collections_index(self, 
                           train_set_path: str, 
                           save_path: str,
                           n_samples: int,
                           test_per: float,
                           
                           ) -> None:

        
        
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

        train_set = pd.read_csv(train_set_path)

        random_indices = [np.random.choice(train_set.index, self.k, replace=False) for _ in range(n_samples)]
        random_indices_array = np.array(random_indices)

        test_collection = random_indices_array[:int(test_per * n_samples)]
        train_collection = random_indices_array[int(test_per * n_samples):]

        with h5py.File(f"{save_path}/train_collection.h5", 'w') as hf:
            hf.create_dataset('train_collection', data=train_collection)
        
        with h5py.File(f"{save_path}/test_collection.h5", 'w') as hf:
            hf.create_dataset('test_collection', data=test_collection)
    


    def retrieve(self, weights_path: str) -> pd.DataFrame:

        pass