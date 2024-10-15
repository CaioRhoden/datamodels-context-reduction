from src.retriever import BaseRetriever
import pandas as pd
import h5py
import numpy as np 


class DatamodelsRetriever(BaseRetriever):

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

        train_set = pd.read_csv(train_set_path)
        random_indices = [np.random.choice(train_set.index, self.k, replace=False) for _ in range(n_samples)]
        random_indices_array = np.array(random_indices)

        train_collection = random_indices_array[:int(test_per * n_samples)]
        test_collection = random_indices_array[int(test_per * n_samples):]

        with h5py.File(f"{save_path}/train_collection.h5", 'w') as hf:
            hf.create_dataset('train_collection', data=train_collection)
        
        with h5py.File(f"{save_path}/test_collection.h5", 'w') as hf:
            hf.create_dataset('test_collection', data=test_collection)
    


    def retrieve(self, weights_path: str) -> pd.DataFrame:

        pass