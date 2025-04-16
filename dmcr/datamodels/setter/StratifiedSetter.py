from dmcr.datamodels.setter.BaseSetter import BaseSetter
from dmcr.datamodels.setter.SetterConfig import StratifiedSetterConfig
import polars as pl
import h5py
import numpy as np 


class StratifiedSetter(BaseSetter):

    def __init__(self,
                load_path_target: str,
                load_path_random: str,
                save_path: str,
                k: int,
                n_samples_target: int,
                n_test_target: int,
                n_samples_mix: int,
                n_test_mix: int,
                n_samples_random: int,
                n_test_random: int,
                index_col: str,
                seed: int
                 
                ) -> None:
        
        """
        Initializes a new instance of the StratifiedSetter class.

        Parameters:
            load_path_target (str): The file path to the target dataset in CSV format.
            load_path_random (str): The file path to the random dataset in CSV format.
            save_path (str): The directory path where the train and test collection indices will be saved.
            k (int): The number of samples to generate for each collection.
            n_samples_target (int): The number of samples to generate for the target dataset.
            n_test_target (int): The number of test samples to generate for the target dataset.
            n_samples_mix (int): The number of samples to generate for the mixed dataset.
            n_test_mix (int): The number of test samples to generate for the mixed dataset.
            n_samples_random (int): The number of samples to generate for the random dataset.
            n_test_random (int): The number of test samples to generate for the random dataset.
            seed (int): The seed to use for randomizing the samples.

        Returns:
            None
        """
        self.config = StratifiedSetterConfig(
            load_path_target,
            load_path_random,
            save_path,
            k,
            n_samples_target,
            n_test_target,
            n_samples_mix,
            n_test_mix,
            n_samples_random,
            n_test_random,
            index_col,
            seed
        )

    def set(self) -> None:

        
        """
        This function reads the target and random datasets, creates collections of samples
        from both datasets, and then splits these collections into train and test collections.
        The collections are saved as HDF5 files.


        Notes:
        - It's expected IPC files (arrow, feather....)
        - The index column should be the same for both datasets
        - The samples that goes to the test are the first ones

        Returns:
            None
        """
        ### Read data
        target_set = pl.read_ipc(self.config.load_path_target)
        random_set = pl.read_ipc(self.config.load_path_random)
        train = pl.concat([target_set, random_set], how="diagonal")
        target_size = len(target_set)


        ### Get target samples

        random_indices_target = [np.random.choice(train[:target_size].select(self.config.index_col).to_numpy().squeeze(), self.config.k, replace=False) for _ in range(self.config.n_samples_target)]
        random_indices_array = np.array(random_indices_target)
        train_target_collection = random_indices_array[int(self.config.n_test_target):]
        test_target_collection = random_indices_array[:int(self.config.n_test_target)]

        ## Get random samples
        random_indices_random = [np.random.choice(train[target_size:].select(self.config.index_col).to_numpy().squeeze(), self.config.k, replace=False) for _ in range(self.config.n_samples_random)]
        random_indices_array = np.array(random_indices_random)
        train_random_collection = random_indices_array[int(self.config.n_test_random):]
        test_random_collection = random_indices_array[:int(self.config.n_test_random)]


        ## Get mix samples
        random_indices_mix = []
        for _ in range(self.config.n_samples_mix):
            n_target = np.random.randint(1, self.config.k-1)
            n_random = self.config.k - n_target
            target_indices = np.random.choice(train[:target_size].select(self.config.index_col).to_numpy().squeeze(), n_target, replace=False)
            random_indices = np.random.choice(train[target_size:].select(self.config.index_col).to_numpy().squeeze(), n_random, replace=False)
            mix_indices = np.concatenate((target_indices, random_indices))
            random_indices_mix.append(mix_indices)

        train_mix_collection = np.array(random_indices_mix)[int(self.config.n_test_mix):]
        test_mix_collection = np.array(random_indices_mix)[:int(self.config.n_test_mix)]


        ## Concatenate train and test
        train_collection = np.concatenate((train_target_collection, train_random_collection, train_mix_collection))
        test_collection = np.concatenate((test_target_collection, test_random_collection, test_mix_collection))

        ## Save files
        with h5py.File(f"{self.config.save_path}/train_collection.h5", 'w') as hf:
            hf.create_dataset('train_collection', data=train_collection)
        
        with h5py.File(f"{self.config.save_path}/test_collection.h5", 'w') as hf:
            hf.create_dataset('test_collection', data=test_collection)

        train.write_csv(f"{self.config.save_path}/train_set.csv")
    