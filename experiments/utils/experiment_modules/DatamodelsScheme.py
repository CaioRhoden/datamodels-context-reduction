from experiments.utils.experiment_modules import BaseExpeimentModule
from experiments.utils.experiment_modules import DatamodelSchemeConfig
from experiments.utils.DatamodelsExperiment import DatamodelsExperiment

import pandas as pd
import numpy as np
import h5py

class DatamodelsScheme(BaseExpeimentModule):

    def __init__(self, config: DatamodelSchemeConfig) -> None:
        self.config = config    


    def __call__(self, experiment: DatamodelsExperiment) -> None:
        """
        Creates a datamodel scheme given an experiment instance.

        Args:
        experiment (DatamodelsExperiment): The experiment instance.

        Returns:
        None
        """

        if self.config.train_file.endswith(".feather") and self.config.test_file.endswith(".feather"):
            train: pd.DataFrame = pd.read_feather(f"{experiment.folder_path}/{self.config.train_file}")
            test: pd.DataFrame = pd.read_feather(f"{experiment.folder_path}/{self.config.test_file}")
        
        elif self.config.train_file.endswith(".csv") and self.config.test_file.endswith(".csv"):
            train: pd.DataFrame = pd.read_csv(f"{experiment.folder_path}/{self.config.train_file}")
            test: pd.DataFrame = pd.read_csv(f"{experiment.folder_path}/{self.config.test_file}")

        else:
            raise ValueError("Invalid file format for train or test file")
        

        ## Select tasks
        tasks: np.ndarray = train["task"].unique()

        ## Random number of tasks
        if isinstance(self.config.tasks, int):
            selected_tasks: np.ndarray = np.random.choice(tasks, self.config.tasks, replace=False)

        ## List of tasks
        elif isinstance(self.config.tasks, list):
            selected_tasks: np.ndarray = self.config.tasks

            """
            TODO: Check if all tasks are in the dataset
            """

        ## String of tasks
        elif isinstance(self.config.tasks, str) and self.config.tasks == "all":
            selected_tasks: np.ndarray = tasks

        else:
            raise ValueError("Invalid tasks self.configuration")

        
        train: pd.DataFrame = train[train["task"].isin(selected_tasks)].reset_index(drop=True)
        test: pd.DataFrame = test[test["task"].isin(selected_tasks)].reset_index(drop=True)

        print("Sampling train dataset")

        ## Sample train dataset
        total_samples: int = self.config.train_samples + self.config.test_samples
        train: pd.DataFrame = train.groupby("task").apply(lambda x: x.sample(n=total_samples)).reset_index(drop=True)


        print("Splitting train into train and dev")

        

        ## Split train into train and dev
        dev_set: pd.DataFrame =  train.groupby("task").apply(lambda x: x.sample(n=min(len(x), self.config.test_samples)))
        dev_idxs: np.ndarray = np.array([i[1] for i in dev_set.index.values])
        train_set: pd.DataFrame = train.drop(dev_idxs).reset_index(drop=True)
        dev_set = dev_set.reset_index(drop=True)


        print("Saving collections and datasets")


        ### Create collection indexes

        total_collections: int = self.config.num_train_collections + self.config.num_test_collections
        split_idx: int = total_collections - self.config.num_test_collections

        random_indices: np.ndarray = np.array([np.random.choice(train_set.index, experiment.k, replace=False) for _ in range(total_collections)])

        train_collection: np.ndarray = random_indices[:split_idx]
        test_collection: np.ndarray = random_indices[split_idx:]

        ### Save files

        with h5py.File(f"{experiment.experiment_path}/train_collection.h5", 'w') as hf:
            hf.create_dataset('train_collection', data=train_collection)
        
        with h5py.File(f"{experiment.experiment_path}/test_collection.h5", 'w') as hf:
            hf.create_dataset('test_collection', data=test_collection)


        train_set.to_feather(f"{experiment.experiment_path}/train_set.feather")
        dev_set.to_feather(f"{experiment.experiment_path}/test_set.feather")
    
    def __str__(self):
        return f"DatamodelsScheme({self.config})"