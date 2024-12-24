import argparse
from src.utils import subset_df, split_dev_set
from src.retriever import NaiveDatamodelsRetriever

import pandas as pd
import numpy as np

def get_reduced_dataset(task_quantity, dataset_name, random_seed=42):

    ## Load dataframe
    path = "../../data/bbh/processed"
    train = pd.read_feather(f"{path}/train.feather")
    test = pd.read_feather(f"{path}/test.feather")

    ## Select random tasks
    tasks = train["task"].unique()
    np.random.seed(random_seed)
    selected_tasks = np.random.choice(tasks, task_quantity, replace=False)
    train = train[train["task"].isin(selected_tasks)]
    test = test[test["task"].isin(selected_tasks)]

    ## Save dataframe
    split_dev_set(df=train, saving_path=f"{path}", k_samples=10, task_column="task", prefix=f"{dataset_name}_")
    
    test.to_feather(f"{path}/test_{dataset_name}")


    ## Create pre collections
    retriever = NaiveDatamodelsRetriever(k=8)
    retriever.create_collections_index(
        "../../data/bbh/processed/bbh_sample_train_set.csv",
        "../../data/bbh/datamodels/reduced_sample",
        n_samples=10000,
        test_per=0.05,

    )




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass start and end index")
    parser.add_argument("-t", "--task_quantity", type=int, help="Number of tasks")
    parser.add_argument("-n", "--dataset_name", type=str, help="Name of dataset to be reduced")
    parser.add_argument("--random_seed", type=int, help="Random seed to be used")

    args = parser.parse_args()

    get_reduced_dataset(args.task_quantity, args.dataset_name, args.random_seed)