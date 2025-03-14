
from dmcr.utils import split_dev_set, subset_df
from dmcr.retrievers import NaiveDatamodelsRetriever
import pandas as pd


def run_pre_collection():
    
    train = pd.read_csv("../../data/instruction-induction-data/processed/induce_tasks_examples.csv")
    train_subset = subset_df(train, 200, "task")
    train_subset.to_csv("../../data/instruction-induction-data/processed/train.csv")

    split_dev_set(
        path="../../data/instruction-induction-data/processed/train.csv",
        saving_path="../../data/instruction-induction-data/datamodels/datamodels_06_11_2024",    
        k_samples=15,
        task_column="task",
    )

    retriever = NaiveDatamodelsRetriever(k=8)
    retriever.create_collections_index(
        "../../data/instruction-induction-data/datamodels/datamodels_06_11_2024/train_set.csv",
        "../../data/instruction-induction-data/datamodels/datamodels_06_11_2024",
        n_samples=1500,
        test_per=0.1,

    )

if __name__ == "__main__":
    run_pre_collection()