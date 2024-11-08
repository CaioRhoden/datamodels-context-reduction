
from src.utils import split_dev_set, subset_df
from src.retriever import NaiveDatamodelsRetriever
import pandas as pd


def run_pre_collection():
    
    train = pd.read_csv("../../data/instruction-induction-data/processed/induce_tasks_examples.csv")
    train_subset = subset_df(train, 25, "task")
    train_subset.to_csv("../../data/instruction-induction-data/processed/train.csv")

    split_dev_set(
        path="../../data/instruction-induction-data/processed/train.csv",
        saving_path="../../data/instruction-induction-data/datamodels/tiny_datamodels_07_11_2024",    
        k_samples=5,
        task_column="task",
    )

    retriever = NaiveDatamodelsRetriever(k=8)
    retriever.create_collections_index(
        "../../data/instruction-induction-data/datamodels/tiny_datamodels_07_11_2024/train_set.csv",
        "../../data/instruction-induction-data/datamodels/tiny_datamodels_07_11_2024",
        n_samples=200,
        test_per=0.1,

    )

if __name__ == "__main__":
    run_pre_collection()