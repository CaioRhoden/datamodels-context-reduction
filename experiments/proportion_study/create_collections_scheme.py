
from dmcr.utils import split_dev_set, subset_df
from dmcr.datamodels.setter import NaiveSetter
import pandas as pd


def run_pre_collection():
    
    train = pd.read_csv("../../data/instruction-induction-data/processed/induce_tasks_examples.csv")
    train_subset = subset_df(train, 25, "task")
    train_subset.to_csv("../../data/instruction-induction-data/processed/train.csv")

    split_dev_set(
        path="../../data/instruction-induction-data/processed/train.csv",
        saving_path="../../data/instruction-induction-data/datamodels/proportion_study/210_5",    
        k_samples=5,
        task_column="task",
    )

    retriever = NaiveSetter(
        load_path= "../../data/instruction-induction-data/datamodels/proportion_study/210_5/train_set.csv",
        save_path="../../data/instruction-induction-data/datamodels/proportion_study/210_5",
        n_samples=120000,
        test_samples=12000,
        k=8
    )
    retriever.set()

if __name__ == "__main__":
    run_pre_collection()