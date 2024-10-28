
from src.utils import split_dev_set, subset_df
from src.retriever import DatamodelsRetriever
from src.datamodels_pipeline import Datamodels, DatamodelConfig
from src.llms import Llama3_1
import pandas as pd

import os

def run_pre_collection():
    
    llama = Llama3_1()

    config = DatamodelConfig(
        k = 8,
        train_collections_idx_path = "../../data/instruction-induction-data/datamodels_15_10_2024/train_collection.h5",
        train_collections_idx = None,
        test_collections_idx_path = "../../data/instruction-induction-data/datamodels_15_10_2024/test_collection.h5",
        test_collections_idx = None,
        test_set = None,
        test_set_path = "../../data/instruction-induction-data/datamodels_15_10_2024/dev_set.csv",
        train_set = None,
        train_set_path = "../../data/instruction-induction-data/datamodels_15_10_2024/train_set.csv",
        collections_path = "../../data/instruction-induction-data/datamodels_15_10_2024/collections/15-10-2024",
        pre_collections_path = "../../data/instruction-induction-data/datamodels_15_10_2024/pre_collections/15-10-2024",
        instructions= None,
        instructions_path= "../../data/instruction-induction-data/datamodels_15_10_2024/intructions.json",
        llm = llama,
        model =  None,
    )


    datamodel = Datamodels(config)
    datamodel.get_test_set()
    datamodel.get_train_set()
    datamodel.get_train_collection_index()
    datamodel.get_test_collection_index()
    datamodel.set_instructions_from_path()

    print(datamodel.train_collections_idx.shape)
    print("Start Creating Pre Collection")
    datamodel.create_pre_collection(start_idx = 390, end_idx = 400, type="train")


if __name__ == "__main__":
    run_pre_collection()