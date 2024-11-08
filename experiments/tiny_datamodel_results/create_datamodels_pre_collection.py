
from src.utils import split_dev_set, subset_df
from src.datamodels_pipeline import DatamodelPipeline, DatamodelConfig
from src.llms import Llama3_1
from src.evaluator import Rouge_L_evaluator
import torch

import pandas as pd

import os

def run_pre_collection():
    


    llama = Llama3_1()

    config = DatamodelConfig(
        k = 8,
        num_models= 105,
        datamodels_path = "../../data/instruction-induction-data/datamodels/tiny_datamodels_07_11_2024",
        train_collections_idx = None,
        test_collections_idx = None,
        test_set = None,
        train_set = None,
        instructions= None,
        llm = llama,
        evaluator=Rouge_L_evaluator(),
    )

    datamodel = DatamodelPipeline(config)
    datamodel.set_collections_index()
    datamodel.set_dataframes()
    datamodel.set_instructions_from_path()

    print("Start Creating Pre Collection")
    datamodel.create_pre_collection(start_idx = 0, end_idx = 20, type="test")


if __name__ == "__main__":
    run_pre_collection()