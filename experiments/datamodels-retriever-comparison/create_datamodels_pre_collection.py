
from dmcr.utils import split_dev_set, subset_df
from dmcr.datamodels_pipeline import DatamodelPipeline, DatamodelConfig
from dmcr.models import Llama3_1
from dmcr.evaluators import Rouge_L_evaluator
import torch

import pandas as pd

import os

def run_pre_collection():
    


    llama = Llama3_1()

    config = DatamodelConfig(
        k = 8,
        num_models= 315,
        datamodels_path = "../../data/instruction-induction-data/datamodels/datamodels_06_11_2024",
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
    datamodel.create_pre_collection(start_idx = 899, end_idx = 900, type="train")


if __name__ == "__main__":
    run_pre_collection()