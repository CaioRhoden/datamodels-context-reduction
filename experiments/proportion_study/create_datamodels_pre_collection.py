
from src.utils import split_dev_set, subset_df
from src.datamodels.pipeline import DatamodelPipeline
from src.datamodels.config import DatamodelConfig
from src.llms import Llama3_1
from src.evaluator import GleuEvaluator
import torch
import argparse

import pandas as pd

import os

def run_pre_collection(start_idx, end_idx):
    
    llama = Llama3_1()

    config = DatamodelConfig(
        k = 8,
        num_models= 105,
        datamodels_path = "../../data/instruction-induction-data/datamodels/proportion_study/210_5",
        train_collections_idx = None,
        test_collections_idx = None,
        test_set = None,
        train_set = None,
        instructions= None,
        llm = llama,
        evaluator=GleuEvaluator(),
    )


    datamodel = DatamodelPipeline(config)
    datamodel.set_collections_index()
    datamodel.set_dataframes()
    datamodel.set_instructions_from_path()

    print("Start Creating Pre Collection")
    datamodel.create_pre_collection(start_idx = start_idx, end_idx = end_idx, type="train")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass start and end index")
    parser.add_argument("-s", "--start_idx", type=int, help="Start index")
    parser.add_argument("-e", "--end_idx", type=int, help="End index")

    args = parser.parse_args()

    run_pre_collection(args.start_idx, args.end_idx)