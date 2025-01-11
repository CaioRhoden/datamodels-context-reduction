
from src.utils import split_dev_set, subset_df
from src.datamodels.pipeline import DatamodelPipeline
from src.datamodels.config import DatamodelConfig, LogConfig
from src.llms import Llama3_1_Instruct, GPT2
from src.evaluator import GleuEvaluator
import torch
import argparse

import pandas as pd

import os

def run_pre_collection(start_idx, end_idx, type):
    
    llama = GPT2()

    config = DatamodelConfig(
        k = 8,
        num_models= 40,
        datamodels_path = "../../data/bbh/datamodels/reduced_sample_gpt2",
        train_collections_idx = None,
        test_collections_idx = None,
        test_set = None,
        train_set = None,
        instructions= None,
        llm = llama,
        evaluator=GleuEvaluator(),
    )

    log_config = LogConfig(
        project="bbh-gp2",
        dir="log/bbh_gpt2",
        id="bbh_gpt2",
        name="bbh_gpt2",
        config={
            "k": 8,
            "num_models": 40,
            "evaluator": "GleuEvaluator",
            "llm": "GPT2",
            "gpu": "Quadro RTX500",
        },
        tags=["bbh", "dl-27", "pre_collections"],
    )



    datamodel = DatamodelPipeline(config)
    datamodel.set_collections_index()
    datamodel.set_dataframes()
    datamodel.set_instructions_from_path()

    print("Start Creating Pre Collection")
    datamodel.create_pre_collection(start_idx = start_idx, end_idx = end_idx, type=type, log=True, log_config=log_config, checkpoint=25)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass start and end index")
    parser.add_argument("-s", "--start_idx", type=int, help="Start index")
    parser.add_argument("-e", "--end_idx", type=int, help="End index")
    parser.add_argument("-t", "--type", type=str, help="Type")

    args = parser.parse_args()

    run_pre_collection(args.start_idx, args.end_idx, args.type)