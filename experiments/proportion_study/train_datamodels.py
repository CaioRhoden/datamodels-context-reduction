
from src.datamodels.pipeline import DatamodelPipeline
from src.datamodels.config import DatamodelConfig
from src.llms import Llama3_1
from src.evaluator import GleuEvaluator
import pandas as pd

import os

def train_datamodels():
    
    llama = Llama3_1()
    data_dir =  "../../data/instruction-induction-data/datamodels/proportion_study/210_5"


    config = DatamodelConfig(
        k = 8,
        num_models= 105,
        datamodels_path = data_dir,
        train_collections_idx = None,
        test_collections_idx = None,
        test_set = None,
        train_set = None,
        instructions= None,
        llm = llama,
        evaluator=GleuEvaluator(),
    )

    datamodel = DatamodelPipeline(config)

    # Specify the folder path
    datamodel.train_datamodels(
        epochs=1000,
        train_batches=1800,
        val_batches=1000,
        val_size=0.1,
        lr=0.0001,
        random_seed=42,
        patience=25,
        subset=40000,
        device="cuda:0",
    )


if __name__ == "__main__":
    train_datamodels()