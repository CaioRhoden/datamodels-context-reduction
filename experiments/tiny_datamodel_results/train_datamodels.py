
from src.datamodels_pipeline import DatamodelPipeline, DatamodelConfig
from src.llms import Llama3_1
from src.evaluator import GleuEvaluator
import pandas as pd

import os

def train_datamodels():
    
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
        evaluator=GleuEvaluator(),
    )

    datamodel = DatamodelPipeline(config)
    datamodel.set_collections_index()
    datamodel.set_dataframes()
    datamodel.set_instructions_from_path()



    # Specify the folder path
    datamodel.train_datamodels(
        epochs=100,
        batch_size=100,
        val_split=0.1,
        lr=0.001,
        random_seed=42,
    )


if __name__ == "__main__":
    train_datamodels()