
from src.datamodels.pipeline import DatamodelPipeline
from src.datamodels.config import DatamodelConfig
from src.llms import Llama3_1_Instruct
from src.evaluator import GleuEvaluator
import pandas as pd

import os

def train_datamodels():
    
    llama = Llama3_1_Instruct()
    data_dir =   "../../data/bbh/datamodels/reduced_sample"


    config = DatamodelConfig(
        k = 8,
        num_models= 40,
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

    # datamodel.load_collections_from_path()

    # Specify the folder path
    datamodel.train_datamodels(
        epochs=1000,
        train_batches=10,
        val_batches=5,
        val_size=0.2,
        lr=0.0001,
        random_seed=42,
        patience=50,
        subset=9500,
        log=True,
        log_epochs=25,
        run_id="bbh_dl_27",
        device="cuda:0",
    )


if __name__ == "__main__":
    train_datamodels()