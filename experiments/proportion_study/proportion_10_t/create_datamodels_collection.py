
from src.datamodels_pipeline import DatamodelPipeline, DatamodelConfig
from src.llms import Llama3_1
from src.evaluator import GleuEvaluator
import pandas as pd

import os

def run_collection():
    
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
    folder_path = '../../data/instruction-induction-data/datamodels/tiny_datamodels_07_11_2024/pre_collections/'

    # Get a list of all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort()
    
    for i in range(0,len(files)):
        print(f"Creating collection for {files[i]}")
        df = pd.read_pickle(folder_path + files[i])
        f_name = os.path.splitext(os.path.basename(files[i]))[0]
        datamodel.create_collection(batch_name=f_name, pre_collection_batch=df)
        print(f_name)
        print(df.shape)


if __name__ == "__main__":
    run_collection()