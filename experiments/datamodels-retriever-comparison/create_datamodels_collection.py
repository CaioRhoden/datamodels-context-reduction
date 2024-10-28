
from src.utils import split_dev_set, subset_df
from src.retriever import DatamodelsRetriever
from src.datamodels_pipeline import Datamodels, DatamodelConfig
from src.llms import Llama3_1
from src.evaluator import Rouge_L_evaluator
import pandas as pd

import os

def run_collection():
    
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
        evaluator= Rouge_L_evaluator(),
    )


    datamodel = Datamodels(config)
    datamodel.set_test_set()
    datamodel.set_train_set()
    datamodel.set_train_collection_index()
    datamodel.set_test_collection_index()
    datamodel.set_instructions_from_path()



    # Specify the folder path
    folder_path = '../../data/instruction-induction-data/datamodels_15_10_2024/pre_collections/15-10-2024/'

    # Get a list of all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    for i in range(0, len(files)):
        print(f"Creating collection for {files[i]}")
        df = pd.read_pickle(folder_path + files[i])
        f_name = os.path.splitext(os.path.basename(files[i]))[0]
        datamodel.create_collection(batch_name=f_name, pre_collection_batch=df)
        print(f_name)
        print(df.shape)


if __name__ == "__main__":
    run_collection()