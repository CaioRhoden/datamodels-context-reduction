
from src.datamodels.pipeline import DatamodelPipeline
from src.datamodels.config import DatamodelConfig
from src.llms import Llama3_1_Instruct
from src.evaluator import GleuEvaluator
import argparse
import pandas as pd
import datetime
import os

def extract_idx(file: str):
    return int(file.split("_")[-1].split(".")[0])

def format_collection_filename(filename: str):
    name = os.path.splitext(os.path.basename(filename))[0].replace("pre_", "")+".feather"
    return name

def run_collection(start_idx, end_idx, test_flag = False):
    

    
    
    llama = Llama3_1_Instruct()
    data_dir =  "../../data/bbh/datamodels/reduced_sample"


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
    datamodel.set_collections_index()
    datamodel.set_dataframes()
    datamodel.set_instructions_from_path()



    # Specify the folder path
    train_pre_collection_path = f"{data_dir}/pre_collections/train/"
    test_pre_collection_path = f"{data_dir}/pre_collections/test/"
    train_collection_path = f"{data_dir}/collections/train/"
    test_collection_path = f"{data_dir}/collections/test/"

    # Get a list of all files in the folder
    train_files = [f for f in os.listdir(train_pre_collection_path) if os.path.isfile(os.path.join(train_pre_collection_path, f))]
    test_files = [f for f in os.listdir(test_pre_collection_path) if os.path.isfile(os.path.join(test_pre_collection_path, f))]
    train_collections = [f for f in os.listdir(train_collection_path) if os.path.isfile(os.path.join(train_collection_path, f))]
    test_collections = [f for f in os.listdir(test_collection_path) if os.path.isfile(os.path.join(test_collection_path, f))]
    
    

    if test_flag:
        ordered_files = sorted(test_files, key=extract_idx)
        f_names = [format_collection_filename(ordered_files[i]) for i in range(len(ordered_files))][start_idx:end_idx]
        f_names = [name for name in f_names if name not in train_collections]

        if end_idx == -1:
            end_idx = len(ordered_files)
    else:
        ordered_files = sorted(train_files, key=extract_idx)
        f_names = [format_collection_filename(ordered_files[i]) for i in range(len(ordered_files))][start_idx:end_idx]
        f_names = [name for name in f_names if name not in test_collections]

        if end_idx == -1:
            end_idx = len(ordered_files)


    
    for i in range(len(f_names)):
        print(f"Creating collection from pre_collection {f_names[i]}, {i} of {len(f_names)}")

        if test_flag:
            pre_collection_path = test_pre_collection_path
            batch_name = f"test/{f_names[i]}"
        else:
            pre_collection_path = train_pre_collection_path
            batch_name = f"train/{f_names[i]}"

        df = pd.read_feather(f"{pre_collection_path}pre_{f_names[i]}")
        datamodel.create_collection(batch_name=batch_name, pre_collection_batch=df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass start and end index")
    parser.add_argument("-s", "--start_idx", type=int, help="Start index")
    parser.add_argument("-e", "--end_idx", type=int, help="End index")
    parser.add_argument("-t", "--test_flag", action='store_true', help="Test flag")

    args = parser.parse_args()

    run_collection(args.start_idx, args.end_idx, args.test_flag)