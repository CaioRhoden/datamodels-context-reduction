
from src.datamodels.pipeline import DatamodelPipeline
from src.datamodels.config import DatamodelConfig
from src.llms import GPT2
from src.evaluator import GleuEvaluator
import argparse
import pandas as pd
import datetime
import os

def extract_idx(file: str):
    return int(file.split("_")[-1].split(".")[0])

def format_collection_filename(filename: str):
    name = os.path.splitext(os.path.basename(filename))[0].replace("pre_", "")
    return name

def run_collection(start_idx, end_idx, test_flag = False):
    

    
    
    llama = GPT2()
    data_dir =  "../../data/instruction-induction-data/datamodels/proportion_study/gpt2_420_5"


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
    pre_collection_path = f"{data_dir}/pre_collections/"
    collection_path = f"{data_dir}/collections/"

    # Get a list of all files in the folder
    files = [f for f in os.listdir(pre_collection_path) if os.path.isfile(os.path.join(pre_collection_path, f))]
    collections = [f for f in os.listdir(collection_path) if os.path.isfile(os.path.join(collection_path, f))]
    print(len(files))
    
    test_files = [f for f in files if f.startswith("test")]
    train_files = [f for f in files if not f.startswith("test")]

    if test_flag:
        ordered_files = sorted(test_files, key=extract_idx)
    else:
        ordered_files = sorted(train_files, key=extract_idx)

    ## Select the non repeated collections
    f_names = [format_collection_filename(ordered_files[i]) for i in range(len(ordered_files))]
    f_names = [name for name in f_names if name not in collections]
    end_idx = min(len(f_names), end_idx)

    if end_idx == -1:
        end_idx = len(ordered_files)
    

    
    for i in range(start_idx, end_idx):
        print(f"Creating collection from pre_collection {i} of {end_idx}")

        df = pd.read_feather(f"{pre_collection_path}pre_{f_names[i]}.feather")
        datamodel.create_collection(batch_name=f"pre_{f_names[i]}", pre_collection_batch=df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass start and end index")
    parser.add_argument("-s", "--start_idx", type=int, help="Start index")
    parser.add_argument("-e", "--end_idx", type=int, help="End index")
    parser.add_argument("-t", "--test_flag", action='store_true', help="Test flag")

    args = parser.parse_args()

    run_collection(args.start_idx, args.end_idx, args.test_flag)