import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import polars as pl
import torch
import numpy as np
import json
import datetime

## LLM
from src.llms import Llama3_1_Instruct

seed = 42
# NumPy
np.random.seed(seed)

# PyTorch
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if torch.cuda.is_available():
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_taco(num_generations: int, num_sequences: int, train_input: pl.DataFrame):
    
    PATH = "../../data/TACO/processed"


    llm = Llama3_1_Instruct()

    for idx in range(len(train_input)):
        print(f"Generating Iteration {idx}")
        print(f"Time: {datetime.datetime.now()}")
        sample_idx = int(train_input[idx].select("id").unique().to_numpy().squeeze(1)[0])
        input_example = train_input[idx].select("input").unique().to_numpy().squeeze(1)[0]
        difficulty = train_input[idx].select("difficulty").unique().to_numpy().squeeze(1)[0]
        outputs = {
            "id": sample_idx,
            "input": input_example,
            "difficulty": difficulty,
            "generations": [],
        }
        for i in range(num_generations//num_sequences):

            

            config = {
                "temperature": 0.7,
                "max_length": 2048,
                "top_p": 0.95,
                "top_k": 50,
                "num_return_sequences": num_sequences
            }

            prompt = f"Please write a Python program \nQUESTION: \n{input_example} \n ANSWER: \n."
            output = llm.run(prompt=prompt, input=input_example, config_params=config)

            for res in output:
                outputs["generations"].append(res)

        json.dump(outputs, open(f"outputs/raw_outputs_{sample_idx}.json", "w", encoding="utf-8"))

if __name__ == "__main__":
    PATH = "../../data/TACO/processed"
    train_input = pl.read_ipc(f"{PATH}/train.feather").group_by(pl.col("difficulty")).head(20)
    run_taco(200, 20, train_input)