import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import polars as pl
import torch
import numpy as np
import json

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


def run_taco(num_generations: int, train_input: pl.DataFrame):
    
    PATH = "../../data/TACO/processed"


    llm = Llama3_1_Instruct()

    for idx in range(len(train_input)):

        sample_idx = train_input[idx].select("sample_idx").unique().to_numpy().squeeze(1)[0]
        input_example = train_input[idx].select("input").unique().to_numpy().squeeze(1)[0]
        outputs = []
        for i in range(20//num_generations):

            config = {
                "temperature": 0.7,
                "max_length": 2048,
                "top_p": 0.95,
                "num_return_sequences": num_generations
            }

            prompt = f"Please write a Python program \nQUESTION: \n{input_example} \n ANSWER: \n."
            output = llm.run(prompt=prompt, input=input_example, config_params=config)

            for res in output:
                outputs.append(res)

        json.dump(outputs, open(f"raw_outputs_{sample_idx}.json", "w"))

if __name__ == "__main__":
    PATH = "../../data/TACO/processed"
    train_input = pl.read_ipc(f"{PATH}/train.feather").filter(pl.col("id").is_in([4,5]))
    run_taco(20, train_input)