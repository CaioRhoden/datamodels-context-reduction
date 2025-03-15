# %% [markdown]
# # TACO Experiment: Cherry-picked context to one specific task
# 
# The goal here is to evaluate how the model will behave when passing the Solutions from an analog problem that were manually analyzed and selected  
# In this specific scenario we want to test it with a very specific group of tasks from the TACO benchmark

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import polars as pl
import torch
import numpy as np
import json
import re
## LLM
from src.llms import GenericIntructModelHF
from src.taco_evaluator import compute, compute_1_pass_by_test
from datasets import load_from_disk
import datetime

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

# %% [markdown]
# ## Load Datasets

# %%
PATH  = "../../../data/TACO/processed"
train = pl.read_ipc(f"{PATH}/train.feather")
train_solutions = pl.read_ipc(f"{PATH}/train_solutions.feather")
train_dict = load_from_disk("../../../data/TACO/train.hf")

# %% [markdown]
# 

# %%
def run_inference(prompt: str, path: str, num_returns = 10, max_length=2048):
    outputs = []
    llm = GenericIntructModelHF("../../../models/llms/Llama-3.2-3B-Instruct")
    for i in range(num_returns//10):
        print(f"Lopp {i}, {datetime.datetime.now()}")
        config = {
                    "temperature": 0.7,
                    "max_length": max_length,
                    "top_p": 0.95,
                    "num_return_sequences": 10
        }
        
        instruction = "You are a coding generation tool that will solve a problem using Python"

        output = llm.run(prompt=prompt, instruction=instruction, config_params=config)

        for res in output:
            outputs.append(res)
    
    llm.delete_model()
    json.dump(outputs, open(path, "w"))

# %%
def parse_generation(generations: list, id: int, path: str):
    
    gens = []
    for i in range(len(generations)):

        code_blocks = re.findall(r'```python(.*?)```', generations[i]["generated_text"], re.DOTALL)
        extracted_code = "\n".join([block.strip() for block in code_blocks])
        gens.append(extracted_code)
    
    results = [{
        "task_id": int(id),
        "output": gens
    }]

    json.dump(results, open(path, "w"))
        

# %% [markdown]
# ## Problem Selection
# 
# The category choosen it was "Geometry" in EASY difficulty  
# The criteria behind the choice is because there isn't a lot of examples of geometry, which facilitates to find samples specific from that scope.  
# The EASY difficulty is for validation purposes

# %%
# train.filter(pl.col("tags") == "Geometry").filter(pl.col("difficulty") == "EASY").count()

# %%
# selected_problem = train.filter(pl.col("tags") == "Geometry").filter(pl.col("difficulty") == "EASY").sample(1)
# print(selected_problem)
## ID = 14186 

selected_problem = train.filter(pl.col("tags") == "Geometry").filter(pl.col("difficulty") == "EASY").filter(pl.col("id") == 10237)
# print(selected_problem)

# %%
# train_solutions.filter(pl.col("id") == 10237).to_struct()

# %%
# print(selected_problem.select("input").to_dict()["input"][0])

# %% [markdown]
# ## Run Baseline - No Context

# %%
prompt_input = selected_problem.select("input").to_struct().to_pandas().iloc[0]["input"]
prompt = f"Please write a Python program \nQUESTION: \n{prompt_input} \n ANSWER: \n."
# run_inference(prompt_input, "no_context.json", num_returns=200)

# %%
# parse_generation(json.load(open("no_context.json")), 10237 , "no_context_parsed.json")

# %%
# compute("no_context_parsed.json", [train_dict[10237]], [1, 10, 100])

# %%
# json.load(open("taco_metrics.json"))

# %%
# compute_1_pass_by_test("no_context_parsed.json", [train_dict[10237]])

# %% [markdown]
# ## Select Context
# 
# Here we will try to get 4 solutions that are related to the problem above
# 

# %%
# df = train.filter(pl.col("tags") == "Geometry").filter(pl.col("difficulty") == "EASY").filter(pl.col("input").str.contains("class Solution")).select(["id", "input"])
# print(df.count())
# df.write_csv("pool.csv")

# %%
selected_ids = [21825, 10745, 1643, 4661]
# train.filter(pl.col("id") == 1643)

# %%
all_inputs  = train.filter(pl.col("id").is_in(selected_ids)).select("input").unique().to_dict()["input"]
all_solutions = train_solutions.filter(pl.col("id").is_in(selected_ids)).group_by(pl.col("id")).head(1).select("solution").unique().to_dict()["solution"]
question_input = selected_problem.select("input").to_dict()["input"][0]

# all_inputs

# %% [markdown]
# ## Full Prompt Run

# %%
context_prompt = "You will have to answer a programming quesiton in geometry, we will pass before some examples of questions and solutions\n"
for i in range(4):
    context_prompt += f"EXAMPLE QUESTION {i}:\n {all_inputs[i]}\n EXAMPLE SOLUTION {i}:\n {all_solutions[i]}\n"

full_prompt = f"Please write a Python program {context_prompt} \nQUESTION: \n{question_input} \n ANSWER: \n."

# %%
# run_inference(
#     prompt=full_prompt,
#     path = "full_prompt.json",
#     num_returns=200,
#     max_length=4096
# )

# %%
# parse_generation(json.load(open("full_prompt.json")), 10237 , "full_prompt_parsed.json")

# # %%
# compute("full_prompt_parsed.json", [train_dict[10237]], [1, 10, 100])

# # %%
# compute_1_pass_by_test("full_prompt_parsed.json", [train_dict[10237]])

# # %%
# json.load(open("full_prompt_metrics.json"))

# %% [markdown]
# ## Only Solutions Run

# %%
context_prompt = "You will have to answer a programming quesiton in geometry, we will pass before some examples of solutions for similar problems\n"
for i in range(4):
    context_prompt += f" EXAMPLE SOLUTION {i}:\n {all_solutions[i]}\n"

solutions_prompt = f"Please write a Python program {context_prompt} \nQUESTION: \n{question_input} \n ANSWER: \n."

# %%
run_inference(
    prompt=solutions_prompt,
    path = "solutions_prompt.json",
    num_returns=200,
    max_length=4096
)

# %%
# parse_generation(json.load(open("solutions_prompt.json")), 10237 , "solutions_parsed.json")

# # %%
# compute("solutions_parsed.json", [train_dict[10237]], [1, 10])

# # %%
# compute_1_pass_by_test("solutions_parsed.json", [train_dict[10237]])

# # %%
# json.load(open("solutions_prompt_metrics.json"))


