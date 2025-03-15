# %% [markdown]
# ## Baseline Runs - Analysis

# %%
import polars as pl
import torch
import numpy as np
import json
import re
## LLM
from src.taco_evaluator import compute, compute_1_pass_by_test
from datasets import load_from_disk
import datetime
import os

from typing import List, Dict, Any, Tuple


# %%
PATH  = "../../../data/TACO/processed"
train = pl.read_ipc(f"{PATH}/train.feather")
train_solutions = pl.read_ipc(f"{PATH}/train_solutions.feather")
train_dict = load_from_disk("../../../data/TACO/train.hf")

# %%
def parse_generation(generations: list, id: int, path: str):
    
    gens = []
    for i in range(len(generations)):

        code_blocks = re.findall(r'```python(.*?)```', generations[i]["generated_text"], re.DOTALL)
        extracted_code = "\n".join([block.strip() for block in code_blocks])
        gens.append(extracted_code)
    
    results = {
        "task_id": int(id),
        "output": gens
    }

    return results

# %%
def join_jsons(repo:str) -> List[Dict[str, Any]]:
    jsons = []
    for file in os.listdir(repo):
        if file.endswith(".json"):
            with open(f"{repo}/{file}", "r") as f:
                jsons.append(json.load(f))
    return jsons


# %% [markdown]
# ## EASY Examples Results

# %%
easy_results = join_jsons("../outputs_easy_experiment")

# %%
parsed = []
for r in easy_results:
    parsed.append(parse_generation(r["generations"], r["id"], "small_experiment"))

# %%
selected_ids = [r["id"] for r in easy_results]
partial_train = []
for i in selected_ids:
    partial_train.append(train_dict[i])

# %%
# json.dump(parsed, open("parsed_small_experiment.json", "w"))

# %%
# compute("parsed_small_experiment.json", partial_train, [1,10,100])

# %%
compute_1_pass_by_test("parsed_small_experiment.json", partial_train, file="taco_easy_1_pass.json")


