
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import evaluate
import ast
import re
import os

rouge = evaluate.load('rouge')

def extract_pred_output(pred):

    pattern = r"Output:\s*(.*)"
    match = re.search(pattern, pred)
    if match:
        prediction = match.group(1).strip()
    else:
        prediction  = ""

    return prediction

def calculate_rouge(pred, groundtruth, possible_outputs):
    scores = []

    # Get prediction


    prediction = [pred]

    #Get references:
    if possible_outputs is str:
        references = ast.literal_eval(possible_outputs)
        references.append(groundtruth)
    
    else:
        references = [groundtruth]
    

    # Calculate ROUGE-1 score between y_pred and each element in possible_output
    for ref in references:
        score = rouge.compute(predictions=prediction, references=[ref])['rougeL']
        scores.append(score)
# Return the highest ROUGE-1 score
    return max(scores)

def append_data(data, model, method, k, experiment_type, input, output, predicted_output, possible_outputs, rouge1, task):
        data["model"].append(model)
        data["method"].append(method)
        data["k"].append(k)
        data["experiment_type"].append(experiment_type)
        data["input"].append(input)
        data["output"].append(output)
        data["predicted_output"].append(predicted_output)
        data["possible_outputs"].append(possible_outputs)
        data["rouge1"].append(rouge1)
        data["task"].append(task)
        return data

def main(dir, out_dir, input_file, output_file):

    pattern = r"results_(.*?)_(.*?)_(.*?)_(.*?)\.csv"
    match = re.match(pattern, input_file)
    if match:
        model, method, k_value, experiment_type = match.groups()
    else:
        raise ValueError(f"Invalid filename: {input_file}")
    

    df = pd.read_csv(os.path.join(dir, input_file))

    data = {
        "model": [],
        "method": [],
        "k": [],
        "experiment_type": [],
        "task": [],
        "input": [],
        "output": [],
        "predicted_output": [],
        "possible_outputs": [],
        "rouge1":  [],

    }

    df["predicted_output"] = df["predicted_output"].apply(extract_pred_output)
    df['rouge1'] = df.apply(lambda row: calculate_rouge(row['predicted_output'], row['output'], row['possible_outputs']), axis=1)
    for index, row in df.iterrows():
        input = row["input"]
        predicted_output = row["predicted_output"]
        output = row["output"]
        possible_outputs = row["possible_outputs"]
        rouge1 = row["rouge1"]
        task = row["task"]
        data = append_data(data, model, method, k_value, experiment_type, input, output, predicted_output, possible_outputs, rouge1, task)

    df = pd.DataFrame(data)
    df.to_csv(f"{out_dir+output_file}")


dir = "data/results/"
out_dr = "data/compiled/"
for filename in os.listdir(dir):
    print(filename)
    if filename.endswith(".csv"):
        main(dir, out_dr, filename, f"compiled__{filename}")