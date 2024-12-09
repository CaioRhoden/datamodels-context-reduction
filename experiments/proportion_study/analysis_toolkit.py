import pandas as pd
import numpy as np
import torch
from collections import Counter
from matplotlib import pyplot as plt


def show_tasks_by_sample(df: pd.DataFrame, sample_idx: int, top_k: int = 10):


    """
    Plot the top k tasks for a given sample in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the sample and task information.
    sample_idx : int
        The index of the sample to plot the tasks for.
    top_k : int, optional
        The number of top tasks to plot. Defaults to 10.

    Returns
    -------
    None
    """
    top_samples = np.argsort(df.loc[sample_idx]["weights"])[:top_k]


    tasks = []
    for i in top_samples:
        tasks.append(df["estimation_task"][sample_idx][i])

    counts = Counter(tasks)

    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)

# Unpack sorted items into keys and values
    items, values = zip(*sorted_items)

    # Create the bar plot
    plt.figure(figsize=(8, 6))
    plt.bar(items, values, color='blue')
    plt.title(f"Top {top_k} Tasks for Sample {sample_idx} from task {df.loc[sample_idx]['task']}")
    plt.xlabel('Task')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.show()
    
    return top_samples


def compare_i_most_high_samples(df: pd.DataFrame, train_df: pd.DataFrame, sample_idx: int, i: int):

    top_samples = np.argsort(df.loc[sample_idx]["weights"])[:i]

    sample_input = df.loc[sample_idx]["input"]
    sample_output = df.loc[sample_idx]["output"]

    i_most_input = train_df.loc[top_samples[i-1]]["input"]
    i_most_output = train_df.loc[top_samples[i-1]]["output"]

    print(f"Sample {sample_idx} input: {sample_input}")
    print(f"Sample {sample_idx} output: {sample_output}")
    print(f"{i} Most influential input: {i_most_input}")
    print(f"{i} Most influential output: {i_most_output}")



def plot_hist_weights(df: pd.DataFrame, top_k: int = 10, sample_idx: int = 0):
    indices = np.arange(top_k)
    values = sorted(df.loc[sample_idx]["weights"], reverse=True)[:top_k]

    plt.figure(figsize=(8, 6))
    plt.bar(indices, values, color='orange')
    plt.title(f"Distribution of Weights for Top {top_k} Samples from Sample {sample_idx}")
    plt.xlabel('Weight')


def median_category_i_appeareance(df: pd.DataFrame, i: int = 1):

    df[["weights", "estimation_task"]] = df.apply(_sort_weights_and_estimations, axis=1)
    df[f"{i}th_appearance_index"] = df.apply(lambda row: _find_ith_appearance(row, i), axis=1)

    median_values = df.groupby("task")[f"{i}th_appearance_index"].median()

    # Plot the horizontal bar chart
    plt.figure(figsize=(8, 5))
    median_values.sort_values().plot(kind="barh", color="lightsalmon", edgecolor="black")

    # Customize the plot
    plt.title(f"Median {i}-th Appearance Index by Task")
    plt.xlabel("Median Index")
    plt.ylabel("Task")
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()

    """
    Bar plot with the mean quantity of examples to appear the i-th sample of the same cateogry

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the sample and task information.
    n : int, optional
        The number of top tasks to plot. Defaults to 10.

    Returns
    -------
    None
    """

def _sort_weights_and_estimations(row):
    weights, estimations = row["weights"], row["estimation_task"]
    # Sort indices of weights in descending order
    sorted_indices = np.argsort(weights)[::-1]
    # Reorder both weights and estimations based on sorted indices
    sorted_weights = [weights[i] for i in sorted_indices]
    sorted_estimations = [estimations[i] for i in sorted_indices]
    return pd.Series({"weights": sorted_weights, "estimation_task": sorted_estimations})

def _find_ith_appearance(row, i):
    """
    Finds the appearence of the i-th sample of the same category

    Parameters:
    row: pd.Series
        The row of the dataframe containing the sample and task information.
    i: int
        The index of the sample to find the appearance of.  

    Returns:
    int
        The index of the appearance of the i-th sample of the same category
    """

    occurrences = [idx for idx, value in enumerate(row["estimation_task"]) if value == row["task"]]
    return occurrences[i - 1] if len(occurrences) >= i else -1


