
import pandas as pd
import numpy as np
import torch
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale, StandardScaler

import seaborn as sns
def split_dev_set(df, saving_path, k_samples, task_column, prefix="", seed=42):

    print("Reading data")


    dev_set =  df.groupby(task_column).apply(lambda x: x.sample(n=min(len(x), k_samples), random_state=seed))
    idxs = [i[1] for i in dev_set.index.values]

    dev_set = dev_set.reset_index(drop=True)

    train_set = df.drop(idxs).reset_index(drop=True)

    print("Saving dev set")
    dev_set.to_csv(f"{saving_path}/{prefix}dev_set.csv", index=False)

    print("Saving train set")
    train_set.to_csv(f"{saving_path}/{prefix}train_set.csv", index=False)

    return train_set, dev_set

def subset_df(df,  k_samples, task_column, seed=42) -> pd.DataFrame:
    
    sampled_df = df.groupby(task_column).apply(lambda x: x.sample(n=k_samples, random_state=seed)).reset_index(drop=True)

    return sampled_df




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
    top_samples = np.flip(np.argsort(df.loc[sample_idx]["weights"]))[:top_k]


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


## Individual 

def compare_i_most_high_samples(df: pd.DataFrame, train_df: pd.DataFrame, sample_idx: int, i: int):
    top_samples = np.flip(np.argsort(df.loc[sample_idx]["weights"]))[:i]

    sample_input = df.loc[sample_idx]["input"]
    sample_output = df.loc[sample_idx]["output"]


    i_most_input = train_df.loc[top_samples[i-1]]["input"]
    i_most_output = train_df.loc[top_samples[i-1]]["output"]

    print(f"Sample {sample_idx} input: {sample_input}")
    print(f"Sample {sample_idx} output: {sample_output}")
    print(f"{i} Most influential input: {i_most_input}")
    print(f"{i} Most influential output: {i_most_output}")


def plot_individual_heatmap_task_indication(
        weights: np.ndarray, 
        num_tasks: int,
        title: str,
        y_label: str,
        x_label: str,
        
        ):
    # weights = minmax_scale(weights)  # Normalize weights
    scaler = StandardScaler()
    weights = scaler.fit_transform(weights.reshape(-1, 1)).reshape(-1)
    weights_rsh = weights.reshape(num_tasks, len(weights)//num_tasks)
    plt.figure(figsize=(10, 5))  # Adjust figure size for better readability
    sns.heatmap(
        weights_rsh,
        cmap="coolwarm",      # Choose a color scheme
        cbar=True,           # Include a color bar
        annot=False,         # Optionally, set to True to display values
        xticklabels=True,    # Add x-axis labels
        yticklabels=True     # Add y-axis labels
    )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()




def plot_hist_weights(df: pd.DataFrame, top_k: int = 10, sample_idx: int = 0):
    indices = np.arange(top_k)
    values = sorted(df.loc[sample_idx]["weights"], reverse=True)[:top_k]

    plt.figure(figsize=(8, 6))
    plt.bar(indices, values, color='orange')
    plt.title(f"Distribution of Weights for Top {top_k} Samples from Sample {sample_idx}")
    plt.xlabel('Weight')



## Global 

def median_category_i_appearance(df: pd.DataFrame, i: int = 1):

    def _find_ith_appearance(row, i):
        occurrences = [idx for idx, value in enumerate(row["estimation_task"]) if value == row["task"]]
        return occurrences[i - 1] if len(occurrences) >= i else -1
    _df = df.copy()
    _df[["weights", "estimation_task"]] = _df.apply(_sort_weights_and_estimations, axis=1)
    _df[f"{i}th_appearance_index"] = _df.apply(lambda row: _find_ith_appearance(row, i), axis=1)

    # Calculate median values
    median_values = _df.groupby("task")[f"{i}th_appearance_index"].median().sort_values()

    # Prepare the data for Seaborn
    median_df = median_values.reset_index()
    median_df.columns = ["task", "median_index"]

    unique_tasks = sorted(median_df["task"].unique())
    palette = sns.color_palette("Set2", len(unique_tasks))
    task_colors = dict(zip(unique_tasks, palette))

    # Plot with Seaborn
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=median_df, 
        y="task", 
        x="median_index", 
        palette=task_colors, 
        edgecolor="black"
    )

    # Customize the plot
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", padding=3)

    # Customize the plot
    plt.title(f"Median {i}-th Appearance Index by Task")
    plt.xlabel("Median Index")
    plt.ylabel("Task")
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()


def median_same_task_first_quarter(df: pd.DataFrame):

    def count_same_task_first_quarter(row):
        train_size = len(row["weights"])
        quarter_size = train_size // 4  # Calculate the first quarter size
        count =  sum(
            a == row["task"]
            for a in row["estimation_task"][:quarter_size]
        )

        return (count/quarter_size)


    _df = df.copy()
    _df[["weights", "estimation_task"]] = _df.apply(_sort_weights_and_estimations, axis=1)
    _df["count_same_task_first_quarter"] = _df.apply(lambda row: count_same_task_first_quarter(row), axis=1)

    # Calculate median values
    median_values = _df.groupby("task")["count_same_task_first_quarter"].median().sort_values()

    # Prepare the data for Seaborn
    median_df = median_values.reset_index()
    median_df.columns = ["task", "median_index"]

    unique_tasks = sorted(median_df["task"].unique())
    palette = sns.color_palette("Set2", len(unique_tasks))
    task_colors = dict(zip(unique_tasks, palette))

    # Plot with Seaborn
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=median_df, 
        y="task", 
        x="median_index", 
        palette=task_colors, 
        edgecolor="black"
    )

    # Customize the plot
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", padding=3)

    # Customize the plot
    plt.title(f"Median Same Task First Quarter Index by Task")
    plt.xlabel("Median Index")
    plt.ylabel("Task")
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_barh(df: pd.DataFrame, x_axis: str, y_axis: str, title: str):

    plot_df = df.groupby(x_axis)[y_axis].sum().reset_index()

    unique_tasks = sorted(plot_df[y_axis].unique())
    palette = sns.color_palette("Set2", len(unique_tasks))
    task_colors = dict(zip(unique_tasks, palette))


    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=plot_df, 
        y=y_axis, 
        x=x_axis, 
        palette=task_colors, 
        edgecolor="black"
    )

    # Customize the plot
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", label_type="edge", padding=3)

    # Customize the plot
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()



## Aux

def _sort_weights_and_estimations(row):
    weights, estimations = row["weights"], row["estimation_task"]
    # Sort indices of weights in descending order
    sorted_indices = np.argsort(weights)[::-1]
    # Reorder both weights and estimations based on sorted indices
    sorted_weights = [weights[i] for i in sorted_indices]
    sorted_estimations = [estimations[i] for i in sorted_indices]
    return pd.Series({"weights": sorted_weights, "estimation_task": sorted_estimations})


