import polars as pl
from torch.utils.data import DataLoader, TensorDataset

def create_colletion_dataloaders(
    df: pl.DataFrame,
    num_tasks: int,
    proportion: range,
    batch_size: int,
    shuffle: bool = True,  
):
    
    """
    Creates a dictionary of data loaders for the given df, with each task 
    filtered by the proportion of samples in the given range.

    Parameters
    ----------
    df : polars.DataFrame
        The dataframe containing the data
    num_tasks : int
        The number of tasks
    proportion : range
        The proportion of samples in the given range for each task
    batch_size : int
        The batch size for the data loader
    shuffle : bool, optional
        Whether to shuffle the data loader, by default True

    Returns
    -------
    dict[str, torch.utils.data.DataLoader]
        A dictionary of data loaders, with each key being the task name and each
        value being the data loader for that task
    """
    dataloaders = {}

    for i in range(num_tasks):
        task = i
        _df = (
            df
            .clone()
            .filter(pl.col(f"count_task_{task}").is_in(proportion))
            .select(["collection_idx"])
        )

        dl = DataLoader(
            TensorDataset(_df.to_torch()),
            batch_size=batch_size,
            shuffle=shuffle
        )

        dataloaders[f"task_{task}"] = dl

    return dataloaders

def create_test_dataloader(
    df: pl.DataFrame,
    task: str,
    batch_size: int,
    shuffle: bool = True,
):

    """
    Creates a data loader for a specific task from the given dataframe.

    Parameters
    ----------
    df : polars.DataFrame
        The dataframe containing the data
    task : str
        The task for which the data loader is created
    batch_size : int
        The batch size for the data loader
    shuffle : bool, optional
        Whether to shuffle the data loader, by default True

    Returns
    -------
    torch.utils.data.DataLoader
        A data loader for the specified task
    """

    _df = (
        df
        .clone()
        .with_columns(pl.arange(0, pl.len()).alias("idx"))
        .filter(pl.col(f"task").eq(task))
        .select(["idx"])
    )

    dl = DataLoader(
            TensorDataset(_df.to_torch()),
            batch_size=batch_size,
            shuffle=shuffle
        )


    return dl