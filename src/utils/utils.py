
import pandas as pd
def split_dev_set(path, saving_path, k_samples, task_column, seed=42):

    print("Reading data")
    
    import pandas as pd
    df = pd.read_csv(path).drop(columns=['Unnamed: 0'])

    dev_set =  df.groupby(task_column).apply(lambda x: x.sample(n=min(len(x), k_samples), random_state=seed))
    idxs = [i[1] for i in dev_set.index.values]

    dev_set = dev_set.reset_index(drop=True)

    train_set = df.drop(idxs).reset_index(drop=True)

    print("Saving dev set")
    dev_set.to_csv(f"{saving_path}/dev_set.csv", index=False)

    print("Saving train set")
    train_set.to_csv(f"{saving_path}/train_set.csv", index=False)

    return train_set, dev_set

def subset_df(df,  k_samples, task_column, seed=42) -> pd.DataFrame:
    
    sampled_df = df.groupby(task_column).apply(lambda x: x.sample(n=k_samples, random_state=seed)).reset_index(drop=True)

    return sampled_df