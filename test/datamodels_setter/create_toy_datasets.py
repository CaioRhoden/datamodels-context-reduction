import polars as pl

target = {
    "input": ["target_1", "target_2", "target_3", "target_4", "target_5", "target_6"],
    "output": ["output_1", "output_2", "output_3", "output_4", "output_5", "output_6"],
}

random = {
    "input": ["random_1", "random_2", "random_3", "random_4", "random_5", "random_6"],
    "output": ["output_1", "output_2", "output_3", "output_4", "output_5", "output_6"],
}

df_target = pl.DataFrame(target).with_row_index("index")
df_random = pl.DataFrame(random).with_row_index("index")

df_target.write_ipc("toy_target.feather")
df_random.write_ipc("toy_random.feather")

