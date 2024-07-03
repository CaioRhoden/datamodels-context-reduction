
def calculate_distances(is_class_specific: bool):
    import pandas as pd
    from scipy.spatial.distance import cosine
    import numpy as np


    train_data = pd.read_pickle("data/pool.pickle").reset_index()
    test_data = pd.read_pickle("data/test.pickle").reset_index()

    distances = []
    for i in range(0, len(test_data)):
        dist_i = []
        print(f"{is_class_specific}, {i} of {len(test_data)}")
        for j in range(0, len(train_data)):

            if is_class_specific:
                if train_data.loc[j]["task"] != test_data.loc[i]["task"]:
                    emb_train = train_data.loc[j]["input_encoding"]
                    emb_test = test_data.loc[i]["input_encoding"]
                    dist_i.append(cosine(emb_test, emb_train))
            else:
                emb_train = train_data.loc[j]["input_encoding"]
                emb_test = test_data.loc[i]["input_encoding"]
                dist_i.append(cosine(emb_test, emb_train))

        distances.append(np.array(dist_i))

    test_data["distances"] = distances

    if is_class_specific:
        test_data.to_pickle("data/class_distances.pickle")
    else:
        test_data.to_pickle("data/distances.pickle")

print("Starting first")
calculate_distances(is_class_specific=True)

print("Starting second")
calculate_distances(is_class_specific=False)