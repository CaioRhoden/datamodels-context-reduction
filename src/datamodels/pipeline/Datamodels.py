from src.datamodels.config import MemMapConfig, DatamodelConfig

import pandas as pd
import numpy as np
import torch

from langchain.prompts import PromptTemplate
from src.datamodels.models import LinearRegressor
import h5py
import json
import os
import pickle
from torch.utils.data import TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
import datetime
import wandb

from pathlib import Path





class DatamodelPipeline:
    
    def __init__(self, config: DatamodelConfig) -> None:

        """
        Initializes a new instance of the Datamodels class.

        Parameters:
            config (DatamodelConfig): The configuration for the datamodels.

        Returns:
            None

        TODO: Add customization for context instruction
        """
        self.k = config.k
        self.num_models = config.num_models
        self.datamodels_path = config.datamodels_path
        self.train_collections_idx = config.train_collections_idx
        self.test_collections_idx = config.test_collections_idx
        self.test_set = config.test_set
        self.train_set = config.train_set
        self.instructions = config.instructions
        self.llm = config.llm
        self.evaluator = config.evaluator

        
    
    def set_collections_index(self):


        """
        Loads the train and test collections index from the datamodels path.

        If the train or test collections index is not already loaded, it will be loaded from the datamodels path.
        The loaded index is stored in the respective attribute of the class.

        Parameters:
            None

        Returns:
            None
        """
        if self.train_collections_idx is None and self.datamodels_path is not None:
            with h5py.File(f"{self.datamodels_path}/train_collection.h5", "r") as f:
                self.train_collections_idx = f["train_collection"][()]
            print("Loaded train collection index")

        if self.test_collections_idx is None and self.datamodels_path is not None:
            with h5py.File(f"{self.datamodels_path}/test_collection.h5", "r") as f:
                self.test_collections_idx = f["test_collection"][()]
            print("Loaded test collection index")
        

        
    def set_dataframes(self):
        """
        Loads the test set if it has not been loaded yet.

        Returns:
            pd.DataFrame: The test set.
        """

        if self.train_set is None and self.datamodels_path is not None:
            self.train_set = pd.read_csv(f"{self.datamodels_path}/train_set.csv")
            print("Loaded train set")

        if self.test_set is None and self.datamodels_path is not None:
            self.test_set = pd.read_csv(f"{self.datamodels_path}/test_set.csv")
            print("Loaded test set")

    
    def set_instructions_from_path(self):

        with open(f"{self.datamodels_path}/instructions.json", "r") as f:
            self.instructions = json.load(f)


    def create_pre_collection(
        self,
        device: str = "cuda:0",
        type: str = "train",
        start_idx: int = 0,
        end_idx: int = 1,
        checkpoint: int = 50,
        input_column: str = "input",
        output_column: str = "output",
        log: bool = False,
        optional_output_column: str | None = None
    
    ):

        if self.train_collections_idx is None:
            raise ValueError("Train collection index not loaded")
        
        pre_collection_dict = self._reset_pre_collection_dict(optional_output_column)
        checkpoint_count = 0


        if end_idx is None:
            end_idx = len(self.train_collections_idx[start_idx:])

        



        for idx_row in range(start_idx, end_idx):

            start_time = datetime.datetime.now()


            checkpoint_count += 1

            ## Convert index to binary vector
            if type == "train":
                binary_idx = self._convert_idx_to_binary(self.train_collections_idx[idx_row], self.train_set)
            elif type == "test":
                binary_idx = self._convert_idx_to_binary(self.test_collections_idx[idx_row], self.test_set)

            ## Get the input output pairs and concatenate into a string
            for dev_idx in range(len(self.test_set)):
                prompt = self._fill_prompt_template(idx_row, dev_idx, input_column, output_column)
                result = self.llm.run(prompt)


                # Add element to pre collection dict
                if optional_output_column is not None:
                    pre_collection_dict = self._add_element_to_collection(pre_collection_dict, idx_row, dev_idx, binary_idx, result, self.test_set.iloc[dev_idx][output_column], self.test_set.iloc[dev_idx][optional_output_column])
                else:
                    pre_collection_dict = self._add_element_to_collection(pre_collection_dict, idx_row, dev_idx, binary_idx, result, self.test_set.iloc[dev_idx][output_column])



            ## Saving condition in checkpoint or end of indezes
            if checkpoint_count == checkpoint or idx_row == end_idx-1:

                print(datetime.datetime.now())

                df = pd.DataFrame(pre_collection_dict)
                print(f"Checkpoint {idx_row} saved")
                
                if type == "train":
                    df.to_feather(f"{self.datamodels_path}/pre_collections/pre_collection_{idx_row}.feather")
                elif type == "test":
                    df.to_feather(f"{self.datamodels_path}/pre_collections/test_pre_collection_{idx_row}.feather")


                pre_collection_dict = self._reset_pre_collection_dict(optional_output_column)
                checkpoint_count = 0

            if log:
                wandb.init( 
                    project="datamodels_pre_collections", 
                    dir="log/pre_collection/24_12_2024", 
                    id=f"{idx_row}_bbh_dl_28", 
                    name=f"{idx_row}_bbh_dl_28",
                    config={
                        "k": self.k,
                        "num_models": self.num_models,
                        "llm": "Llama3_1_8B-Instruct",
                        "gpu": "NVIDIA A100",
                    }, 
                )
                wandb.log({"idx": idx_row, "end_time": str(datetime.datetime.now()), "full_duration": str((datetime.datetime.now() - start_time).total_seconds())})
                wandb.finish()

            
            

    def create_collection(
        self,
        batch_name: str,
        device: str = "cuda:0",
        pre_collection_batch: pd.DataFrame = None,


    ):
        
        
        
        evaluation = self.evaluator.evaluate(pre_collection_batch["true_output"].to_numpy(),  pre_collection_batch["predicted_output"].to_numpy(), pre_collection_batch["optinal_output"].to_numpy())
        pre_collection_batch["evaluation"] = evaluation
        pre_collection_batch["input"] =  pre_collection_batch["input"].apply(lambda x: np.array(x))
        collection = pre_collection_batch[["collection_idx","test_idx","input", "evaluation"]]
        collection.to_pickle(f"{self.datamodels_path}/collections/{batch_name}.pickle")

    


    def train_datamodels(
            self,
            epochs: int = 1,
            batch_size: int = 100,
            val_split: float = 0.1,
            lr: float = 0.01,
            random_seed: int = 42,
            patience: int = 5,
            device: str = "cuda:0",
                         
        ):

        self._load_collections_from_path()

        train_dataset = torch.load(f"{self.datamodels_path}/datasets/train_dataset.pt")
        test_dataset = torch.load(f"{self.datamodels_path}/datasets/test_dataset.pt")

        stacked_weights = torch.zeros(len(train_dataset), len(train_dataset[0][0][0]))
        stacked_bias =  torch.zeros(len(train_dataset))
        


        for idx in range(0, len(train_dataset)):
            print(f"Idx: {idx}")
            
            dataset = train_dataset[idx]

            ### Create val dataset
            val_size = int(len(dataset) * val_split)
            train_size = len(dataset) - val_size
            
            
            ##Inititalize weights and bias
            weights = torch.randn(len(dataset[0][0]), requires_grad=True) # Randomly initialized weights
            bias = torch.randn(1, requires_grad=True)

            model = LinearRegressor(weights, bias)
            criterion = nn.MSELoss()
            optimizer = optim.SGD([weights, bias], lr=lr)

            best_mse = float('inf')
            early_stopping_counter = 0

            indices = torch.randperm(len(dataset[0]), generator=torch.Generator())
            dataset= (dataset[0][indices], dataset[1][indices])

            train = (dataset[0][:train_size], dataset[1][:train_size])
            val = (dataset[0][train_size:], dataset[1][train_size:])

            for epoch in range(epochs):

                # Shuffle indexes
                


                total_loss = 0
                total_mse = 0

                for collection in range(len(train[0])):
                    model.weights = weights
                    model.bias = bias
                    # Apply the mask to the weights
                    output = model.forward(train[0][collection]) # Add batch dimension


                    # Compute loss
                    loss = criterion(output, dataset[1][collection].unsqueeze(0)).to(device)  # Add batch dimension to target

                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    
            
                for collection in range(len(val[0])):

                    x = val[0][collection]
                    y = val[1][collection]

                    total_mse += model.evaluate(x, y)
                
                mean_loss = round(total_loss / len(train[0]), 3)
                mean_mse = round(total_mse / len(val[0]), 3)

                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {mean_loss:.4f}, Val MSE: {mean_mse:.4f}')

                if mean_mse < best_mse:
                    best_mse = mean_mse
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            stacked_weights[idx] = weights
            stacked_bias[idx] = bias

        
        torch.save(stacked_weights, f"{self.datamodels_path}/estimations/weights.pt")
        torch.save(stacked_bias, f"{self.datamodels_path}/estimations/bias.pt")


            



        

        


        



    def _load_collections_from_path(self):

        input_samples_train = np.array([])
        results_train = np.array([])
        input_samples_test = np.array([])
        results_test = np.array([])

        for filename in os.listdir(f"{self.datamodels_path}/collections/"):
            print("Collections under processing")
            file = os.path.join(f"{self.datamodels_path}/collections/", filename)

            with open(file, 'rb') as f:
                collection = pickle.load(f)

                if filename.startswith("test"):

                    collection_input = collection["input"].to_numpy()
                    input_samples_test = np.concatenate((input_samples_test, collection_input))

                    collection_result = collection["evaluation"].to_numpy()
                    results_test = np.concatenate((results_test, collection_result))

                else:

                    collection_input = collection["input"].to_numpy()
                    input_samples_train = np.concatenate((input_samples_train, collection_input))

                    collection_result = collection["evaluation"].to_numpy()
                    results_train = np.concatenate((results_train, collection_result))

        ## Convert mask arrays to bool
        input_samples_train = np.array([list(arr) for arr in input_samples_train], dtype=np.bool_)
        input_samples_test = np.array([list(arr) for arr in input_samples_test], dtype=np.bool_)
        print(input_samples_train.shape, input_samples_train.dtype)



        ## Reshape for input
        X_train = input_samples_train.reshape(self.num_models, len(input_samples_train)//self.num_models, input_samples_train.shape[1])
        y_train = results_train.reshape(self.num_models, len(results_train)//self.num_models)
        X_test = input_samples_test.reshape(self.num_models, len(input_samples_test)//self.num_models, input_samples_test.shape[1])
        y_test = results_test.reshape(self.num_models, len(results_test)//self.num_models)

        ## Create torch datasets and save them
        X_train = torch.tensor(X_train, dtype=torch.bool)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.bool)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        print("Saving tensors")

        torch.save(train_dataset, f"{self.datamodels_path}/datasets/train_dataset.pt")
        torch.save(test_dataset, f"{self.datamodels_path}/datasets/test_dataset.pt")


    
    def _add_element_to_collection(self, pre_collection_dict, collection_idx, test_idx, input, predicted_output, true_output, optinal_output=None):

        pre_collection_dict["collection_idx"].append(collection_idx)
        pre_collection_dict["test_idx"].append(test_idx)
        pre_collection_dict["input"].append(input)
        pre_collection_dict["predicted_output"].append(predicted_output)
        pre_collection_dict["true_output"].append(true_output)
        if optinal_output is not None:
            print(optinal_output)
            pre_collection_dict["optinal_output"].append(optinal_output)

        return pre_collection_dict

    
    def _fill_prompt_template(
            self,
            idx_row: int,
            idx_test: int,
            input_column: str = "input",
            output_column: str = "output",
            task_column: str = "task",
        ) -> str:
        
        template = """
            Fill the expected Output according to the instruction
            Intruction: {instruction}

            Examples:
            {context}

            User Input:
            {input}

            Model Output:
        """

        context = ""
        for idx in self.train_collections_idx[idx_row]:
            input = self.train_set.loc[idx][input_column]
            output = self.train_set.loc[idx][output_column]
            context += f"Input: {input} \nOutput: {output}\n"

        
        input = self.test_set.loc[idx_test][input_column]

        instruction = self.instructions[self.test_set.loc[idx_test][task_column]]
        prompt = PromptTemplate.from_template(template).format(instruction=instruction, context=context, input=input)

        return prompt


    def _convert_idx_to_binary(self, arr: np.ndarray, df: pd.DataFrame = None) -> np.ndarray:
        
        """
        Convert an array of indices into a binary numpy array of the same length as a given DataFrame.
        
        Parameters
        ----------
        arr : np.ndarray
            The array of indices to convert.
        df : pd.DataFrame, optional
            The DataFrame to use to determine the length of the output array.
            Defaults to None, in which case the length of the output array is the same as the length of the input array.
        
        Returns
        -------
        np.ndarray
            The binary numpy array where the indices from the input array are 1 and the rest are 0.
        """
        indeces_df =  np.zeros(len(df), dtype=int)
        indeces_df[arr] = 1
        return indeces_df
        
    def _reset_pre_collection_dict(self, optional_column: str | None = None) -> dict:
        pre_collection_dict = {
            "collection_idx": [],
            "test_idx": [],
            "input": [],
            "predicted_output": [],
            "true_output": [],
        }

        if optional_column is not None:
            pre_collection_dict["optinal_output"] = []


        return pre_collection_dict