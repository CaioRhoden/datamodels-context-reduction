from dmcr.datamodels.config import MemMapConfig, DatamodelConfig, LogConfig
from dmcr.evaluators import BaseEvaluator
from dmcr.models import BaseLLM, GenericInstructModelHF

import pandas as pd
import polars as pl
import numpy as np
import torch

from langchain.prompts import PromptTemplate
from dmcr.datamodels.models import LinearRegressor
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
from torch.utils.data import Subset, random_split






class DatamodelsNQPipeline:
    
    def __init__(self, config: DatamodelConfig, test_flag=False) -> None:

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



        

        if not test_flag:
            self._verify_repo_structure()
            self._set_collections_index()
            self._set_dataframes()


        
    
    def _set_collections_index(self):


        """
        Loads the train and test collections index from the datamodels path.

        If the train or test collections index is not already loaded, it will be loaded from the datamodels path.
        The loaded index is stored in the respective attribute of the class.

        Parameters:
            None

        Returns:
            None
        """
        with h5py.File(f"{self.datamodels_path}/train_collection.h5", "r") as f:
            self.train_collections_idx = f["train_collection"][()]
        print("Loaded train collection index")

        with h5py.File(f"{self.datamodels_path}/test_collection.h5", "r") as f:
            self.test_collections_idx = f["test_collection"][()]
        print("Loaded test collection index")
        
    def _verify_repo_structure(self):
        if not os.path.exists(f"{self.datamodels_path}/train_set.csv"):
            raise Exception("Train set not found in datamodels path")
        if not os.path.exists(f"{self.datamodels_path}/test_set.csv"):
            raise Exception("Test set not found in datamodels path")
        if not os.path.exists(f"{self.datamodels_path}/train_collection.h5"):
            raise Exception("Train collection not found in datamodels path")
        if not os.path.exists(f"{self.datamodels_path}/test_collection.h5"):
            raise Exception("Test collection not found in datamodels path")
        
    def _set_dataframes(self):
        """
        Loads the test set if it has not been loaded yet.

        Returns:
            pd.DataFrame: The test set.
        """

        self.train_set = pd.read_csv(f"{self.datamodels_path}/train_set.csv")
        print("Loaded train set")

        self.test_set = pd.read_csv(f"{self.datamodels_path}/test_set.csv")
        print("Loaded test set")




    def create_pre_collection(
        self,
        mode: str,
        instruction: dict | str,
        llm: BaseLLM,
        start_idx: int = 0,
        end_idx: int = -1,
        checkpoint: int = 50,
        title_column: str = "title",
        text_column: str = "text",
        question_column: str = "question",
        output_column: str = "output",
        log: bool = False,
        log_config: LogConfig | None = None,
        model_configs: dict | None = None,
        
        
        optional_output_column: str | None = None
    
    ):

        if self.train_collections_idx is None:
            raise ValueError("Train collection index not loaded")
        
        pre_collection_dict = self._reset_pre_collection_dict(optional_output_column)
        checkpoint_count = 0

        if self.train_collections_idx is None or self.test_collections_idx is None:
            raise Exception("Combinations for pre-collections creation not present")



        if end_idx == -1:
            end_idx = len(self.train_collections_idx[start_idx:])

        



        for idx_row in range(start_idx, end_idx):

            start_time = datetime.datetime.now()


            checkpoint_count += 1

            ## Convert index to binary vector
            if mode == "train":
                binary_idx = self._convert_idx_to_binary(self.train_collections_idx[idx_row], self.train_set)
            elif mode == "test":
                print(self.test_collections_idx.shape)
                binary_idx = self._convert_idx_to_binary(self.test_collections_idx[idx_row], self.train_set)

            ## Get the input output pairs and concatenate into a string
            for dev_idx in range(len(self.test_set)):
                prompt = self._fill_prompt_template(idx_row, dev_idx, title_column, text_column, question_column)
                if isinstance(llm, GenericInstructModelHF):
                    result = llm.run(prompt, instruction=str(instruction), config_params=model_configs)[0]["generated_text"]


                else:
                    result = llm.run(prompt)


                # Add element to pre collection dict
                if optional_output_column is not None:
                    pre_collection_dict = self._add_element_to_collection(pre_collection_dict, idx_row, dev_idx, binary_idx, result, self.test_set.iloc[dev_idx][output_column], self.test_set.iloc[dev_idx][optional_output_column])

                else:
                    pre_collection_dict = self._add_element_to_collection(pre_collection_dict, idx_row, dev_idx, binary_idx, result, self.test_set.iloc[dev_idx][output_column])



            ## Saving condition in checkpoint or end of indezes
            if checkpoint_count == checkpoint or idx_row == end_idx-1:

                print(datetime.datetime.now())

                df = pl.DataFrame(pre_collection_dict)
                print(f"Checkpoint {idx_row} saved")
                
                if not os.path.exists(f"{self.datamodels_path}/pre_collections"):
                    os.mkdir(f"{self.datamodels_path}/pre_collections")
                
                if not os.path.exists(f"{self.datamodels_path}/pre_collections/train"):
                    os.mkdir(f"{self.datamodels_path}/pre_collections/train")

                if not os.path.exists(f"{self.datamodels_path}/pre_collections/test"):
                    os.mkdir(f"{self.datamodels_path}/pre_collections/test")

                if mode == "train":
                    df.write_ipc(f"{self.datamodels_path}/pre_collections/train/pre_collection_{idx_row}.feather")
                elif mode == "test":
                    df.write_ipc(f"{self.datamodels_path}/pre_collections/test/pre_collection_{idx_row}.feather")


                pre_collection_dict = self._reset_pre_collection_dict(optional_output_column)
                checkpoint_count = 0

            if log:

                print(log_config.id)

                if log_config is None:
                    raise Exception("Please provide a log configuration.")
                
                try:
                    wandb.init( 
                        project = log_config.project, 
                        dir = log_config.dir, 
                        id = f"{idx_row}_{log_config.id}", 
                        name = f"{idx_row}_{log_config.name}",
                        config = log_config.config,
                        tags = log_config.tags
                    )
                    wandb.log({"idx": idx_row, "end_time": str(datetime.datetime.now()), "full_duration": str((datetime.datetime.now() - start_time).total_seconds())})
                    wandb.finish()

                except:
                    raise Exception("Wandb not initialized, please check your log configuration")

            
            

    def create_collection(
        self,
        evaluator: BaseEvaluator,
        collection_name: str,
        mode: str = "train",
        device: str = "cuda:0",

    ):
        
        ## Verfiy if pre-collections exist
        
        pre_collections_path = f"{self.datamodels_path}/pre_collections/{mode}"
        assert os.path.exists(pre_collections_path)       

        ## Read all pre-collections
        feather_files = Path(pre_collections_path).glob('*.feather')
        dfs = [pl.read_ipc(file) for file in sorted(feather_files)]
        pre_collections = pl.concat(dfs, how='vertical')

        ## Verify if pre-collections are not empty
        assert len(pre_collections) > 0
        
        try:
            optional_output =  pre_collections["optinal_output"].to_numpy()
        except:
            optional_output = None
        
        ## Evaluate the pre_collections and add them to the dataframe
        evaluation = evaluator.evaluate(pre_collections["true_output"].to_numpy(),  pre_collections["predicted_output"].to_numpy(), optional_output)
        pre_collections  = pre_collections.with_columns(pl.Series("evaluation", evaluation).cast(pl.Float64).alias("evaluation"))
        collection = pre_collections[["collection_idx","test_idx","input", "evaluation"]]
        

        ## Guarantees the necessary directories
        if not os.path.exists(f"{self.datamodels_path}/collections"):
            os.mkdir(f"{self.datamodels_path}/collections")
        
        if not os.path.exists(f"{self.datamodels_path}/collections/train"):
            os.mkdir(f"{self.datamodels_path}/collections/train")

        if not os.path.exists(f"{self.datamodels_path}/collections/test"):
            os.mkdir(f"{self.datamodels_path}/collections/test")

        ## Save file
        collection.write_ipc(f"{self.datamodels_path}/collections/{mode}/{collection_name}.feather")

    


    def train_datamodels(
            self,
            epochs: int = 1,
            train_batches: int = 1,
            val_batches: int = 1,
            val_size: float = 0.1,
            lr: float = 0.0001,
            random_seed: int = 42,
            patience: int = 5,
            subset: int = 40000,
            log: bool = False,
            log_epochs: int = 10,
            run_id: str = "generic",
            device: str = "cuda:0",
                         
    ):


            torch.manual_seed(random_seed)

            ## Initialize place to save weights and bias
            stacked_weights = torch.tensor([], device=device)
            stacked_bias = torch.tensor([], device=device)
            

            for idx in range(self.num_models):
                print(f"Idx: {idx}")

                
                dataset = torch.load(f"{self.datamodels_path}/datasets/train/dt_{idx}.pt")

                ## Random Sampling
                random_indices = torch.randperm(len(dataset))[:subset].tolist()
                dataset = Subset(dataset, random_indices)
                train_dt, val_dt = random_split(dataset, [1 - val_size, val_size], generator=torch.Generator().manual_seed(random_seed)) 

                train = torch.utils.data.DataLoader(train_dt, batch_size=train_batches, shuffle=True)
                val = torch.utils.data.DataLoader(val_dt, batch_size=val_batches, shuffle=True)


                
                ## Model Creation
                model = LinearRegressor(len(dataset[0][0]), 1)
                criterion = nn.MSELoss()
                print(model.parameters())
                optimizer = optim.SGD(model.parameters(), lr=lr)

                ## Earlt Stopping Config
                best_mse = float('inf')
                early_stopping_counter = 0

                if log:
                    wandb.init(project="bbh_reduced_sample_training", 
                            dir = f"logs/{run_id}",
                            id = f"{run_id}_{idx}",
                            name = f"{run_id}_{idx}",
                            config = {
                                "epochs": epochs,
                                "train_batches": train_batches,
                                "val_batches": val_batches,
                                "val_size": val_size,
                                "lr": lr,
                                "random_seed": random_seed,
                                "patience": patience,
                                "subset": subset, 
                                "model": str(model),
                                "criterion": str(criterion),
                                "optimizer": str(optimizer),
                                "endtime": datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                                },
                                tags=[f"task_{idx // 5}", run_id],


                    )




                for epoch in range(epochs):

                    # Shuffle indexes
                    

                
                    total_loss = 0
                    total_mse = 0

                    for x_train_batch, y_train_batch in train:

                        x_train_batch, y_train_batch = x_train_batch.to(device), y_train_batch.to(device)

                        # Apply the mask to the weights
                        y_pred = model(x_train_batch) # Add batch dimension

                        # Compute loss
                        loss = criterion(y_pred, y_train_batch).to(device)  # Add batch dimension to target

                        # Backward pass and optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()
                        
                
                    for x_val_batch, y_val_batch in val:
                        total_mse += model.evaluate(x_val_batch.to(device), y_val_batch.to(device))
                    
                    mean_loss = round(total_loss / len(train), 4)
                    mean_mse = round(total_mse / len(val), 4)

                    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {mean_loss:.4f}, Val MSE: {mean_mse:.4f}')

                    if log and epoch % log_epochs == 0:
                        wandb.log({"epoch": epoch, "mean_loss": mean_loss, "mean_metric": mean_mse})

                    if mean_mse < best_mse:
                        best_mse = mean_mse
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1

                    if early_stopping_counter >= patience:
                        if log:
                            wandb.log({"early_stopping_counter": epoch, "epoch": epoch, "mean_loss": mean_loss, "mean_metric": mean_mse})
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

                stacked_weights = torch.concat((stacked_weights, model.get_weights()), dim=0)
                stacked_bias = torch.concat((stacked_bias, model.get_bias()), dim=0)

            
                if log:
                    wandb.finish()

            torch.save(stacked_weights, f"estimations/{run_id}/weights.pt")
            torch.save(stacked_bias, f"estimations/{run_id}/bias.pt")


            



        

        


        



    def load_collections_from_path(self, test_flag: bool = False):

        input_samples_train = np.array([])
        results_train = np.array([])

        if test_flag:
            input_samples_test = np.array([])
            results_test = np.array([])

        print("Collections under processing")
        for filename in os.listdir(f"{self.datamodels_path}/collections/train/"):
            file = os.path.join(f"{self.datamodels_path}/collections/train/", filename)

            with open(file, 'rb') as f:
                collection = pd.read_feather(f)

                # collection_input = collection["input"].to_numpy()
                # input_samples_test = np.concatenate((input_samples_test, collection_input))

                # collection_result = collection["evaluation"].to_numpy()
                # results_test = np.concatenate((results_test, collection_result))

                # elif not filename.startswith("test"):

                collection_input = collection["input"].to_numpy()
                input_samples_train = np.concatenate((input_samples_train, collection_input))

                collection_result = collection["evaluation"].to_numpy()
                results_train = np.concatenate((results_train, collection_result))


        ## Convert mask arrays to bool
        input_samples_train = np.array([list(arr) for arr in input_samples_train], dtype=np.float32)





        ## Reshape for input
        X_train = input_samples_train.reshape(self.num_models, len(input_samples_train)//self.num_models, input_samples_train.shape[1])
        y_train = results_train.reshape(self.num_models, len(results_train)//self.num_models)
        

        ## Create torch datasets and save them
        for i in range(self.num_models):
            train = torch.tensor(X_train[i], dtype=torch.float32)
            test = torch.tensor(y_train[i], dtype=torch.float32)
            train_dataset = TensorDataset(train, test)
            torch.save(train_dataset, f"{self.datamodels_path}/datasets/train/dt_{i}.pt")

        if test_flag:

            print("Saving test tensors")
            input_samples_test = np.array([list(arr) for arr in input_samples_test], dtype=np.float32)
            X_test = input_samples_test.reshape(self.num_models, len(input_samples_test)//self.num_models, input_samples_test.shape[1])
            y_test = results_test.reshape(self.num_models, len(results_test)//self.num_models)

            for i in range(self.num_models):
                train = torch.tensor(X_train[i], dtype=torch.float32)
                test = torch.tensor(y_train[i], dtype=torch.float32)
                train_dataset = TensorDataset(train, test)
                torch.save(train_dataset, f"{self.datamodels_path}/datasets/test/dt_{i}.pt")


    
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
            title_column: str,
            text_column: str,
            question_column: str      ) -> str:
        
        template = """
            Documents:
            {context}

            Question: {input}\nAnswer: 
        """

        context = ""
        count = 0
        for idx in self.train_collections_idx[idx_row]:
            title = self.train_set.loc[idx][title_column]
            text = self.train_set.loc[idx][text_column]
            context += f"Document[{count}](Title: {title}){text}\n\n"
            count += 1

        
        input = self.test_set.loc[idx_test][question_column]

        prompt = PromptTemplate.from_template(template).format(context=context, input=input)

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