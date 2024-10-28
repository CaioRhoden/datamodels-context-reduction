from src.llms import BaseLLM
from src.evaluator import BaseEvaluator

import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from langchain.prompts import PromptTemplate
import h5py
import json
import os
import pickle
import models.datamodels.datamodels.regression.write_dataset as write_dataset

@dataclass
class DatamodelConfig:

    k: int
    datamodels_path: str | None
    train_collections_idx: np.ndarray | None
    test_collections_idx: np.ndarray | None
    test_set: pd.DataFrame | None
    train_set: pd.DataFrame | None
    instructions: dict | None
    llm: BaseLLM | None
    evaluator: BaseEvaluator | None




class Datamodels:
    
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
                self.test_collections_idx = f["train_collection"][()]
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
        checkpoint: int = 10,
        input_column: str = "input",
        output_column: str = "output",
        optinal_output_column: str|None = "possible_outputs"
    
    ):

        if self.train_collections_idx is None:
            raise ValueError("Train collection index not loaded")
        
        pre_collection_dict = self._reset_pre_collection_dict(optinal_output_column)
        checkpoint_count = 0


        if end_idx is None:
            end_idx = len(self.train_collections_idx[start_idx:])

        for idx_row in range(start_idx, end_idx):

            print(f"Collection id: {idx_row}")
            checkpoint_count += 1

            ## Convert index to binary vector
            if type == "train":
                binary_idx = self._convert_idx_to_binary(self.train_collections_idx[idx_row], self.train_set)
            elif type == "test":
                binary_idx = self._convert_idx_to_binary(self.test_collections_idx[idx_row], self.train_set)

            ## Get the input output pairs and concatenate into a string
            for dev_idx in range(len(self.test_set)):
                prompt = self._fill_prompt_template(idx_row, dev_idx, input_column, output_column)
                result = self.llm.run(prompt)

                # Add element to pre collection dict

                pre_collection_dict = self._add_element_to_collection(pre_collection_dict, idx_row, dev_idx, binary_idx, result, self.test_set.iloc[dev_idx][output_column], self.test_set.iloc[dev_idx][optinal_output_column])
            



            ## Saving condition in checkpoint or end of indezes
            if checkpoint_count == checkpoint or idx_row == end_idx-1:

                df = pd.DataFrame(pre_collection_dict)
                print(f"Checkpoint {idx_row} saved")
                
                if type == "train":
                    df.to_pickle(f"{self.datamodels_path}/pre_collections/pre_collection_{idx_row}.pickle")
                elif type == "test":
                    df.to_pickle(f"{self.datamodels_path}/pre_collections/test_pre_collection_{idx_row}.pickle")


                pre_collection_dict = self._reset_pre_collection_dict(optinal_output_column)
                checkpoint_count = 0
            
            

    def create_collection(
        self,
        batch_name: str,
        device: str = "cuda:0",
        pre_collection_batch: pd.DataFrame = None,


    ):
        def extract_output(text):
            # Split the string and strip leading/trailing spaces and newlines
            return text.split("Model Output:\n ", 1)[-1].strip()
        
        
        
        evaluation = self.evaluator.evaluate(pre_collection_batch["true_output"].to_numpy(),  pre_collection_batch["predicted_output"].apply(extract_output).to_numpy(), pre_collection_batch["optinal_output"].to_numpy())
        pre_collection_batch["evaluation"] = evaluation
        pre_collection_batch["input"] =  pre_collection_batch["input"].apply(lambda x: np.array(x))
        collection = pre_collection_batch[["collection_idx","test_idx","input", "evaluation"]]
        collection.to_pickle(f"{self.collections_path}/{batch_name}.pickle")



        
        


    def train_datamodels(self):
        ## TODO: Implement Fast L1 model and see JAX Options
        ## Create .beton to be used for train datamodel
        # self._load_collections_from_path(self.datamodels_path)
        write_dataset(
            data_dir=f"{self.datamodels_path}",
            out_path=f"{self.datamodels_path}/estimations/",
            x_name="X_train",
            y_name="y_train",
            completed_name="X_train"

        )



    def _load_collections_from_path(self):

        input_samples_train = np.array([])
        results_train = np.array([])
        input_samples_test = np.array([])
        results_test = np.array([])

        for filename in os.listdir(f"{self.datamodels_path}/collections/"):
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



        

        np.save(f"{self.datamodels_path}/X_train.npy", input_samples_train)
        np.save(f"{self.datamodels_path}/Y_train.npy", results_train)
        np.save(f"{self.datamodels_path}/X_test.npy", input_samples_test)
        np.save(f"{self.datamodels_path}/Y_test.npy", results_test)
        




    

    
    def _add_element_to_collection(self, pre_collection_dict, collection_idx, test_idx, input, predicted_output, true_output, optinal_output=None):

        pre_collection_dict["collection_idx"].append(collection_idx)
        pre_collection_dict["test_idx"].append(test_idx)
        pre_collection_dict["input"].append(input)
        pre_collection_dict["predicted_output"].append(predicted_output)
        pre_collection_dict["true_output"].append(true_output)
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
        
    def _reset_pre_collection_dict(self, optional_column: str = None) -> dict:
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