from src.llms import BaseLLM
from src.evaluator import BaseEvaluator

import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from langchain.prompts import PromptTemplate
import h5py
import json


@dataclass
class DatamodelConfig:

    k: int
    train_collections_idx_path: str | None
    train_collections_idx: np.ndarray | None
    test_collections_idx_path: str | None
    test_collections_idx: np.ndarray | None
    test_set: pd.DataFrame | None
    test_set_path: str | None
    train_set: pd.DataFrame | None
    train_set_path: str | None
    collections_path: str | None
    pre_collections_path: str | None
    instructions_path: str | None
    instructions: dict | None
    llm: BaseLLM | None
    evaluator: BaseEvaluator | None
    model: LinearRegression | None



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
        self.train_collections_idx_path = config.train_collections_idx_path
        self.train_collections_idx = config.train_collections_idx
        self.test_collections_idx_path = config.test_collections_idx_path
        self.test_collections_idx = config.test_collections_idx
        self.test_set = config.test_set
        self.test_set_path = config.test_set_path
        self.train_set = config.train_set
        self.train_set_path = config.train_set_path
        self.collections_path = config.collections_path
        self.pre_collections_path = config.pre_collections_path
        self.instructions_path = config.instructions_path
        self.instructions = config.instructions
        self.llm = config.llm
        self.evaluator = config.evaluator
        self.model = config.model
        

    

    def get_train_collection_index(self):
        """
        Loads the train collection index from the file if it has not been loaded yet.

        Returns:
            np.ndarray: The train collection index.
        """
        if self.train_collections_idx is None:
            with h5py.File(self.train_collections_idx_path, "r") as f:
                self.train_collections_idx = f["train_collection"][()]
            print("Loaded train collection index from ", self.train_collections_idx_path)
        return self.train_collections_idx
    
    def get_test_collection_index(self):
        """
        Loads the train collection index from the file if it has not been loaded yet.

        Returns:
            np.ndarray: The train collection index.
        """
        if self.test_collections_idx is None:
            with h5py.File(self.test_collections_idx_path, "r") as f:
                self.test_collections_idx = f["test_collection"][()]
            print("Loaded train collection index from ", self.test_collections_idx_path)
        return self.test_collections_idx



    def create_collection_index(self, path_collection):

        pass

    def get_test_set(self):
        """
        Loads the test set if it has not been loaded yet.

        Returns:
            pd.DataFrame: The test set.
        """
        if self.test_set is None:
            self.test_set = pd.read_csv(self.test_set_path)
            print("Loaded test set from ", self.test_set_path)
        return self.test_set

    def get_train_set(self):
        """
        Loads the train set if it has not been loaded yet.

        Returns:
            pd.DataFrame: The train set.
        """

        if self.train_set is None:
            self.train_set = pd.read_csv(self.train_set_path)
            print("Loaded train set from ", self.train_set_path)
        return self.train_set

    
    def set_instructions_from_path(self):

        with open(self.instructions_path, "r") as f:
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
        
        pre_collection_dict = {
            "collection_idx": [],
            "test_idx": [],
            "input": [],
            "predicted_output": [],
            "true_output": [],
        }

        if optinal_output_column is not None:
            pre_collection_dict["optinal_output"] = []


        if end_idx is None:
            end_idx = len(self.train_collections_idx[start_idx:])

        for idx_row in range(start_idx, end_idx):

            print(f"Collection id: {idx_row}")

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
            if (idx_row % checkpoint == 0 and idx_row > start_idx) or idx_row == end_idx-1:

                df = pd.DataFrame(pre_collection_dict)
                print(f"Checkpoint {idx_row} saved")
                
                if type == "train":
                    df.to_pickle(f"{self.pre_collections_path}/pre_collection_{idx_row}.pickle")
                elif type == "test":
                    df.to_pickle(f"{self.pre_collections_path}/test_pre_collection_{idx_row}.pickle")
                pre_collection_dict = {
                    "collection_idx": [],
                    "test_idx": [],
                    "input": [],
                    "predicted_output": [],
                    "true_output": [],
                }

                if optinal_output_column is not None:
                    pre_collection_dict["optinal_output"] = []

    def create_collection(
        self,
        batch_name: str,
        device: str = "cuda:0",
        pre_collection_batch: pd.DataFrame = None,


    ):
        def extract_output(text):
            # Split the string and strip leading/trailing spaces and newlines
            return text.split("Model Output:\n ", 1)[-1].strip()
        
        
        
        evaluation = self.evaluator.evaluate(pre_collection_batch["true_output"].to_numpy().to(device),  pre_collection_batch["predicted_output"].apply(extract_output).to_numpy().to(device))
        pre_collection_batch["evaluation"] = evaluation
        pre_collection_batch["input"] =  pre_collection_batch["input"].apply(lambda x: np.array(x))
        collection = pre_collection_batch[["collection_idx","test_idx","input", "evaluation"]]
        collection.to_pickle(f"{self.collections_path}/{batch_name}.pickle")



        
        


    def train_datamodels(self):
        ## TODO: Implement Fast L1 model and see JAX Options
        pass

    
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
        
