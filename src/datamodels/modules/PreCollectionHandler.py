## internal packages
from typing import Any, Dict, List, Optional
from src.datamodels.modules.BaseModule import BaseModule
from src.datamodels.modules.ModuleConfig import PreCollectionsConfig
from src.datamodels.pipeline.BaseDatamodelsPipeline import BaseDatamodelsPipeline
from src.llms import BaseLLM

## external packages
import wandb
import datetime
import pandas as pd
from langchain.prompts import PromptTemplate
import numpy as np


class PreCollectionHandler(BaseModule):

    def __init__(self, handler_config: PreCollectionsConfig):
        """
        Initializes a new instance of the PreCollectionHandler class.

        Parameters:
            handler_config(PreCollectionsConfig): The configuration for the PreCollectionHandler.

        The following parameters are set from the handler_config:

        - log: indicates if the module should be logged in wandb
        - log_config: configuration for the logging (see LogConfig)
        - start_idx: the start index for the pre collection
        - end_idx: the end index for the pre collection
        - checkpoint: the checkpoint to save the pre collection
        - input_column: the column name for the input data in the dataframe
        - output_column: the column name for the output data in the dataframe
        - optional_output_column: the column name for the optional output data in the dataframe
        """
        super().__init__(handler_config)

        self.log = handler_config.log
        self.log_config = handler_config.log_config

        self.type = handler_config.type
        self.start_idx = handler_config.start_idx
        self.end_idx = handler_config.end_idx
        self.checkpoint = handler_config.checkpoint
        self.input_column = handler_config.input_column
        self.output_column = handler_config.output_column
        self.optional_output_column = handler_config.optional_output_column

    def __str__(self):
        return f"PreCollectionHandler({self.start_idx}, {self.end_idx}, {self.checkpoint}, {self.input_column}, {self.output_column}, {self.optional_output_column})"


    def run(
        self, 
        datamodel: BaseDatamodelsPipeline, 
        llm: BaseLLM
    ) -> None:
        """
        Executes the main functionality of the PreCollectionHandler.

        This method creates a pre collection from the given datamodel and llm.
        It iterates over the given range of indices and for each index it
        creates a prompt, runs the prompt on the llm, and adds the result to
        a dictionary. The dictionary is then saved as a feather file at the
        given checkpoint.

        The method also logs the progress of the pre collection.

        Parameters:
            datamodel (BaseDatamodelsPipeline): The datamodel to use for the pre collection.
            llm (BaseLLM): The llm to use for the pre collection.

        Returns:
            None
        """
        # Initialize variables
        pre_collection_dict: Dict[str, List[Any]] = self._initiate_pre_collection_dict()
        checkpoint_count = 0
        test_set_length = len(datamodel.test_set)
        end_idx = self.end_idx - 1

        # Iterate over the given range of indices
        for idx_row in range(self.start_idx, self.end_idx):
            
            # Log the initial time
            start_time = datetime.datetime.now()

            # Increment the checkpoint count
            checkpoint_count += 1

            # Convert the index to a binary vector
            binary_idx = self._convert_idx_to_binary(
                datamodel.train_collections_idx[idx_row] 
                if self.type == "train" 
                else datamodel.test_collections_idx[idx_row], 
                datamodel.train_set 
                if self.type == "train" 
                else datamodel.test_set
            )

            # Iterate over the test set
            for dev_idx in range(test_set_length):

                # Get the llm inference
                prompt = self._fill_prompt_template(
                    idx_row, dev_idx, datamodel.train_set, datamodel.test_set, datamodel.train_collections_idx
                )
                result = llm.run(prompt)

                # Get the output data
                output_data = datamodel.test_set.iloc[dev_idx][self.output_column]

                # Get the optional output data
                optional_data = (datamodel.test_set.iloc[dev_idx][self.optional_output_column] 
                                 if self.optional_output_column else None)

                # Add the element to the pre collection dictionary
                pre_collection_dict = self._add_element_to_collection(
                    pre_collection_dict, idx_row, dev_idx, binary_idx, result, output_data, optional_data
                )

            # If the checkpoint count is equal to the checkpoint or the end index is reached,
            # save the pre collection dictionary as a feather file
            if checkpoint_count == self.checkpoint or idx_row == end_idx:
                print(datetime.datetime.now())

                # Create a pandas dataframe from the pre collection dictionary
                df = pd.DataFrame(pre_collection_dict)
                print(f"Checkpoint {idx_row} saved")

                # Save the dataframe as a feather file
                subfolder = "train" if self.type == "train" else "test"
                df.to_feather(f"{self.datamodels_path}/pre_collections/{subfolder}/pre_collection_{idx_row}.feather")

                # Reset variables
                pre_collection_dict = self._initiate_pre_collection_dict()
                checkpoint_count = 0

            # If logging is enabled, log the progress of the pre collection
            if self.log:
                self._log(idx_row, start_time)
                
                

    
    def _initiate_pre_collection_dict(self) -> dict[str, list]:
        """
        Initiates a pre collection dictionary with the given columns.

        If the optional_output_column is not None, the dictionary will have an additional column for the optional output.

        Returns:
            dict[str, list]: A dictionary with the given columns, where each column is a list.
        """

        
        columns = ["collection_idx", "test_idx", "input", "predicted_output", "true_output"]
        if self.optional_output_column is not None:
            columns.append("optinal_output")
        return {col: [] for col in columns}


    def _fill_prompt_template(
        self,
        idx_row: int,
        idx_test: int,
        train_set: pd.DataFrame,
        test_set: pd.DataFrame,
        train_collections_idx: np.ndarray
    ) -> str:
        """
        Fills a prompt template with the instruction, context, and input of a data point.

        The context is constructed by concatenating the input and output of the data points
        in the train collection at the given index. The instruction is retrieved from the
        instructions dictionary with the task of the test data point at the given index.

        Args:
            idx_row (int): The index of the collection in the train collection index.
            idx_test (int): The index of the data point in the test set.
            train_set (pd.DataFrame): The train set.
            test_set (pd.DataFrame): The test set.
            train_collections_idx (np.ndarray): The train collection index.

        Returns:
            str: The filled prompt template.
        """
        template: str = """
            {instruction}

            Examples:
            {context}

            Input:
            {input}
        """

        context: str = ""
        for idx in train_collections_idx[idx_row]:
            input: str = train_set.loc[idx][self.input_column]
            output: str = train_set.loc[idx][self.output_column]
            context += f"{input}\n  {output}\n\n"

        input: str = test_set.loc[idx_test][self.input_column]
        instruction: str = self.instructions[test_set.loc[idx_test]["task"]]
        prompt: str = PromptTemplate.from_template(template).format(instruction=instruction, context=context, input=input)

        return prompt


    def _convert_idx_to_binary(self, arr: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """
        Convert an array of indices into a binary numpy array of the same length as a given DataFrame.

        Parameters
        ----------
        arr : np.ndarray
            The array of indices to convert.
        df : pd.DataFrame
            The DataFrame to use to determine the length of the output array.

        Returns
        -------
        np.ndarray
            The binary numpy array where the indices from the input array are 1 and the rest are 0.
        """
        indeces_df = np.zeros(len(df), dtype=int)
        indeces_df[arr] = 1
        return indeces_df

    
    def _add_element_to_collection(
        self, 
        pre_collection_dict: Dict[str, List[Any]], 
        collection_idx: int, 
        test_idx: int, 
        input: str, 
        predicted_output: str, 
        true_output: str, 
        optinal_output: Optional[str] = None
    ) -> Dict[str, List[Any]]:
        """
        Adds a new element to a pre collection dictionary.

        Parameters
        ----------
        pre_collection_dict : Dict[str, List[Any]]
            The pre collection dictionary to add the element to.
        collection_idx : int
            The index of the collection in the train collection index.
        test_idx : int
            The index of the datapoint in the test set.
        input : str
            The input of the datapoint.
        predicted_output : str
            The predicted output of the datapoint.
        true_output : str
            The true output of the datapoint.
        optinal_output : Optional[str], optional
            The optional output of the datapoint. Defaults to None.

        Returns
        -------
        Dict[str, List[Any]]
            The updated pre collection dictionary.
        """
        
        data = {
            "collection_idx": collection_idx,
            "test_idx": test_idx,
            "input": input,
            "predicted_output": predicted_output,
            "true_output": true_output
        }
        
        if optinal_output is not None:
            data["optinal_output"] = optinal_output

        for key, value in data.items():
            pre_collection_dict[key].append(value)

        return pre_collection_dict

        

    def _log(self, idx_row: int, start_time: datetime.datetime) -> None:
        """
        Logs the current pre collection to Weights and Biases.

        Args:
            idx_row (int): The current index of the pre collection.
            start_time (datetime.datetime): The start time of the logging process.

        Returns:
            None
        """
        wandb.init(
            project=self.log_config.project,
            dir=self.log_config.dir,
            id=f"{idx_row}_{self.log_config.id}",
            name=f"{idx_row}_{self.log_config.name}",
            config=self.log_config.config,
            tags=self.log_config.tags
        )

        wandb.log(
            {
                "idx": idx_row,
                "end_time": str(datetime.datetime.now()),
                "full_duration": str((datetime.datetime.now() - start_time).total_seconds()),
            }
        )
        wandb.finish()

        


