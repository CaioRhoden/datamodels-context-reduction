import datetime
from typing import Literal, Callable
import wandb
import os
import polars as pl
import json
import numpy as np
from pathlib import Path



from dmcr.datamodels.pipeline.DatamodelsPipelineData import DatamodelsPreCollectionsData
from dmcr.datamodels.config import LogConfig
from dmcr.models.BaseLLM import BaseLLM
from dmcr.models.BatchModel import BatchModel
from dmcr.models.GenericInstructModelHF import GenericInstructModelHF
from dmcr.models.GenericInstructBatchHF import GenericInstructBatchHF




class PreCollectionsPipeline():

    def __init__(self, datamodels_data: DatamodelsPreCollectionsData):
        self.datamodels_data = datamodels_data
        


    def process(
        self,
    ) -> None: 
        pass


    def _add_row(self, pre_collection_dict, collection_idx, test_idx, input, predicted_output, true_output):

        """
        Adds a new row of data to the `pre_collection_dict`.

        Args:
            pre_collection_dict (dict): Dictionary to store pre-collection data.
            collection_idx: Index of the collection.
            test_idx: Index of the test.
            input: Input data for the test case.
            predicted_output: Predicted output from the model.
            true_output: Actual output to compare against.

        Returns:
            dict: Updated `pre_collection_dict` with the new row of data added.
        """

        pre_collection_dict["collection_idx"].append(collection_idx)
        pre_collection_dict["test_idx"].append(test_idx)
        pre_collection_dict["input"].append(input)
        pre_collection_dict["predicted_output"].append(predicted_output)
        pre_collection_dict["true_output"].append(true_output)

        return pre_collection_dict
    

    def _convert_idx_to_binary(self, arr: np.ndarray, collection_size: int) -> np.ndarray:
        
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
        indeces_df =  np.zeros(collection_size, dtype=int)
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
    
    def _validate_inputs(self, mode, model, rag_indexes):
        if mode not in ("train", "test"):
            raise ValueError(f"Invalid mode: {mode}")
        assert self.datamodels_data.test_set is not None
        assert self.datamodels_data.train_set is not None
        assert rag_indexes is not None

    def _save_checkpoint(self, pre_collection_dict, idx_row, mode):
        df = pl.DataFrame(pre_collection_dict)
        save_dir = Path(self.datamodels_data.datamodels_path) / "pre_collections" / mode
        save_dir.mkdir(parents=True, exist_ok=True)
        df.write_ipc(save_dir / f"pre_collection_{idx_row}.feather", compression="zstd")

        return self._reset_pre_collection_dict()
    
    def _init_logging(self, log_config: LogConfig):
        """
        Initialize wandb logging.

        Args:
            log_config (LogConfig): Configuration for logging.

        Raises:
            ValueError: If log_config is None.
            RuntimeError: If wandb initialization fails.
        """
        try:
            wandb.init(
                project=log_config.project,
                dir=log_config.dir,
                id=log_config.id,
                name=log_config.name,
                config=log_config.config,
                tags=log_config.tags
            )
        except Exception as e:
            raise RuntimeError("Wandb initialization failed") from e

    def _parse_generation_output(self, output: dict, thinking: bool=False) -> str:
        """
        Parse the output of the generation model, analyze if is it "enable_thinking"

        Parameters:
        - output (str): The raw output from the generation model.

        Returns:
        - str: The parsed output.
        """
        # Implement your parsing logic here
        
        if thinking:
            # Example parsing logic for "enable_thinking"
            # This is a placeholder; replace with actual logic as needed
            parsed_output = str(output["generated_text"].split("</think>")[-1].strip())
        else:
            parsed_output = str(output["generated_text"])

        return parsed_output
    
class BaseLLMPreCollectionsPipeline(PreCollectionsPipeline):

    def __init__(self, 
                datamodels_data: DatamodelsPreCollectionsData,
                mode: Literal["train", "test"],
                instruction: dict | str,
                model: BaseLLM,
                context_strategy: Callable[[int, int, dict, DatamodelsPreCollectionsData], str],
                rag_indexes_path: str,
                output_column: str,
                start_idx: int = 0,
                end_idx: int = -1,
                checkpoint: int = 50,
                log: bool = False,
                log_config: LogConfig | None = None,
                model_configs: dict | None = None,      
            
                ):
        self.datamodels_data = datamodels_data
        self.mode = mode
        self.instruction = instruction
        self.model = model
        self.context_strategy = context_strategy
        self.rag_indexes_path = rag_indexes_path
        self.output_column = output_column
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.checkpoint = checkpoint
        self.log = log
        self.log_config = log_config
        self.model_configs = model_configs
    

    def process(self) -> None: 
       
        ## guarantees it's a single inference scenario
        """
        Process a single inference scenario.

        This method guarantees a single inference scenario.
        It loads the rag indexes, validates the inputs, and then iterates over the combinations.
        It generates prompts using the context strategy and runs the model.
        It then formats the output and saves it in a pre collection dictionary.
        Finally, it saves the pre collection dictionary in a checkpoint or at the end of the indezes.

        Args:
            None

        Returns:
            None
        """
        rag_indexes = json.load(open(self.rag_indexes_path, "r"))

        
        self._validate_inputs(self.mode, self.model, rag_indexes)
        if not isinstance(self.model, BaseLLM):
            raise TypeError("Model must be an instance of BaseLLM for this PreCollectionPipeline class")



        
        pre_collection_dict = self._reset_pre_collection_dict()
        checkpoint_count = 0

        ### Set start and end index
        if self.end_idx == -1:
            dataset_idx = getattr(self.datamodels_data, f"{self.mode}_collections_idx")
            self.end_idx = self.start_idx + len(dataset_idx[self.start_idx:])
        
        ### Get size
        _keys = list(rag_indexes.keys())
        collection_size = len(rag_indexes[_keys[0]])

        ## Iterate over the combinations
        if self.log:
            self._init_logging(self.log_config)
            
        for idx_row in range(self.start_idx, self.end_idx):

            start_time = datetime.datetime.now()

            if self.mode == "train":
                binary_idx = self._convert_idx_to_binary(self.datamodels_data.train_collections_idx[idx_row], collection_size=collection_size)
            elif self.mode == "test":
                binary_idx = self._convert_idx_to_binary(self.datamodels_data.test_collections_idx[idx_row],collection_size=collection_size)
            

            for sample_idx, _ in enumerate(self.datamodels_data.test_set["idx"]):
                prompt = self.context_strategy(idx_row, sample_idx, rag_indexes, self.datamodels_data)
                print(f"Train collection index: {idx_row}, Dev index: {sample_idx}")

                if isinstance(self.model, GenericInstructModelHF):
                    result = self._parse_generation_output(self.model.run(prompt, instruction=str(self.instruction), config_params=self.model_configs)[0], self.model.thinking)
                else:
                    result = self.model.run(prompt)

                ## Get true output and verify the expected behavior    
                try:
                    true_output = self.datamodels_data.test_set[sample_idx][self.output_column].to_numpy().flatten()[0].tolist()
                    assert isinstance(true_output, list)
                    assert isinstance(true_output[0], str)


                except AssertionError as exc:
                    raise ValueError(
                        "True output format not supported, the expected is a list of strings in a Polars dataframe but what received to extract was : "
                        f"{self.datamodels_data.test_set[sample_idx][self.output_column]}"
                    ) from exc
                pre_collection_dict = self._add_row(pre_collection_dict, idx_row, sample_idx, binary_idx, result, true_output)
            
            ## Saving condition in checkpoint or end of indezes
            checkpoint_count += 1
            if checkpoint_count == self.checkpoint or idx_row == (self.end_idx-1):
                pre_collection_dict = self._save_checkpoint(pre_collection_dict, idx_row, self.mode)
                checkpoint_count = 0

            if self.log:
                wandb.log({"idx": idx_row, "end_time": str(datetime.datetime.now()), "full_duration": str((datetime.datetime.now() - start_time).total_seconds())})

        if self.log:
            wandb.finish()

class BatchLLMPreCollectionsPipeline(PreCollectionsPipeline):

    def __init__(self, 
                datamodels_data: DatamodelsPreCollectionsData,
                mode: Literal["train", "test"],
                instruction: dict | str,
                model: BatchModel,
                batch_size: int,
                context_strategy: Callable[[int, int, dict, DatamodelsPreCollectionsData], str],
                rag_indexes_path: str,
                output_column: str,
                start_idx: int = 0,
                end_idx: int = -1,
                checkpoint: int = 50,
                log: bool = False,
                log_config: LogConfig | None = None,
                model_configs: dict | None = None,      
            
                ):
        """
        Initialize a BatchLLMPreCollectionsPipeline.

        Args:
            datamodels_data (DatamodelsPreCollectionsData): The pipeline to use for generating the pre-collection data.
            mode (Literal["train", "test"]): The mode to use for generating the pre-collection data.
            instruction (dict | str): The instruction to use for generating the pre-collection data.
            model (BatchModel): The model to use for generating the pre-collection data.
            context_strategy (Callable[[int, int, dict, DatamodelsPreCollectionsData], str]): The strategy (function) to use for generating the context for the model.
            rag_indexes_path (str): The path to the rag indexes file.
            output_column (str): The column to write the output to.
            start_idx (int, optional): The starting index for generating the pre-collection data. Defaults to 0.
            end_idx (int, optional): The ending index for generating the pre-collection data. Defaults to -1.
            checkpoint (int, optional): The checkpoint interval for saving the pre-collection data. Defaults to 50.
            log (bool, optional): Whether to log the pre-collection data generation progress. Defaults to False.
            log_config (LogConfig | None, optional): The configuration for logging. Defaults to None.
            model_configs (dict | None, optional): The configurations for the model. Defaults to None.
        """
        self.datamodels_data = datamodels_data
        self.mode = mode
        self.instruction = instruction
        self.model = model
        self.batch_size = batch_size
        self.context_strategy = context_strategy
        self.rag_indexes_path = rag_indexes_path
        self.output_column = output_column
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.checkpoint = checkpoint
        self.log = log
        self.log_config = log_config
        self.model_configs = model_configs
    

    def process(self) -> None: 
            """
            Process the dataset in batches, generating model outputs and saving results.

            This method iterates over a range of collection indices, batching prompts and true outputs for efficient model inference.
            For each collection index, it generates prompts using the context strategy, collects true outputs, and accumulates them into batches.
            When a batch is ready (either the batch size is reached or the last sample is processed), it runs the model on the batch and stores the results.
            The results are added to a dictionary, which is periodically saved to disk as a checkpoint.
            Logging is performed if enabled.
            """
            # Load RAG (Retrieval-Augmented Generation) indexes from file
            with open(self.rag_indexes_path, "r") as f:
                rag_indexes = json.load(f)

            # Validate input arguments and model type
            self._validate_inputs(self.mode, self.model, rag_indexes)
            if not isinstance(self.model, BatchModel):
                raise TypeError("Model must be an instance of BatchModel for this PreCollectionPipeline class")

            # Initialize the dictionary to store pre-collection results
            pre_collection_dict = self._reset_pre_collection_dict()
            checkpoint_count = 0

            # Set the end index if not provided
            if self.end_idx == -1:
                dataset_idx = getattr(self.datamodels_data, f"{self.mode}_collections_idx")
                self.end_idx = self.start_idx + len(dataset_idx[self.start_idx:])

            # Determine the size of each collection
            _keys = list(rag_indexes.keys())
            collection_size = len(rag_indexes[_keys[0]])

            # Initialize logging if enabled
            if self.log:
                self._init_logging(self.log_config)

            # Iterate over each collection index in the specified range
            for idx_row in range(self.start_idx, self.end_idx):

                start_time = datetime.datetime.now()  # Track start time for logging

                # Convert collection indices to binary representation for the current collection
                if self.mode == "train":
                    binary_idx = self._convert_idx_to_binary(self.datamodels_data.train_collections_idx[idx_row], collection_size=collection_size)
                elif self.mode == "test":
                    binary_idx = self._convert_idx_to_binary(self.datamodels_data.test_collections_idx[idx_row], collection_size=collection_size)

                batch_buffer = 0  # Counter for the current batch size
                batch_pairs = []  # List to accumulate (prompt, true_output, sample_idx) tuples

                # Iterate over each sample in the test set
                for sample_idx, _ in enumerate(self.datamodels_data.test_set["idx"]):
                    # Generate the prompt for the current sample
                    prompt = self.context_strategy(idx_row, sample_idx, rag_indexes, self.datamodels_data)
                    print(f"Train collection index: {idx_row}, Dev index: {sample_idx}")

                    # Retrieve the true output and verify its format
                    true_output = self.datamodels_data.test_set[sample_idx][self.output_column].to_numpy().flatten()[0].tolist()
                    assert isinstance(true_output, list)
                    assert isinstance(true_output[0], str)

                    # Accumulate batch until batch_size is reached or last sample is processed
                    if batch_buffer < (self.batch_size-1) and sample_idx < (len(self.datamodels_data.test_set) - 1):
                        batch_pairs.append([prompt, true_output, sample_idx])
                        batch_buffer += 1
                    else:
                        # Add the last sample to the batch
                        batch_pairs.append((prompt, true_output, sample_idx))
                        # Run the model on the batch
                        if isinstance(self.model, GenericInstructBatchHF):
                            _list_results = self.model.run([pair[0] for pair in batch_pairs], instruction=str(self.instruction), config_params=self.model_configs)
                            results = [self._parse_generation_output(result[0], self.model.thinking) for result in _list_results]
                        else:
                            results = self.model.run(prompt)

                        # Store results in the pre_collection_dict
                        
                        for _results_idx in range(len(results)):
                            pre_collection_dict = self._add_row(
                                pre_collection_dict,
                                idx_row,
                                batch_pairs[_results_idx][2],  # current sample_idx (may be redundant)
                                binary_idx,
                                results[_results_idx],
                                batch_pairs[_results_idx][1],  # true_output
                            )
                            print(f"Added row {idx_row}, for question {sample_idx}")

                        # Reset batch buffer and pairs for the next batch
                        batch_buffer = 0
                        batch_pairs = []

                # Save checkpoint if needed (either at interval or at the end)
                checkpoint_count += 1
                if checkpoint_count == self.checkpoint or idx_row == self.end_idx-1:
                    pre_collection_dict = self._save_checkpoint(pre_collection_dict, idx_row, self.mode)
                    checkpoint_count = 0

                # Log progress if enabled
                if self.log:
                    wandb.log({
                        "idx": idx_row,
                        "end_time": str(datetime.datetime.now()),
                        "full_duration": str((datetime.datetime.now() - start_time).total_seconds())
                    })

            # Finish logging session if enabled
            if self.log:
                wandb.finish()