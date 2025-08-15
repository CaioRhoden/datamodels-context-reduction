import datetime
from typing import Literal, Callable
import wandb
import os
import polars as pl
import json
import numpy as np


from dmcr.datamodels.pipeline import DatamodelsIndexBasedPipeline
from dmcr.datamodels.config import LogConfig
from dmcr.models.BaseLLM import BaseLLM
from dmcr.models.BatchModel import BatchModel
from dmcr.models.GenericInstructModelHF import GenericInstructModelHF




class PreCollectionsPipeline():

    def __init__(self, 
                 datamodelsPipeline: DatamodelsIndexBasedPipeline):
        self.datamodels = datamodelsPipeline,
        


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
        assert self.datamodels.test_set is not None
        assert self.datamodels.train_set is not None
        assert rag_indexes is not None

    def _save_checkpoint(self, pre_collection_dict, idx_row, mode):
        df = pl.DataFrame(pre_collection_dict)
        save_dir = Path(self.datamodels.datamodels_path) / "pre_collections" / mode
        save_dir.mkdir(parents=True, exist_ok=True)
        df.write_ipc(save_dir / f"pre_collection_{idx_row}.feather", compression="zstd")
        return self.datamodels._reset_pre_collection_dict()
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

    
class BaseLLMPreCollectionsPipeline(PreCollectionsPipeline):

    def __init__(self, 
                datamodelsPipeline: DatamodelsIndexBasedPipeline,
                mode: Literal["train", "test"],
                instruction: dict | str,
                model: BaseLLM | BatchModel,
                context_strategy: Callable[[int, int, dict, DatamodelsIndexBasedPipeline], str],
                rag_indexes_path: str,
                output_column: str,
                start_idx: int = 0,
                end_idx: int = -1,
                checkpoint: int = 50,
                log: bool = False,
                log_config: LogConfig | None = None,
                model_configs: dict | None = None,      
            
                ):
        super().__init__(datamodelsPipeline)
        self.mode = mode,
        self.instruction = instruction,
        self.model = model,
        self.context_strategy = context_strategy,
        self.rag_indexes_path = rag_indexes_path,
        self.output_column = output_column,
        self.start_idx = start_idx,
        self.end_idx = end_idx,
        self.checkpoint = checkpoint,
        self.log = log,
        self.log_config = log_config,
        self.model_configs = model_configs
    

    def process(self) -> None: 
       
        ## guarantees it's a single inference scenario
        with open(self.rag_indexes_path, "r") as f:
            rag_indexes = json.load(f)
        
        self._validate_inputs(self.mode, self.model, rag_indexes)
        if not isinstance(self.model, BaseLLM):
            raise TypeError("Model must be an instance of BaseLLM for this PreCollectionPipeline class")



        
        pre_collection_dict = self._reset_pre_collection_dict()
        checkpoint_count = 0

        ### Set start and end index
        if self.end_idx == -1:
            dataset_idx = getattr(self.datamodels, f"{self.mode}_collections_idx")
            end_idx = self.start_idx + len(dataset_idx[self.start_idx:])
        
        ### Get size
        _keys = list(rag_indexes.keys())
        collection_size = len(rag_indexes[_keys[0]])

        ## Iterate over the combinations
        if self.log:
            self._init_logging(self.log_config)
            
        for idx_row in range(self.start_idx, self.end_idx):

            start_time = datetime.datetime.now()

            if self.mode == "train":
                binary_idx = self._convert_idx_to_binary(self.datamodels.train_collections_idx[idx_row], collection_size=collection_size)
            elif self.mode == "test":
                binary_idx = self._convert_idx_to_binary(self.datamodels.test_collections_idx[idx_row],collection_size=collection_size)
            
            for dev_idx, _ in enumerate(self.datamodels.test_set):
                prompt = self.context_strategy(idx_row, dev_idx, rag_indexes, self.datamodels)
                print(f"Train collection index: {idx_row}, Dev index: {dev_idx}")

                if isinstance(self.model, GenericInstructModelHF):
                    result = self.model.run(prompt, instruction=str(self.instruction), config_params=model_configs)[0]["generated_text"]
                else:
                    result = self.model.run(prompt)

                ## Get true output and verify the expected behavior    
                try:
                    true_output = self.datamodels.test_set[dev_idx][self.output_column].to_numpy().flatten()[0].tolist()
                    assert isinstance(true_output, list)
                    assert isinstance(true_output[0], str)


                except AssertionError as exc:
                    raise ValueError(
                        "True output format not supported, the expected is a list of strings in a Polars dataframe but what received to extract was : "
                        f"{self.datamodels.test_set[dev_idx][self.output_column]}"
                    ) from exc
                pre_collection_dict = self._add_row(pre_collection_dict, idx_row, dev_idx, binary_idx, result, true_output)
            
            ## Saving condition in checkpoint or end of indezes
            checkpoint_count += 1
            if checkpoint_count == self.checkpoint or idx_row == end_idx-1:
                pre_collection_dict = self._save_checkpoint(pre_collection_dict, idx_row, self.mode)
                checkpoint_count = 0

            if self.log:
                wandb.log({"idx": idx_row, "end_time": str(datetime.datetime.now()), "full_duration": str((datetime.datetime.now() - start_time).total_seconds())})

        if self.log:
            wandb.finish()


    