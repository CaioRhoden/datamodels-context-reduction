from abc import ABC, abstractmethod
import torch
from dmcr.models.BaseLLM import BaseLLM
from dmcr.models.BatchModel import BatchModel
import wandb
import os
import polars as pl
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from dataclasses import dataclass

from dmcr.datamodels.pipeline import DatamodelsIndexBasedPipeline
from dmcr.datamodels.config import LogConfig

@dataclass
class PreCollectionsConfig():
    mode: str
    instruction: dict | str
    llm: BaseLLM | BatchModel
    rag_indexes_path: str
    start_idx: int = 0
    end_idx: int = -1
    checkpoint: int = 50
    log: bool = False
    log_config: LogConfig | None = None
    model_configs: dict | None = None


class PreCollectionsPipeline(ABC):

    def __init__(self, datamodelsPipeline: DatamodelsIndexBasedPipeline):
        self.datamodels = datamodelsPipeline

    @abstractmethod
    def process(
        self,
        config: PreCollectionsConfig
    )
    
class BaseLLMPreCollectionsPipeline(PreCollectionsPipeline):

    def __init__(self, datamodelsPipeline: DatamodelsIndexBasedPipeline):
        super().__init__(datamodelsPipeline)

    def process(
        self,
        config: PreCollectionsConfig
    
    ):
        
        ## guarantees it's a single inference scenario
        assert isinstance(config.llm, BaseLLM)

        if rag_indexes_path.endswith(".json"):
            with open(rag_indexes_path, "r") as f:
                rag_indexes = json.load(f)
        
        else:
            raise ValueError(f"Unsupported file format, expected .json: {rag_indexes_path}")

        assert self.datamodels.test_set is not None
        assert rag_indexes is not None
        assert self.datamodels.train_set is not None
        assert self.datamodels.train_collections_idx is not None
        assert self.datamodels.test_collections_idx is not None

        if self.datamodels.train_collections_idx is None:
            raise ValueError("Train collection index not loaded")
        
        pre_collection_dict = self.datamodels._reset_pre_collection_dict()
        checkpoint_count = 0

        if self.datamodels.train_collections_idx is None or self.datamodels.test_collections_idx is None:
            raise Exception("Combinations for pre-collections creation not present")

        ### Set start and end index
        if end_idx == -1 and mode == "train":
            end_idx = start_idx + len(self.datamodels.train_collections_idx[start_idx:])
        elif end_idx == -1 and mode == "test":
            end_idx = start_idx + len(self.datamodels.test_collections_idx[start_idx:])
        
        ### Get size
        _keys = list(rag_indexes.keys())
        collection_size = len(rag_indexes[_keys[0]])

        ## Iterate over the combinations
        if log:


                if log_config is None:
                    raise Exception("Please provide a log configuration.")
                
                try:
                    wandb.init( 
                        project = log_config.project, 
                        dir = log_config.dir, 
                        id = f"{log_config.id}", 
                        name = f"{log_config.name}",
                        config = log_config.config,
                        tags = log_config.tags
                    )

                except:
                    raise Exception("Wandb not initialized, please check your log configuration")

        for idx_row in range(start_idx, end_idx):

            start_time = datetime.datetime.now()


            checkpoint_count += 1

            ## Convert index to binary vector
            if mode == "train":
                binary_idx = self.datamodels._convert_idx_to_binary(self.datamodels.train_collections_idx[idx_row], collection_size=collection_size)
            elif mode == "test":
                binary_idx = self.datamodels._convert_idx_to_binary(self.datamodels.test_collections_idx[idx_row],collection_size=collection_size)

            ## Get the input output pairs and concatenate into a string
            for dev_idx in range(len(self.datamodels.test_set)):
                prompt = self.datamodels._fill_prompt_template(idx_row, dev_idx, title_column, text_column, rag_indexes, question_column)
                print(f"Train collection index: {idx_row}, Dev index: {dev_idx}")

                if isinstance(llm, GenericInstructModelHF):
                    result = llm.run(prompt, instruction=str(instruction), config_params=model_configs)[0]["generated_text"]
                else:
                    result = llm.run(prompt)



                ## Get true output and verify the expected behavior    
                try:
                    true_output = self.datamodels.test_set[dev_idx][output_column].to_numpy().flatten()[0].tolist()
                    assert type(true_output) is list
                    assert type(true_output[0]) is str
                except AssertionError:
                    raise Exception("True output format not supported, the expected is a list of strings in a Polars dataframe but what received to extract was : ", self.datamodels.test_set[dev_idx][output_column])
                
                
                
                pre_collection_dict = self.datamodels._add_element_to_collection(pre_collection_dict, idx_row, dev_idx, binary_idx, result, true_output)
            
            ## Saving condition in checkpoint or end of indezes
            if checkpoint_count == checkpoint or idx_row == end_idx-1:

                print(datetime.datetime.now())

                df = pl.DataFrame(pre_collection_dict)
                print(f"Checkpoint {idx_row} saved")
                
                if not os.path.exists(f"{self.datamodels.datamodels_path}/pre_collections"):
                    os.mkdir(f"{self.datamodels.datamodels_path}/pre_collections")
                
                if not os.path.exists(f"{self.datamodels.datamodels_path}/pre_collections/train"):
                    os.mkdir(f"{self.datamodels.datamodels_path}/pre_collections/train")

                if not os.path.exists(f"{self.datamodels.datamodels_path}/pre_collections/test"):
                    os.mkdir(f"{self.datamodels.datamodels_path}/pre_collections/test")

                if mode == "train":
                    df.write_ipc(f"{self.datamodels.datamodels_path}/pre_collections/train/pre_collection_{idx_row}.feather")
                elif mode == "test":
                    df.write_ipc(f"{self.datamodels.datamodels_path}/pre_collections/test/pre_collection_{idx_row}.feather")


                pre_collection_dict = self.datamodels._reset_pre_collection_dict()
                checkpoint_count = 0

            if log:
                wandb.log({"idx": idx_row, "end_time": str(datetime.datetime.now()), "full_duration": str((datetime.datetime.now() - start_time).total_seconds())})

        if log:
            wandb.finish()


    