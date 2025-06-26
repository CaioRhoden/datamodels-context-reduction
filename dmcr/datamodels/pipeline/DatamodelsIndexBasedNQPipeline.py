from dmcr.datamodels.config import DatamodelIndexBasedConfig, LogConfig
from dmcr.datamodels.pipeline.TrainModelsPipeline import TrainModelsPipeline
from dmcr.evaluators import BaseEvaluator
from dmcr.models import BaseLLM, GenericInstructModelHF

import polars as pl
import numpy as np
import torch

from langchain.prompts import PromptTemplate
from dmcr.datamodels.models import FactoryLinearRegressor
import h5py
import json
import os
from torch.utils.data import TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
import datetime
import wandb

from pathlib import Path






class DatamodelsIndexBasedNQPipeline:
    
    def __init__(self, config: DatamodelIndexBasedConfig, test_flag=False) -> None:

        """
        Initializes a new instance of the Datamodels class.

        Parameters:
            config (DatamodelConfig): The configuration for the datamodels.

        Returns:
            None

        TODO: Add customization for context instruction
        """

        self.train_collections_idx = None
        self.test_collections_idx = None
        self.train_set = None
        self.test_set = None

        self.k = config.k
        self.num_models = config.num_models
        self.datamodels_path = config.datamodels_path


        if not test_flag:
            self._verify_repo_structure()
            self.set_collections_index()
            self.set_dataframes(config.train_set_path, config.test_set_path)
    
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
        with h5py.File(f"{self.datamodels_path}/train_collection.h5", "r") as f:
            self.train_collections_idx = f["train_collection"][()]
        print("Loaded train collection index")

        with h5py.File(f"{self.datamodels_path}/test_collection.h5", "r") as f:
            self.test_collections_idx = f["test_collection"][()]
        print("Loaded test collection index")
        
    def _verify_repo_structure(self):

        if not os.path.exists(f"{self.datamodels_path}/train_collection.h5"):
            raise Exception("Train collection not found in datamodels path")
        if not os.path.exists(f"{self.datamodels_path}/test_collection.h5"):
            raise Exception("Test collection not found in datamodels path")
        
    def set_dataframes(self, train_set_path: str, test_set_path: str):
        """
        Loads the test set if it has not been loaded yet.

        Returns:
            None
        """

        if train_set_path.endswith(".csv"):
            self.train_set = pl.read_csv(train_set_path)
        elif train_set_path.endswith(".feather"):
            self.train_set = pl.read_ipc(train_set_path)
        else:
            raise ValueError(f"Unsupported file format: {train_set_path}")
        
        if test_set_path.endswith(".csv"):
            self.test_set = pl.read_csv(test_set_path)
        elif test_set_path.endswith(".feather"):
            self.test_set = pl.read_ipc(test_set_path)
        else:
            raise ValueError(f"Unsupported file format: {test_set_path}")



        print("Loaded train set")


    def create_pre_collection(
        self,
        mode: str,
        instruction: dict | str,
        llm: BaseLLM,
        rag_indexes_path: str,
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
    
    ):
        

        if rag_indexes_path.endswith(".json"):
            with open(rag_indexes_path, "r") as f:
                rag_indexes = json.load(f)
        
        else:
            raise ValueError(f"Unsupported file format, expected .json: {rag_indexes_path}")

        assert self.test_set is not None
        assert rag_indexes is not None
        assert self.train_set is not None
        assert self.train_collections_idx is not None
        assert self.test_collections_idx is not None

        if self.train_collections_idx is None:
            raise ValueError("Train collection index not loaded")
        
        pre_collection_dict = self._reset_pre_collection_dict()
        checkpoint_count = 0

        if self.train_collections_idx is None or self.test_collections_idx is None:
            raise Exception("Combinations for pre-collections creation not present")

        ### Set start and end index
        if end_idx == -1 and mode == "train":
            end_idx = start_idx + len(self.train_collections_idx[start_idx:])
        elif end_idx == -1 and mode == "test":
            end_idx = start_idx + len(self.test_collections_idx[start_idx:])
        
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
                binary_idx = self._convert_idx_to_binary(self.train_collections_idx[idx_row], collection_size=collection_size)
            elif mode == "test":
                binary_idx = self._convert_idx_to_binary(self.test_collections_idx[idx_row],collection_size=collection_size)

            ## Get the input output pairs and concatenate into a string
            for dev_idx in range(len(self.test_set)):
                prompt = self._fill_prompt_template(idx_row, dev_idx, title_column, text_column, rag_indexes, question_column)
                print(f"Train collection index: {idx_row}, Dev index: {dev_idx}")

                if isinstance(llm, GenericInstructModelHF):
                    result = llm.run(prompt, instruction=str(instruction), config_params=model_configs)[0]["generated_text"]
                else:
                    result = llm.run(prompt)



                ## Get true output and verify the expected behavior    
                try:
                    true_output = self.test_set[dev_idx][output_column].to_numpy().flatten()[0].tolist()
                    assert type(true_output) is list
                    assert type(true_output[0]) is str
                except AssertionError:
                    raise Exception("True output format not supported, the expected is a list of strings in a Polars dataframe but what received to extract was : ", self.test_set[dev_idx][output_column])
                
                
                
                pre_collection_dict = self._add_element_to_collection(pre_collection_dict, idx_row, dev_idx, binary_idx, result, true_output)
            
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


                pre_collection_dict = self._reset_pre_collection_dict()
                checkpoint_count = 0

            if log:
                wandb.log({"idx": idx_row, "end_time": str(datetime.datetime.now()), "full_duration": str((datetime.datetime.now() - start_time).total_seconds())})

        if log:
            wandb.finish()
        

            

    def create_collection(
        self,
        evaluator: BaseEvaluator,
        collection_name: str,
        mode: str = "train",
        log: bool = False,
        log_config: LogConfig | None = None,
        checkpoint: int | None = None
    ):
        
        start_time = datetime.datetime.now()
        ## Verfiy if pre-collections exist
        
        pre_collections_path = f"{self.datamodels_path}/pre_collections/{mode}"
        assert os.path.exists(pre_collections_path)       

        ## Read all pre-collections
        feather_files = Path(pre_collections_path).glob('*.feather')
        dfs = [pl.read_ipc(file) for file in sorted(feather_files)]
        pre_collections = pl.concat(dfs, how='vertical')

        ## Verify if pre-collections are not empty
        assert len(pre_collections) > 0

        ##Checkponints
        if checkpoint is None:
            checkpoint = len(pre_collections)

        ## Init Log
        if log:

            if log_config is None:
                raise Exception("Please provide a log configuration.")
            
            try:
                wandb.init( 
                    project = log_config.project, 
                    dir = log_config.dir, 
                    id = f"{collection_name}_{log_config.id}", 
                    name = f"{collection_name}_{log_config.name}",
                    config = log_config.config,
                    tags = log_config.tags
                )

            except:
                raise Exception("Wandb not initialized, please check your log configuration")


        ## Break chunks
        chunk_size = 0
        print(f"len pre collections {len(pre_collections)}")
        while chunk_size < len(pre_collections):
            print(f"Starting chunk {chunk_size}")
            next_chunk = max(chunk_size+checkpoint, len(pre_collections))
            pre_collections_chunk = pre_collections[chunk_size:next_chunk]
            ## Evaluate the pre_collections and add them to the dataframe
            evaluation = evaluator.evaluate(pre_collections_chunk["true_output"].to_numpy(),  pre_collections_chunk["predicted_output"].to_numpy())
            pre_collections_chunk  = pre_collections_chunk.with_columns(pl.Series("evaluation", evaluation).cast(pl.Float64).alias("evaluation"))
            collection = pre_collections_chunk[["collection_idx","test_idx","input", "evaluation"]]

            ## Guarantees the necessary directories
            if not os.path.exists(f"{self.datamodels_path}/collections"):
                os.mkdir(f"{self.datamodels_path}/collections")
            
            if not os.path.exists(f"{self.datamodels_path}/collections/train"):
                os.mkdir(f"{self.datamodels_path}/collections/train")

            if not os.path.exists(f"{self.datamodels_path}/collections/test"):
                os.mkdir(f"{self.datamodels_path}/collections/test")

            ## Save file
            collection.write_ipc(f"{self.datamodels_path}/collections/{mode}/{collection_name}_{chunk_size}.feather")
            print(f"Chunk {chunk_size} saved")

            ## Log duration and evaluationss
            if log:
                wandb.log({
                    "collection_name": f"{collection_name}_{mode}_{chunk_size}", 
                    "end_time": str(datetime.datetime.now()), 
                    "full_duration": str((datetime.datetime.now() - start_time).total_seconds()), 
                    "evaluation": evaluation
                })
            
            chunk_size = chunk_size + checkpoint
            
            

                
                
        if log:
            wandb.finish()



    def train_datamodels(
            self,
            model_factory: FactoryLinearRegressor,
            collection_name: str,
            epochs: int,
            train_batches: int,
            val_batches: int,
            val_size: float,
            lr: float ,
            patience: int,
            random_seed: int = 42,
            log: bool = False,
            log_epochs: int = 1,
            log_config: LogConfig | None = None,
            run_id: str = "weights",
                         
    ):
            
        """
        Trains multiple data models using the specified parameters.

        Parameters:
            model (LinearRegressor): The linear regression model to be trained.
            collection_name (str): The name of the data collection to be used for training.
            epochs (int): The number of epochs for training each model.
            train_batches (int): The number of batches for training data.
            val_batches (int): The number of batches for validation data.
            val_size (float): The proportion of the dataset to include in the validation split.
            lr (float): The learning rate for the optimizer.
            patience (int): The number of epochs to wait for improvement before early stopping.
            random_seed (int, optional): The seed for random number generation to ensure reproducibility. Default is 42.
            log (bool, optional): Whether to log the training process. Default is False.
            log_epochs (int, optional): Frequency of logging in terms of epochs. Default is 1.
            log_config (LogConfig | None, optional): Configuration for logging. Default is None.
            run_id (str, optional): The identifier for the training run. Default is "weights".

        Returns:
            None
        """

        train_models = TrainModelsPipeline(self, model_factory)
        train_models.train_datamodels(
            collection_name, 
            epochs, 
            train_batches, 
            val_batches, 
            val_size, 
            lr, 
            patience, 
            random_seed, 
            log, 
            log_epochs, 
            log_config, 
            run_id
        )

    def evaluate_test_collections(
            self,
            evaluation_id: str,
            collection_name: str,
            model_id: str,
            metric: str = "R2Score",
            log: bool = False,
            log_config: LogConfig | None = None
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        ## Load model parameters
        weigths = torch.load(f"{self.datamodels_path}/models/{model_id}/weights.pt", weights_only=True)
        bias = torch.load(f"{self.datamodels_path}/models/{model_id}/bias.pt", weights_only=True)

       ## Create and verify list of files from collection
        colleciton_path = f"{self.datamodels_path}/collections/test/"
        collections_arr = [os.path.join(colleciton_path, f) for f in os.listdir(colleciton_path) if f.endswith(".feather")]
        if len(collections_arr) == 0:
            raise Exception("No collections found in test folder")

        df = pl.concat([pl.read_ipc(file) for file in collections_arr], how="vertical")



        evaluations = {
            f"metric_{metric}": [],
            "test_idx": [],
        }
        ### Run evaluation for each test collection
        if log:
            if log_config is None:
                raise Exception("Please provide a log configuration.")
            
            if not os.path.exists(log_config.dir):
                os.mkdir(log_config.dir)

            wandb.init( 
                project = log_config.project, 
                dir = log_config.dir, 
                id = f"{collection_name}_{log_config.id}", 
                name = f"{collection_name}_{log_config.name}",
                config = log_config.config,
                tags = log_config.tags
            )


        for idx in range(self.num_models):
                
                print(f"Model {idx} under evaluation")

                ## Preoare dataset
                
                _temp = (
                    df.filter(pl.col("test_idx") == idx)
                    .select(pl.col("input"), pl.col("evaluation"))
                )

                _x = _temp["input"].to_numpy()
                _y = _temp["evaluation"].to_numpy()

                dataset = torch.utils.data.TensorDataset(torch.tensor(_x, device=device), torch.tensor(_y, device=device))
                test_loader = torch.utils.data.DataLoader(dataset, batch_size=len(_x))

                ## Load models
                model = LinearRegressor(len(dataset[0][0]), 1)
                model.load_state_dict({ "linear.weight": weigths[idx].unsqueeze(0), "linear.bias": bias[idx].unsqueeze(0) })
                model.to(device)

                ## Evaluate in test
                inputs, target = next(iter(test_loader))
                total_metric = model.evaluate(inputs.to(device).to(dtype=torch.float32), target.to(device).to(dtype=torch.float32), metric=metric)

                evaluations[f"metric_{metric}"].append(total_metric)
                evaluations["test_idx"].append(idx)
                if log:
                    wandb.log({"test_idx": idx, "mean_metric": total_metric, "metric": metric})

        ## Save evaluations
        if not os.path.exists(f"{self.datamodels_path}/evaluations"):
            os.mkdir(f"{self.datamodels_path}/evaluations")

        pl.DataFrame(evaluations).write_ipc(f"{self.datamodels_path}/evaluations/{evaluation_id}.feather")

        if log:
            wandb.finish()

        





    
    def _add_element_to_collection(self, pre_collection_dict, collection_idx, test_idx, input, predicted_output, true_output):

        pre_collection_dict["collection_idx"].append(collection_idx)
        pre_collection_dict["test_idx"].append(test_idx)
        pre_collection_dict["input"].append(input)
        pre_collection_dict["predicted_output"].append(predicted_output)
        pre_collection_dict["true_output"].append(true_output)

        return pre_collection_dict

    def _fill_prompt_template(
            self,
            idx_row: int,
            idx_test: int,
            title_column: str,
            text_column: str,
            rag_indexes: dict,
            question_column: str      ) -> str:
        
        template = """
            Documents:
            {context}

            Question: {input}\nAnswer: 
        """

        context = ""
        count = 0
        for collection_idx in self.train_collections_idx[idx_row]:

            idx = rag_indexes[str(idx_test)][collection_idx]
            title = self.train_set[idx][title_column].to_numpy().flatten()[0]
            text = self.train_set[idx][text_column].to_numpy().flatten()[0]
            context += f"Document[{count}](Title: {title}){text}\n\n"
            count += 1

        
        input = self.test_set[idx_test][question_column].to_numpy().flatten()[0]

        prompt = PromptTemplate.from_template(template).format(context=context, input=input)

        return prompt

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