

import polars as pl
import numpy as np
import torch

from langchain.prompts import PromptTemplate
from dmcr.datamodels.models import FactoryLinearRegressor, LinearRegressor
import h5py
import json
import os
import datetime
import wandb
from pathlib import Path
from typing import Literal, Callable



from dmcr.datamodels.config import DatamodelIndexBasedConfig, LogConfig
from dmcr.datamodels.pipeline.TrainModelsPipeline import TrainModelsPipeline
from dmcr.evaluators import BaseReferenceEvaluator, BaseUnsupervisedEvaluator
from dmcr.models import BaseLLM, GenericInstructModelHF, BatchModel
from dmcr.datamodels.pipeline.DatamodelsPipelineData import DatamodelsPreCollectionsData
from dmcr.datamodels.pipeline.PreCollectionsPipeline import BaseLLMPreCollectionsPipeline, BatchLLMPreCollectionsPipeline






class DatamodelsIndexBasedNQPipeline:
    
    def __init__(self, config: DatamodelIndexBasedConfig, soft_test_flag=False, hard_test_flag=False) -> None:

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

        
        if not hard_test_flag and not soft_test_flag:
            self._verify_repo_structure()
            self.set_collections_index()
            self.set_dataframes(config.train_set_path, config.test_set_path)
        elif not hard_test_flag:
            self.set_dataframes(config.train_set_path, config.test_set_path)
        else:
            pass
    
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
            self.train_set = pl.read_ipc(train_set_path, memory_map=False)
        else:
            raise ValueError(f"Unsupported file format: {train_set_path}")
        
        if test_set_path.endswith(".csv"):
            self.test_set = pl.read_csv(test_set_path)
        elif test_set_path.endswith(".feather"):
            self.test_set = pl.read_ipc(test_set_path, memory_map=False)
        else:
            raise ValueError(f"Unsupported file format: {test_set_path}")



        print("Loaded train set")

    def _validate_create_collection_args(self, mode: str, start_idx: int, end_idx: int | None, checkpoint: int | None):
        """Private helper to validate arguments and load pre_collections.

        Returns tuple (pre_collections_path, pre_collections_polars_df)
        May raise AssertionError or ValueError on invalid inputs.
        """
        # basic mode and repo structure checks
        if mode not in ["train", "test"]:
            raise ValueError("Mode must be 'train' or 'test'")

        pre_collections_path = f"{self.datamodels_path}/pre_collections/{mode}"
        assert os.path.exists(pre_collections_path), f"Pre-collections path not found: {pre_collections_path}"

        # Read all pre-collections
        feather_files = Path(pre_collections_path).glob('*.feather')
        dfs = [pl.read_ipc(file, memory_map=False) for file in feather_files]
        if len(dfs) == 0:
            raise AssertionError(f"No pre-collection files found in {pre_collections_path}")

        pre_collections = pl.concat(dfs, how='vertical').sort(["collection_idx", "test_idx"])

        # Verify if pre-collections are not empty
        if len(pre_collections) == 0:
            raise AssertionError("Concatenated pre-collections is empty")

        # clamp end_idx
        if end_idx is not None and end_idx > len(pre_collections):
            print(f"Setting idx to max {len(pre_collections)}")
            end_idx = len(pre_collections)

        # validate checkpoint
        effective_end = end_idx if end_idx is not None else len(pre_collections)
        effective_checkpoint = checkpoint if checkpoint is not None else (effective_end - start_idx)
        if effective_checkpoint > (effective_end - start_idx):
            raise ValueError(f"checkpoint {effective_checkpoint} is greater than the number of pre-collections to process {effective_end - start_idx}")

        ## validating expected collection size mathcing the real collection size
        if mode == "train":
            assert len(pre_collections) == (self.num_models * len(self.train_collections_idx)), f"Pre-collections size {len(pre_collections)} does not match expected size {self.num_models * len(self.train_collections_idx)}"
        elif mode == "test":
            assert len(pre_collections) == (self.num_models * len(self.test_collections_idx)), f"Pre-collections size {len(pre_collections)} does not match expected size {self.num_models * len(self.test_collections_idx)}"

        return pre_collections_path, pre_collections
    


    def create_pre_collection(
        self,
        pre_collection_pipeline: BaseLLMPreCollectionsPipeline | BatchLLMPreCollectionsPipeline
    ):
       
        pre_collection_pipeline.process()

    def create_collection(
        self,
        evaluator: BaseReferenceEvaluator|BaseUnsupervisedEvaluator,
        collection_name: str,
        mode: str = "train",
        log: bool = False,
        log_config: LogConfig | None = None,
        checkpoint: int | None = None,
        start_idx: int = 0,
        end_idx: int | None = None
    ):
        
        """
        Create a collection of evaluated model outputs from pre-computed "pre-collections" and save them
        to disk (in IPC/Feather format). The function reads pre-collection files from
        {datamodels_path}/pre_collections/{mode}, evaluates each chunk with the provided evaluator,
        and writes resulting collection chunks to {datamodels_path}/collections/{mode}/{collection_name}_{chunk_start}.feather.
        Optionally logs progress and metrics to Weights & Biases (wandb).
        Parameters
        ----------
        evaluator : BaseReferenceEvaluator | BaseUnsupervisedEvaluator
            An evaluator instance used to compute a scalar evaluation for each item in a pre-collection.
            - If BaseReferenceEvaluator: its evaluate(true_outputs, predicted_outputs) will be called.
              Both arguments are expected to be numpy-like arrays extracted from the "true_output" and
              "predicted_output" columns of the pre-collections.
            - If BaseUnsupervisedEvaluator: its evaluate(predicted_outputs, questions=...) will be called.
              The function will assemble `questions` by looking up the "question" field from the pipeline's
              self.test_set for each `test_idx` in the pre-collection chunk.
        collection_name : str
            Base name for output files. Each saved chunk will be named:
            {collection_name}_{chunk_start}.feather and stored under
            {datamodels_path}/collections/{mode}/.
        mode : str, optional
            Either "train" or "test" (default "train"). Determines which pre-collections subfolder is read
            and under which subfolder the resulting collection files are written.
        log : bool, optional
            If True, attempt to initialize wandb with log_config and log per-chunk metadata and
            evaluation vectors. When logging is enabled, wandb.finish() is called at the end.
        log_config : LogConfig | None, optional
            Required if log is True. Must contain the fields used when calling wandb.init:
            project, dir, id, name, config, tags. If wandb cannot be initialized an exception is raised.
        checkpoint : int | None, optional
            Number of pre-collection rows (i.e. examples) to process per chunk. If None, defaults to
            (end_idx - start_idx) which results in a single chunk spanning the requested range.
            Must be <= (end_idx - start_idx) or a ValueError is raised.
        start_idx : int, optional
            Starting index (inclusive) within the concatenated pre-collections to begin processing.
            Defaults to 0.
        end_idx : int | None, optional
            End index (exclusive) within the concatenated pre-collections to stop processing.
            If None, defaults to the total number of rows in the concatenated pre-collections.
        Behavior and side effects
        -------------------------
        - Expects a directory at {datamodels_path}/pre_collections/{mode} containing one or more
          .feather/.ipc files. These files are read and concatenated into a single polars DataFrame and
          sorted by ["collection_idx", "test_idx"].
        - Asserts that the pre-collections directory exists and that the concatenated DataFrame is non-empty.
          Missing path or empty pre-collections result in an AssertionError.
        - Validates checkpoint, start_idx, end_idx relationships and will clamp end_idx to the available
          number of pre-collection rows if it exceeds that length.
        - Processes the pre-collections in chunks. For each chunk:
          - Calls the evaluator according to its type to compute an evaluation vector.
          - Adds a column named "evaluation" (float) to the chunk.
          - Selects columns ["collection_idx", "test_idx", "input", "evaluation"] and writes them to
            {datamodels_path}/collections/{mode}/{collection_name}_{chunk_start}.feather using zstd compression.
          - Creates the directories {datamodels_path}/collections, {datamodels_path}/collections/train and
            {datamodels_path}/collections/test if they do not already exist.
          - If logging is enabled, logs chunk metadata and the evaluation vector to wandb.
        - After all chunks complete (or immediately if checkpoint equals the requested range), wandb.finish()
          is called when logging is enabled.
        Exceptions
        ----------
        - ValueError
            - If mode is not "train" or "test".
            - If the provided evaluator is not an instance of BaseReferenceEvaluator or BaseUnsupervisedEvaluator.
            - If checkpoint is greater than the number of items to process.
        - AssertionError
            - If the pre-collections path does not exist.
            - If concatenated pre-collections is empty.
        - Exception
            - If log is True but log_config is not provided, or if wandb initialization fails.
        Notes
        -----
        - The function writes files and mutates the filesystem; it does not return a value.
        - It expects self to expose: datamodels_path (base path string), train_set and test_set (used
          when composing questions for unsupervised evaluation), and possibly train_collections_idx / test_collections_idx
          for consistency checks (the helper validation exists in the implementation).
        - The exact shape and types of columns in the pre-collections must match what the evaluator expects:
          typically "predicted_output" and "true_output" (when present) should be array-like / convertible
          to numpy arrays, and "test_idx" must index into self.test_set.
        Returns
        -------
        None
        """
        start_time = datetime.datetime.now()

        # perform validations and prepare pre_collections, pre_collections_path and checkpoint/end_idx
        pre_collections_path, pre_collections = self._validate_create_collection_args(
            mode, start_idx, end_idx, checkpoint
        )

        if end_idx is None:
            end_idx = len(pre_collections)

        if checkpoint is None:
            checkpoint = end_idx - start_idx

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
        chunk_size = start_idx
        print(f"Overiw collection {collection_name} from {start_idx} to {end_idx} with checkpoint {checkpoint}")
        while chunk_size < end_idx:
            print(f"Starting chunk {chunk_size}")
            next_chunk = min(chunk_size+checkpoint, end_idx)
            pre_collections_chunk = pre_collections[chunk_size:next_chunk]

            ## Evaluate the pre_collections and add them to the dataframe
            if isinstance(evaluator, BaseReferenceEvaluator):
                evaluation = evaluator.evaluate(pre_collections_chunk["true_output"].to_numpy(),  pre_collections_chunk["predicted_output"].to_numpy())
            elif isinstance(evaluator, BaseUnsupervisedEvaluator):
                questions  = [self.test_set[idx_test]["question"].to_numpy().flatten()[0] for idx_test in pre_collections_chunk["test_idx"]]
                
                evaluation = evaluator.evaluate(pre_collections_chunk["predicted_output"].to_numpy(),  questions=questions)
            else:
                raise ValueError("Evaluator must be an instance of BaseReferenceEvaluator or BaseUnsupervisedEvaluator")
            
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
            collection.write_ipc(f"{self.datamodels_path}/collections/{mode}/{collection_name}_{chunk_size}.feather", compression="zstd")
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
            start_idx: int = 0,
            end_idx: int | None = None,
            checkpoint: int | None = None,
            root_dir: str = "."
                         
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
            run_id,
            start_idx,
            end_idx,
            checkpoint,
            root_dir=root_dir
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

        df = pl.concat([pl.read_ipc(file, memory_map=False) for file in collections_arr], how="vertical")



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

        pl.DataFrame(evaluations).write_ipc(f"{self.datamodels_path}/evaluations/{evaluation_id}.feather", compression="zstd")

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
