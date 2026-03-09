
import torch
import wandb
import os
import polars as pl
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from datetime import datetime

from dmcr.datamodels.pipeline import DatamodelsIndexBasedNQPipeline
from dmcr.datamodels.config import LogConfig
from dmcr.datamodels.models import FactoryLinearRegressor

class TrainModelsPipeline():

    def __init__(self, datamodelsPipeline: DatamodelsIndexBasedNQPipeline, model_factory: FactoryLinearRegressor) -> None:
        self.datamodels = datamodelsPipeline
        self.model_factory = model_factory


    def train_datamodels(
            self,
            collection_name: str,
            val_size: float,
            random_seed: int = 42,
            log: bool = False,
            log_config: LogConfig | None = None,
            run_id: str = "default_run_id",
            # --- New arguments for range and checkpointing ---
            start_idx: int = 0,
            end_idx: int | None = None,
            checkpoint: int | None = None,
            root_dir: str = "."
                         
    ) -> None:
        """Train multiple datamodels using the specified configuration and data collections.
        
        This method trains a sequence of linear regression models on data collections,
        with support for early stopping, checkpointing, and optional Weights & Biases logging.
        Each model is trained on a subset of data filtered by test index, and the trained
        weights and biases are saved to disk at specified checkpoint intervals.
        
        Args:
            collection_name (str): Name prefix of the collection files to load from the train folder.
                                 Files must be in .feather format and start with this name.
            epochs (int): Maximum number of training epochs per model.
            train_batches (int): Batch size for training data loader.
            val_batches (int): Batch size for validation data loader.
            val_size (float): Fraction of data to use for validation (between 0 and 1).
            lr (float): Learning rate for model optimization.
            patience (int): Number of epochs to wait for validation improvement before early stopping.
            random_seed (int, optional): Random seed for data splitting. Defaults to 42.
            log (bool, optional): Whether to enable Weights & Biases logging. Defaults to False.
            log_epochs (int, optional): Frequency of logging (every N epochs). Defaults to 1.
            log_config (LogConfig | None, optional): Configuration for W&B logging. Required if log=True.
            run_id (str, optional): Identifier for the training run, used for folder naming. 
                                  Defaults to "default_run_id".
            start_idx (int, optional): Starting model index for training. Defaults to 0.
            end_idx (int | None, optional): Ending model index (exclusive). If None, trains all models.
            checkpoint (int, optional): Save checkpoint every N models. Defaults to 1.
            
        Returns:
            None: Models are saved to disk in the format:
                  {datamodels_path}/models/{run_id}/weights_{start}_{end}.pt
                  {datamodels_path}/models/{run_id}/bias_{start}_{end}.pt
                  
        Raises:
            Exception: If no collection files are found matching the collection_name.
            Exception: If log=True but log_config is None.
            
        Notes:
            - Uses early stopping based on validation MSE
            - Supports resuming training from any start_idx
            - Automatically creates output directories if they don't exist
            - Each model is trained on data filtered by test_idx
            - Checkpoints are saved when (model_idx + 1) % checkpoint == 0 or at the last model
            - If logging is enabled, each model gets its own W&B run and checkpoints are logged as artifacts
        """
        
        run_dir = f"{root_dir}/{self.datamodels.datamodels_path}/models/{run_id}"

        ## Create run_id folder
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        
        ## Set end_idx to the total number of models if not specified
        if end_idx is None or end_idx > self.datamodels.num_models:
            end_idx = self.datamodels.num_models

        ## Set checkpoint to  "end_idx" if not specified
        if checkpoint is None:
            checkpoint = end_idx

        ## Create and verify list of files from collection
        collection_path = f"{self.datamodels.datamodels_path}/collections/train/"
        collections_arr = [os.path.join(collection_path, f) for f in os.listdir(collection_path) if f.endswith(".feather") and f.startswith(collection_name)]
        if len(collections_arr) == 0:
            raise Exception("No collections found in train folder")

        ## Load existing weights and bias if resuming a run
        stacked_weights = []
        stacked_bias = []

        df = pl.concat([pl.read_ipc(file, memory_map=False) for file in collections_arr], how="vertical")
        last_checkpoint = start_idx

        for idx in range(start_idx, end_idx):
            model = self.model_factory.create_model()
            starting_timestamp = datetime.now()
            print(f"Model {idx} training")
            
            _temp = (
                df.filter(pl.col("test_idx") == idx)
                .select(pl.col("input"), pl.col("evaluation"))
            )

            _x = _temp["input"].to_numpy()
            _y = _temp["evaluation"].to_numpy()

            dataset = torch.utils.data.TensorDataset(torch.tensor(_x, device="cpu"), torch.tensor(_y, device="cpu"))
            
            train_dt, val_dt = random_split(dataset, [float(1 - val_size), val_size], generator=torch.Generator().manual_seed(random_seed)) 
            train = torch.utils.data.DataLoader(train_dt, batch_size=len(train_dt), shuffle=True)
            val = torch.utils.data.DataLoader(val_dt, batch_size=len(val_dt), shuffle=True)

            if log:
                if log_config is None:
                    raise Exception("Please provide a log configuration.")
                
                if not os.path.exists(log_config.dir):
                    os.mkdir(log_config.dir)

                wandb.init( 
                    project = log_config.project, 
                    dir = log_config.dir, 
                    id = f"{collection_name}_{run_id}_{log_config.id}_model_{idx}", 
                    name = f"{collection_name}_{run_id}_{log_config.name}_model_{idx}",
                    config = log_config.config,
                    tags = log_config.tags,
                    resume="allow"
                )

         
            x_train, y_train = next(iter(train))
            x_val, y_val = next(iter(val))

            model.train(x_train, y_train)
            val_mse = model.evaluate(x_val, y_val)

            if log:
                wandb.log({"val_mse": val_mse, "total_time": (datetime.now() - starting_timestamp).total_seconds()})
            stacked_weights.append(model.get_weights())
            stacked_bias.append(model.get_bias())

            # --- Checkpoint Saving Logic ---
            is_checkpoint = (idx + 1) % checkpoint == 0
            is_last_model = (idx + 1) == end_idx

            if is_checkpoint or is_last_model:
                print(f"Checkpointing at model index {idx}. Saving models to {run_dir}")
                torch.save(torch.tensor(stacked_weights), f"{run_dir}/{last_checkpoint}_{idx}_weights.pt")
                torch.save(torch.tensor(stacked_bias), f"{run_dir}/{last_checkpoint}_{idx}_bias.pt")
                last_checkpoint = idx+1
                stacked_weights = []
                stacked_bias = []
            if log:
                wandb.finish()
