
import torch
import wandb
import os
import polars as pl
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

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
        stacked_weights = torch.tensor([], device=self.model_factory.device)
        stacked_bias = torch.tensor([], device=self.model_factory.device)

        df = pl.concat([pl.read_ipc(file, memory_map=False) for file in collections_arr], how="vertical")
        last_checkpoint = start_idx

        for idx in range(start_idx, end_idx):
            model = self.model_factory.create_model()
            print(f"Model {idx} training")
            
            _temp = (
                df.filter(pl.col("test_idx") == idx)
                .select(pl.col("input"), pl.col("evaluation"))
            )

            _x = _temp["input"].to_numpy()
            _y = _temp["evaluation"].to_numpy()

            dataset = torch.utils.data.TensorDataset(torch.tensor(_x, device=model.device), torch.tensor(_y, device=model.device))
            
            train_dt, val_dt = random_split(dataset, [float(1 - val_size), val_size], generator=torch.Generator().manual_seed(random_seed)) 
            train = torch.utils.data.DataLoader(train_dt, batch_size=train_batches, shuffle=True)
            val = torch.utils.data.DataLoader(val_dt, batch_size=val_batches, shuffle=True)

            best_mse = float('inf')
            early_stopping_counter = 0

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

            for epoch in range(epochs):
                total_loss = 0
                total_mse = 0
                for x_train_batch, y_train_batch in train:
                    x_train_batch, y_train_batch = x_train_batch.to(model.device).to(dtype=torch.float32), y_train_batch.to(model.device).to(dtype=torch.float32)
                    y_pred = model.forward(x_train_batch).squeeze()
                    total_loss += model.optimize(y_pred, y_train_batch, lr)
                    
                for x_val_batch, y_val_batch in val:
                    total_mse += model.evaluate(x_val_batch.to(model.device).to(dtype=torch.float32), y_val_batch.to(model.device).to(dtype=torch.float32))
                
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

            model.detach()

            # --- Checkpoint Saving Logic ---
            is_checkpoint = (idx + 1) % checkpoint == 0
            is_last_model = (idx + 1) == end_idx

            if is_checkpoint or is_last_model:
                print(f"Checkpointing at model index {idx}. Saving models to {run_dir}")
                torch.save(stacked_weights, f"{run_dir}/{last_checkpoint}_{idx}_weights.pt")
                torch.save(stacked_bias, f"{run_dir}/{last_checkpoint}_{idx}_bias.pt")
                last_checkpoint = idx+1

                if log:
                    # Log a single artifact for the checkpoint file
                    artifact_name = f"model_{run_id}_to_{idx+1}"
                    artifact = wandb.Artifact(name=artifact_name, type="model_checkpoint")
                    artifact.add_file(f"{run_dir}/{last_checkpoint}_{idx}_weights.pt")
                    artifact.add_file(f"{run_dir}/{last_checkpoint}_{idx}_bias.pt")
                    wandb.log_artifact(artifact)

            if log:
                wandb.finish()
