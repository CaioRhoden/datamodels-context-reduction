import torch
import wandb
import os
import polars as pl
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

from dmcr.datamodels.pipeline import DatamodelsIndexBasedNQPipeline
from dmcr.datamodels.models import LinearRegressor
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
                         
    ) -> None:
            
            

            ## Create dirs
            if not os.path.exists(f"{self.datamodels.datamodels_path}/models"):
                os.mkdir(f"{self.datamodels.datamodels_path}/models")

            ## Create run_id folder
            if not os.path.exists(f"{self.datamodels.datamodels_path}/models/{run_id}"):
                os.mkdir(f"{self.datamodels.datamodels_path}/models/{run_id}")
            
            ## Create and verify list of files from collection
            collection_path = f"{self.datamodels.datamodels_path}/collections/train/"
            collections_arr = [os.path.join(collection_path, f) for f in os.listdir(collection_path) if f.endswith(".feather") and f.startswith(collection_name)]
            if len(collections_arr) == 0:
                raise Exception("No collections found in train folder")

            ## Initialize place to save weights and bias
            stacked_weights = torch.tensor([], device=self.model_factory.device)
            stacked_bias = torch.tensor([], device=self.model_factory.device)

            df = pl.concat([pl.read_ipc(file, memory_map=False) for file in collections_arr], how="vertical")

            for idx in range(self.datamodels.num_models):
                model = self.model_factory.create_model()
                print(f"Model {idx} training")
                
                _temp = (
                    df.filter(pl.col("test_idx") == idx)
                    .select(pl.col("input"), pl.col("evaluation"))
                )

                _x = _temp["input"].to_numpy()
                _y = _temp["evaluation"].to_numpy()

                dataset = torch.utils.data.TensorDataset(torch.tensor(_x, device=model.device), torch.tensor(_y, device=model.device))
                
                ## Random Sampling
                train_dt, val_dt = random_split(dataset, [float(1 - val_size), val_size], generator=torch.Generator().manual_seed(random_seed)) 
                train = torch.utils.data.DataLoader(train_dt, batch_size=train_batches, shuffle=True)
                val = torch.utils.data.DataLoader(val_dt, batch_size=val_batches, shuffle=True)

                # ## Model Creation
                # model = LinearRegressor(len(dataset[0][0]), 1)
                

                ## Earlt Stopping Config
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
                        tags = log_config.tags
                    )

                for epoch in range(epochs):

                    # Shuffle indexes        
                    total_loss = 0
                    total_mse = 0
                    for x_train_batch, y_train_batch in train:

                        x_train_batch, y_train_batch = x_train_batch.to(model.device).to(dtype=torch.float32), y_train_batch.to(model.device).to(dtype=torch.float32)

                        # Apply the mask to the weights
                        y_pred = model.forward(x_train_batch).squeeze() # Add batch dimension

                        # Compute loss
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

                torch.save(stacked_weights, f"{self.datamodels.datamodels_path}/models/{run_id}/weights.pt")
                torch.save(stacked_bias, f"{self.datamodels.datamodels_path}/models/{run_id}/bias.pt")

                if log:
                    artifact = wandb.Artifact(name=f"model_{run_id}", type="file")
                    artifact.add_file(f"{self.datamodels.datamodels_path}/models/{run_id}/weights.pt")
                    artifact.add_file(f"{self.datamodels.datamodels_path}/models/{run_id}/bias.pt")
                    wandb.log_artifact(artifact)
                    wandb.finish()