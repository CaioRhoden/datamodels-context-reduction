from abc import ABC, abstractmethod
import torch
from src.datamodels.models import LinearRegressor
from src.datamodels.evaluator.BaseDatamodelsEvaluator import BaseDatamodelsEvaluator


class LinearRegressorEvaluator(BaseDatamodelsEvaluator):

    def __init__(
            self , 
            weights_arr: torch.Tensor, 
            bias_arr: torch.Tensor, 
            targets: torch.Tensor,
            inputs: torch.Tensor, 
            metric: str, 
            device: torch.device | str = "cuda",
        ) -> None:
        

        self.device = device
        self.weights_arr = weights_arr
        self.bias_arr = bias_arr
        self.targets = targets
        self.inputs = inputs
        self.metric = metric

    def evaluate(self, model_idx: int, interval: int) -> float:

        """
        Evaluate the linear regression model on a specified subset of the test set.

        This method selects a subset of the test set, based on the provided model index
        and interval, and evaluates the performance of the linear regression model using
        the specified metric.

        Args:
            model_idx (int): The index of the model to evaluate.
            interval (int): The interval to select the test set samples.

        Returns:
            float: The calculated metric value indicating the model's performance.

        Raises:
            ValueError: If the specified metric is not recognized.

        The evaluation process involves:
        - Selecting a subset of the test set using the model index and interval.
        - Making predictions using the linear regression model.
        - Calculating the specified metric (e.g., MAPE) to assess model performance.
        """

        if torch.cuda.is_available():
            torch.set_default_device(self.device)
        


        self.model = LinearRegressor(self.weights_arr[model_idx], self.bias_arr[model_idx])
        
        selected_index = []


        ## Get Test Set
        curr_idx = model_idx
        while curr_idx < self.inputs.size(0):
            selected_index.append(curr_idx)
            curr_idx += interval

        input_subset = self.inputs[torch.tensor(selected_index)]
        target_subset = self.targets[torch.tensor(selected_index)].view(-1)
        
        
        ## Get Preds

        with torch.no_grad():
            preds = torch.zeros(len(target_subset))
            for idx in range(0, len(target_subset)):
                preds[idx] = self.model.forward(input_subset[idx])

            print(target_subset[0])
            print(preds[0])

        ## Calculate Metrics

        match self.metric:

            ## Calculate MAPE
            case "mse":

                result = torch.mean((target_subset - preds) ** 2)
            
            ## Not identified metric
            case _:
                raise ValueError(f"Invalid metric: {self.metric}")

        
        ## Result
        return result.item()


        

    def batch_evaluate(self, models_idx: list[int] | str, interval: int) -> torch.Tensor:
        
        return torch.tensor([self.evaluate(model_idx, interval) for model_idx in models_idx])