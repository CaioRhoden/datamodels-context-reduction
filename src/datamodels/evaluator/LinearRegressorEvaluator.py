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
            metric: str
        ) -> None:
        
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
        
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


        self.model = LinearRegressor(self.weights_arr[model_idx], self.bias_arr[model_idx])
        
        selected_index = []
        
        ## Get Test Set
        curr_idx = model_idx
        while curr_idx < self.test_set.size(0):
            selected_index.append(curr_idx)
            curr_idx += interval
        test_subset = self.test_set[torch.tensor(selected_index)]
        
        ## Get Preds

        with torch.no_grad():
            preds = torch.zeros(len(test_subset))
            for idx in range(0, len(test_subset)):
                preds[idx] = self.model(test_subset[idx][0])

        ## Calculate Metrics

        match self.metric:

            ## Calculate MAPE
            case "mape":
                targets = test_subset[:, 1]
                epsilon = 1e-8
                targets = torch.where(targets == 0, torch.tensor(epsilon), targets)

                result = torch.mean(torch.abs((preds - targets) / torch.abs(targets))) * 100
            
            ## Not identified metric
            case _:
                raise ValueError(f"Invalid metric: {self.metric}")

        
        ## Result
        return result.item()


        

    # def batch_evaluate(self, models_idx: list[int] | str) -> torch.Tensor:
    #     pass