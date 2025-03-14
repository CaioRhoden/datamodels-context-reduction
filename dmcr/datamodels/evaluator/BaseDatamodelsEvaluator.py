from abc import ABC, abstractmethod
import torch


class BaseDatamodelsEvaluator(ABC):

    def __init__(self , weights_arr: torch.Tensor, bias_arr: torch.Tensor, test_set: torch.Tensor, interval: int) -> None:
        self.weights_arr = weights_arr
        self.bias_arr = bias_arr
        self.test_set = test_set
        self.interval = interval

    @abstractmethod
    def evaluate(self, model_idx: int, interval: int) -> float:
        pass

    @abstractmethod
    def batch_evaluate(self, models_idx: list[int] | str) -> list[float]:
        pass