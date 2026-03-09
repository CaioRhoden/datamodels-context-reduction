from abc import ABC, abstractmethod


class Regressor(ABC):
    
    @abstractmethod
    def __init__(self, model_configs: dict = None):
        pass

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def get_bias(self):
        pass

    @abstractmethod
    def predict(self, x):
        pass
    
    @abstractmethod
    def predict_proba(self, x):
        pass

    @abstractmethod
    def train(self, x, y):
        pass

    @abstractmethod
    def evaluate(self, x, y, metric: str = "mse") -> float:
        pass
                