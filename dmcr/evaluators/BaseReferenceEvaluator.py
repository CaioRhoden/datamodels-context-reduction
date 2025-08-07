import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class BaseReferenceEvaluator(ABC):

    @abstractmethod
    def __init__(self) -> None:
        pass
    @abstractmethod
    def evaluate(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass