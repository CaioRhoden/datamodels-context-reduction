import pandas as pd
import numpy as np

class BaseEvaluator():

    def __init__(self) -> None:
        pass

    def evaluate(self, y: pd.DataFrame, y_pred: pd.DataFrame) -> np.ndarray:
        pass