import pandas as pd
import numpy as np
from dmcr.evaluators import BaseEvaluator
import evaluate

class GenericEvaluator(BaseEvaluator):

    def __init__(self,  metric: str, key: str) -> None:
        """
        Initialize the evaluator with a specified metric and key.

        Args:
            metric (str): The name of the metric to use, e.g. "rouge", "bleu".
            key (str): The key to use for the metric, e.g. "rougeL", "bleu-4".
        """
        self.evaluator = evaluate.load(metric)
        self.key = key



        

    def evaluate(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Evaluate the quality of predictions using a specified metric.

        Args:
            y (np.ndarray): The references, where each element is a list of reference sentences.
            y_pred (np.ndarray): The predictions, where each element is a predicted sentence.

        Returns:
            np.ndarray: The scores calculated using the specified metric for each prediction-reference pair.
            The highest score for each reference pair is taken.

        Raises:
            ValueError: If the shape of predictions and references do not match.
        """

        results = []
        for pred, ref in zip(y_pred, y):
            max_result = 0
            for ref_i in ref:
                print(pred, ref_i)
                result = self.evaluator.compute(predictions=[pred], references=[ref_i])
                max_result = max(result[self.key], max_result)
            
            results.append(max_result)
        
        return np.array(results)