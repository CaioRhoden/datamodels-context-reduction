import pandas as pd
import numpy as np
from dmcr.evaluators import BaseEvaluator
import evaluate
import ast

class Rouge_L_evaluator(BaseEvaluator):

    def __init__(self) -> None:
        self.rouge_l = evaluate.load("rouge")

    def evaluate(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Evaluate the quality of predictions using the Rouge-L metric.

        Args:
            y (np.ndarray): The references, each element being a list of reference sentences.
            y_pred (np.ndarray): The predictions, each element being a predicted sentence.

        Returns:
            np.ndarray: The Rouge-L scores, one for each prediction-reference pair, where each 
            reference pair is evaluated individually and the highest score is taken.

        Raises:
            ValueError: If the shape of predictions and references do not match.
        """

        
        # Calculate Rouge-L for each pair of sentences
        results = []
        for pred, ref in zip(y_pred, y):
            max_result = 0
            for ref_i in ref:
                result = self.rouge_l.compute(predictions=[pred], references=[ref_i])
                max_result = max(result["rougeL"], max_result)
            
            results.append(max_result)
        
        return np.array(results)