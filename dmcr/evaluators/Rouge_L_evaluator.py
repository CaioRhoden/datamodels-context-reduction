import pandas as pd
import numpy as np
from dmcr.evaluators import BaseReferenceEvaluator
import evaluate
import ast
import uuid

class Rouge_L_evaluator(BaseReferenceEvaluator):

    def __init__(self) -> None:
        self.rouge_l = evaluate.load("rouge", experiment_id=str(uuid.uuid4()))

    def evaluate(self, y: np.ndarray[list[str]], y_pred: np.ndarray[list[str]]) -> np.ndarray:
        """
        Evaluate predictions using the ROUGE-L metric.

        Args:
            y (np.ndarray): Iterable of reference(s) for each sample.
            y_pred (np.ndarray): Array with the list of predicted strings.

        Returns:
            np.ndarray: 1-D array of length n_samples. Each element is the maximum ROUGE-L
            F1 score (range [0, 1]) computed between the prediction and that sample's
            reference(s). If a sample has multiple references, the best (highest) score
            across references is returned.

        Raises:
            ValueError: If len(y) != len(y_pred).

        Notes:
            This method uses the `evaluate` library's ROUGE implementation and specifically
            returns the 'rougeL' (F1) score. The caller is responsible for ensuring that
            inputs are text strings or iterables of text strings.
        """

        
        # Calculate Rouge-L for each pair of sentences
        results = []
        for pred, ref in zip(y_pred, y):
            for pred_i in pred:
                preds = []
                max_result = 0
                for ref_i in ref:
                    result = self.rouge_l.compute(predictions=[pred_i], references=[ref_i])
                    max_result = max(result["rougeL"], max_result)
                preds.append(max_result)
            
            results.append(np.mean(preds))
        
        return np.array(results, dtype=np.float64)