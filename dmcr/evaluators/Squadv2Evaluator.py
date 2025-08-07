import pandas as pd
import numpy as np
from dmcr.evaluators import BaseReferenceEvaluator
import evaluate
import ast
import random
class SquadV2Evaluator(BaseReferenceEvaluator):

    def __init__(self, squadv2_key: str) -> None:
        self.rouge_l = evaluate.load("squad_v2")
        self.key = squadv2_key

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
            formatted_predictions = [{"id": "id", "prediction_text": str(pred), "no_answer_probability": 0.0}]
            formatted_references = [{"answers": {"text": [], "answer_start": []}, "id": "id"}]
            for ref_i in ref:
                formatted_references[0]["answers"]["text"].append(str(ref_i))
                formatted_references[0]["answers"]["answer_start"].append(0)

            result = self.rouge_l.compute(predictions=formatted_predictions, references=formatted_references)
            max_result = max(result[self.key]/100, max_result)
            
            results.append(max_result)
        
        return np.array(results)