import pandas as pd
import numpy as np
from src.evaluator import BaseEvaluator
import evaluate

class Rouge_L_evaluator(BaseEvaluator):

    def __init__(self) -> None:
        pass

    def evaluate(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        rouge_l = evaluate.load("rouge")

        if y.shape != y_pred.shape:
            raise ValueError("The shape of predictions and references must be the same.")
        
        # Calculate Rouge-L for each pair of sentences
        rouge_l_scores = []
        for pred, ref in zip(y_pred, y):
            count += 1
            result = rouge_l.compute(predictions=[pred], references=[ref], use_stemmer=True)
            rouge_l_scores.append(result["rougeL"])

        
        return np.array(rouge_l_scores)