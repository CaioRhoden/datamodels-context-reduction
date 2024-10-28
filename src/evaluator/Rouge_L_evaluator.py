import pandas as pd
import numpy as np
from src.evaluator import BaseEvaluator
import evaluate
import ast

class Rouge_L_evaluator(BaseEvaluator):

    def __init__(self) -> None:
        self.rouge_l = evaluate.load("rouge")

    def evaluate(self, y: np.ndarray, y_pred: np.ndarray, possible_outputs: None | np.ndarray) -> np.ndarray:


        if y.shape != y_pred.shape:
            raise ValueError("The shape of predictions and references must be the same.")
        
        # Calculate Rouge-L for each pair of sentences
        rouge_l_scores = []

        if possible_outputs is None:
            for pred, ref in zip(y_pred, y):
                result = self.rouge_l.compute(predictions=[pred], references=[ref], use_stemmer=True)
                rouge_l_scores.append(result["rougeL"])

        else:
            for pred, ref, poss in zip(y_pred, y, possible_outputs):
                result = self.rouge_l.compute(predictions=[pred], references=[ref], use_stemmer=True)
                if type(poss) is not float:
                    poss = ast.literal_eval(poss)
                    for i in range(len(poss)):
                        candidate = self.rouge_l.compute(predictions=[pred], references=[poss[i]], use_stemmer=True)
                        if candidate["rougeL"] > result["rougeL"]:
                            result = candidate
                rouge_l_scores.append(result["rougeL"])

        
        return np.array(rouge_l_scores)