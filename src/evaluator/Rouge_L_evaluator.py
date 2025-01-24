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
        results = []
        if possible_outputs is None:
            for pred, ref in zip(y_pred, y):
                result = self.rouge_l.compute(predictions=[pred], references=[ref])
                if result is not None:
                    results.append(result["rougeL"])
                else:
                    raise ValueError("Google BLEU cannot be computed.")
        else:
            for pred, ref, poss in zip(y_pred, y, possible_outputs):
                result = self.rouge_l.compute(predictions=[pred], references=[ref])
                if type(poss) is not float:
                    poss = ast.literal_eval(poss)

                    try:
                        possible_preds = [self.rouge_l.compute(predictions=[pred], references=[p])["rougeL"] for p in poss] # type: ignore
                    except:
                        raise ValueError("Google BLEU cannot be computed for possible outputs")
                    
                    if result is not None:
                         results.append(max(result["rougeL"],
                                        *possible_preds))
                    else:
                        raise ValueError("Google BLEU cannot be computed.")
                   
                elif result is not None:
                    results.append(result["rougeL"])

                else:
                    raise ValueError("Google BLEU cannot be computed.")
        
        return np.array(results)