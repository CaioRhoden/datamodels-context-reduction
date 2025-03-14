import pandas as pd
import numpy as np
from dmcr.evaluator import BaseEvaluator
import evaluate
import ast

class GleuEvaluator(BaseEvaluator):

    def __init__(self) -> None:
        """
        Initialize the Google BLEU evaluator.

        The evaluator is used to compute the quality of predictions using the Google BLEU metric.
        The metric is a variant of the BLEU metric that is more suitable for evaluating sentences.
        """
        self.gleu = evaluate.load("google_bleu")

    def evaluate(self, y: np.ndarray, y_pred: np.ndarray, possible_outputs: None | np.ndarray) -> np.ndarray:
        """
        Evaluate the quality of predictions using the Google BLEU metric.

        Args:
        y (np.ndarray): The references
        y_pred (np.ndarray): The predictions
        possible_outputs (None | np.ndarray): The possible outputs for each reference (default is None)

        Returns:
        np.ndarray: The Google BLEU scores, one for each pair of prediction and reference
        """
        if y.shape != y_pred.shape:
            raise ValueError("The shape of predictions and references must be the same.")
        
        # Calculate Rouge-L for each pair of sentences
        results = []
        if possible_outputs is None:
            for pred, ref in zip(y_pred, y):
                result = self.gleu.compute(predictions=[pred], references=[ref])
                if result is not None:
                    results.append(result["google_bleu"])
                else:
                    raise ValueError("Google BLEU cannot be computed.")
        else:
            for pred, ref, poss in zip(y_pred, y, possible_outputs):
                result = self.gleu.compute(predictions=[pred], references=[ref])
                if type(poss) is not float:
                    poss = ast.literal_eval(poss)

                    try:
                        possible_preds = [self.gleu.compute(predictions=[pred], references=[p])["google_bleu"] for p in poss] # type: ignore
                    except:
                        raise ValueError("Google BLEU cannot be computed for possible outputs")
                    
                    if result is not None:
                         results.append(max(result["google_bleu"],
                                        *possible_preds))
                    else:
                        raise ValueError("Google BLEU cannot be computed.")
                   
                elif result is not None:
                    results.append(result["google_bleu"])

                else:
                    raise ValueError("Google BLEU cannot be computed.")
        
        return np.array(results)
