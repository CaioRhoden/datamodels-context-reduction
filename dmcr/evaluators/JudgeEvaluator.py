import numpy as np
import os

from dmcr.evaluators import BaseUnsupervisedvaluator


class JudgeEvaluator(BaseUnsupervisedvaluator):
    """
    Base class for unsupervised evaluators that judge the quality of predictions.
    
    This class provides a basic structure for implementing unsupervised evaluators that
    assess the quality of predictions without requiring reference data.
    """

    def __init__(self, model_path: str) -> None:
        super().__init__()
        if os.path.exists(model_path):
            self.model_path = model_path
        else:
            raise ValueError(f"Model path {model_path} for LLM as judge evaluatordoes not exist.")
        
    def evaluate(self, y: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Evaluate the data using an unsupervised approach.

        Args:
            y (np.ndarray): The input data to be evaluated.
            *args: Additional arguments for the evaluation.
            **kwargs: Keyword arguments for the evaluation.

        Returns:
            np.ndarray: The evaluation results as an array.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")