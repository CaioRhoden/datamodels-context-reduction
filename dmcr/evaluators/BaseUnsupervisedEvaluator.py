from abc import ABC, abstractmethod
import numpy as np


class BaseUnsupervisedvaluator(ABC):
    """
    Base class for unsupervised evaluators.

    This class provides a basic structure for implementing unsupervised evaluators.
    It defines the interface for the evaluate method, which should be implemented
    by subclasses.

    Attributes:
        None

    Methods:
        evaluate: Evaluates the data using an unsupervised approach.
    """


    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
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

