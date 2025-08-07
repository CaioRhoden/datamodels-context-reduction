from typing import Callable
import os
import re
import numpy as np


from dmcr.evaluators import BaseUnsupervisedEvaluator
from dmcr.models import GenericInstructModelHF


class JudgeEvaluator(BaseUnsupervisedEvaluator):
    """
    Base class for unsupervised evaluators that judge the quality of predictions.
    
    This class provides a basic structure for implementing unsupervised evaluators that
    assess the quality of predictions without requiring reference data.
    """

    def __init__(self, model_path: str, model_configs: dict, instruction: str, format_template: Callable[[str, str], str], regex_pattern: str= r'Rating: \[\[(\d+)\]\]') -> None:
        super().__init__()

        if os.path.exists(model_path):
            self.model_path = model_path
        else:
            raise ValueError(f"Model path {model_path} for LLM as judge evaluatordoes not exist.")
        
        self.model_configs = model_configs
        self.instruction = instruction
        self.format_template = format_template
        self.regex_pattern = regex_pattern
        
    def evaluate(self, y: np.ndarray, questions:np.ndarray) -> np.ndarray:
        """
        Evaluate the data using an unsupervised approach.

        Args:
            y (np.ndarray): The input data to be evaluated.
            *args: Additional arguments for the evaluation.
            **kwargs: Keyword arguments for the evaluation.

        Returns:
            np.ndarray: The evaluation results as an array.
        """
        judge = GenericInstructModelHF(path=self.model_path)

        results = []
        for pred, question in zip(y, questions):
            judge_input = self.format_template(question, pred)
            grades = judge.run(prompt=judge_input, instruction=self.instruction, config_params=self.model_configs)
            score = self._calculate_rating_mean(grades if isinstance(grades, list) else [grades])
            results.append(score)

        return np.array(results)




    def _calculate_rating_mean(self, grades: list) -> float:
        """
        Calculate the mean rating from generated texts containing the class attribute "regex_pattern".
        
        Args:
            data: List of dictionaries with 'generated_text' keys
            
        Returns:
            Mean rating (float), or None if no ratings found
        """
        ratings = []
        
        for item in grades:
            text = item['generated_text']
            # Search for rating pattern
            match = re.search(self.regex_pattern, text)
            if match:
                ratings.append(int(match.group(1)))
        
        if not ratings:
            print("No ratings found in generated text")
            return 0
        
        return sum(ratings) / len(ratings)

        

