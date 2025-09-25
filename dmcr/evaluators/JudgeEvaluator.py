from typing import Callable
import os
import re
import numpy as np


from dmcr.evaluators import BaseUnsupervisedEvaluator
from dmcr.models import GenericInstructModelHF, GenericInstructBatchHF


class JudgeEvaluator(BaseUnsupervisedEvaluator):
    """
    Base class for unsupervised evaluators that judge the quality of predictions.
    
    This class provides a basic structure for implementing unsupervised evaluators that
    assess the quality of predictions without requiring reference data.
    """

    def __init__(
            self, 
            model_path: str, 
            model_configs: dict, 
            instruction: str, 
            format_template: Callable[[str, str], str], 
            regex_pattern: str= r'Rating: \[\[(\d+)\]\]', 
            attn_implementation="sdpa", 
            batch_size: int=1,
            thinking=False
        ) -> None:
        super().__init__()

        if os.path.exists(model_path):
            self.model_path = model_path
        else:
            raise ValueError(f"Model path {model_path} for LLM as judge evaluatordoes not exist.")
        
        self.model_configs = model_configs
        self.instruction = instruction
        self.format_template = format_template
        self.regex_pattern = regex_pattern
        self.attn_implementation = attn_implementation
        self.thinking = thinking
        self.batch_size = batch_size

        if self.batch_size <=  0:
            raise ValueError("Batch size must be greater than 0")

        self.judge = GenericInstructBatchHF(
            path=self.model_path, 
            attn_implementation=self.attn_implementation, 
            thinking=self.thinking, 
        )
        


        
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



        results = []
        for i in range(0, len(y), self.batch_size):
            pred_ = y[i:i+self.batch_size]
            question = questions[i:i+self.batch_size]
            judge_inputs = [self.format_template(q, p) for q, p in zip(question, pred_)]
            grades = self.judge.run(prompts=judge_inputs, instruction=self.instruction, config_params=self.model_configs)
            scores = [self._calculate_rating_mean(g if isinstance(g, list) else [g]) for g in grades]
            results.extend(scores)

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
        
        return (sum(ratings) / len(ratings))/10

        

