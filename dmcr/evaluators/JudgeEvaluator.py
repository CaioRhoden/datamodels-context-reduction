from typing import Callable
import os
import re
import numpy as np
from typing import Optional


from dmcr.evaluators import BaseUnsupervisedEvaluator
from dmcr.models import GenericInstructModelHF, GenericInstructBatchHF, GenericVLLMBatch


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
            thinking=False,
            judge: Optional[GenericInstructBatchHF | GenericInstructModelHF | GenericVLLMBatch] = None
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
        
        ## Initialize LM model
        if judge is not None:
            self.judge = judge
        else:
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
            pred = y[i:min(i+self.batch_size, len(y))]
            question = questions[i:min(i+self.batch_size, len(y))]

            judge_inputs = []
            mapping = []


            for pi, sample in enumerate(pred):  # sample corresponds to one top-level item in the batch slice
                pi_prompts = []
                sample = sample.tolist()
                for pj, group in enumerate(sample):  # group is expected to be iterable of generated outputs
                    question_for_sample = question[pi]
                    # Build prompts for this group (one prompt per item in group)
                    prompts = [self.format_template(question_for_sample, group)]
                    judge_inputs.extend(prompts)
                    pi_prompts.extend(prompts)
                mapping.append((pi, len(pi_prompts)))

            assert len(judge_inputs) == (len(pred) * len(pred[0])), "Mismatch in number of judge inputs constructed"

            if not judge_inputs:
                # nothing to evaluate in this batch slice; continue
                continue

            # Run the judge in one batch
            grades = self.judge.run(prompts=judge_inputs, instruction=self.instruction, config_params=self.model_configs)

            # Now reconstruct scores: grades is expected to be a list/dict entries matching judge_inputs order
            idx = 0
            for (pi, n_items) in mapping:
                slice_grades = grades[idx: (idx + n_items) ]
                idx += n_items

                score = self._calculate_rating_mean(slice_grades)
                results.append(score)



        assert len(results) == len(y), "Mismatch between number of results and input samples"
        return np.array(results, dtype=np.float64)




    def _calculate_rating_mean(self, grades: list) -> float:
        """
        Calculate the mean rating from generated texts containing the class attribute "regex_pattern".
        
        Args:
            data: List of dictionaries with 'generated_text' keys
            
        Returns:
            Mean rating (float), or None if no ratings found
        """
        ratings = []
        for pred in grades:
            for item in pred:
                text = item['generated_text']
                # Search for rating pattern
                match = re.search(self.regex_pattern, text)
                if match:
                    ratings.append(int(match.group(1)))
            
        if not ratings:
            print("No ratings found in generated text")
            return 0
    
        return (sum(ratings) / len(ratings))/10



