import torch
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer,  pipeline
from scipy.spatial.distance import cosine
import datetime
import pandas as pd
import re
import numpy as np

class Sampler:
    def __init__(self, access_token: str, model_id: str, is_with_class: bool) -> None:
        """
        Initializes a new instance of the Sampler class.

        Parameters:
            access_token (str): The access token for the sampler.
            model_kwargs (dict): The keyword arguments for the model.

        Returns:
            None
        """
        self.access_token = access_token
        self.model_kwargs = {"torch_dtype": torch.bfloat16}
        self.model_id = model_id
        self.train_data = pd.read_pickle("data/pool.pickle").reset_index()
        if is_with_class:
            self.test_data = pd.read_pickle("data/class_distances.pickle").reset_index()
        else:
            self.test_data = pd.read_pickle("data/distances.pickle").reset_index()




    def generate_prompt(self, examples, suffix = None, prefix = None, input_variables = None, prompt_template = None) -> None:
        """
        Generate a prompt for few-shot learning based on the given examples.

        Args:
            examples (List[Dict[str, str]]): A list of dictionaries representing the examples. Each dictionary
                should have the keys "input" and "output" representing the input and output of the example.
            suffix (str, optional): The suffix to append to the prompt. Defaults to None.
            prefix (str, optional): The prefix to prepend to the prompt. Defaults to None.
            input_variables (List[str], optional): The input variables to include in the prompt. Defaults to None.
            prompt_template (PromptTemplate, optional): The template to use for the prompt. Defaults to None.

        Returns:
            None: This function does not return anything.

        Raises:
            None

        Examples:
            >>> sampler = Sampler()
            >>> examples = [
            ...     {"input": "What is the capital of France?", "output": "Paris"},
            ...     {"input": "What is the capital of Germany?", "output": "Berlin"}
            ... ]
            >>> sampler.generate_prompt(examples)
            Prompt created successfully.
        """

        if prompt_template is None:
            prompt_template = PromptTemplate(
            input_variables=["input", "output"], template="Example Input: {input}\n Example Output: {output}"
            )
        
        suffix = "Input: {input}\n Output:" if suffix is None  else suffix
        prefix = "Fill the expected Output\n" if prefix is None  else prefix
        input_variables = ["input"] if input_variables is None  else input_variables


        prompt = FewShotPromptTemplate(
            examples= examples,
            example_prompt= prompt_template,
            suffix= suffix,
            prefix= prefix,
            input_variables= input_variables
        )

        return prompt

    
    def create_llm(self) -> None:
        """
        Initializes a new instance of the HuggingFacePipeline class and assigns it to the 'llm' attribute.

        This function creates a new instance of the HuggingFacePipeline class using the 'pipeline' attribute as the input. The 'pipeline' attribute is expected to be an instance of the pipeline class from the transformers library.

        Parameters:
            None

        Returns:
            None
        """
        pipe = HuggingFacePipeline.from_model_id(
            model_id=self.model_id,
            task="text-generation",
            device_map="auto",
            model_kwargs=self.model_kwargs,
            pipeline_kwargs={"max_new_tokens": 25},
        )
        return pipe
    
    def sample_data(self, df, k_samples, idx) -> pd.DataFrame:
        pass

    def run(self, k) -> None:

        results = {
        "k": [],
        "task": [],
        "input": [],
        "output": [],
        "predicted_output": [],
        "possible_outputs": []
        }

        pattern = r'\n Output:[^\n]*'
        for i in range(0, len(self.test_data)):
            print(f"Iteration {i}")
            print(datetime.datetime.now())
            examples =  self.sample_data(self.train_data, k, i)
            print(len(examples))
            prompt = self.generate_prompt(examples)
            print(prompt)
            llm = self.create_llm()
            print(llm)
            chain = prompt | llm


            with torch.no_grad():
                predicted_output =  re.search(pattern, chain.invoke({"input": self.test_data.loc[i]["input"]})).group()
            results["k"].append(k)
            results["task"].append( self.test_data.loc[i]["task"])
            results["input"].append( self.test_data.loc[i]["input"])
            results["output"].append( self.test_data.loc[i]["output"])
            results["predicted_output"].append(predicted_output)
            results["possible_outputs"].append(self.test_data.loc[i]["possible_outputs"])
            break

        return pd.DataFrame(results)

class RandomSampler(Sampler):

    def sample_data(self, df, k_samples, idx) -> pd.DataFrame:

        selected_df = df.loc[df["task"] == self.test_data.loc[idx]["task"]].sample(n=k_samples, replace=False).reset_index(drop=True)

        samples = []
        for i in range(0,k_samples):

            sample = {
                "input": selected_df.loc[i]["input"],
                "output": selected_df.loc[i]["output"]
            }

            samples.append(sample)

        return samples
    
class KateSampler(Sampler):

    
    def _k_smallest(self, arr, k):
        if k <= 0:
            return np.array([])  # Return an empty array if k is not positive
        k = min(k, len(arr))  # Ensure k does not exceed the length of the array
        indices = np.argpartition(arr, k-1)[:k]
        return indices[np.argsort(arr[indices])]

    def sample_data(self, df, k_samples, idx) -> pd.DataFrame:

        small_idx = self._k_smallest(self.test_data.loc[idx]["distances"], k_samples)

        samples = []
        for i in range(0, k_samples):

            sample = {
                "input": self.train_data.loc[small_idx[i]]["input"],
                "output": self.train_data.loc[small_idx[i]]["output"]
            }

            samples.append(sample)

        return samples



    
    

        


