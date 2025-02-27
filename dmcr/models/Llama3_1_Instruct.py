from dmcr.models import BaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from transformers import BitsAndBytesConfig
import torch
from accelerate import Accelerator
import os




class Llama3_1_Instruct(BaseLLM):

    def __init__(
            self,
            path = "../../../models/llms/Llama-3.1-8B-Instruct",
        ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.accelerator = Accelerator()
        self.model = AutoModelForCausalLM.from_pretrained(
                path, 
                device_map={"": self.accelerator.process_index},
                torch_dtype=torch.bfloat16,

            )

    
    def run(self, input: str) -> str:

        messages = [
            {"role": "system", "content": "You have to complete the desired task from the user that will be passed with some examples. Be objective, without explanations"},
            {"role": "user", "content": input},
        ]

        pipe = pipeline("text-generation",
                        model = self.model,
                        tokenizer = self.tokenizer,
                        return_full_text=False,
                         eos_token_id=self.tokenizer.eos_token_id
                        
         
                )
        try:
            output = pipe(input, max_new_tokens=20, )
            result = output[0]["generated_text"]
        except:
            raise Exception("Output structure not as expected")
        

        return result
    

    def pipe(self, temperature: float = 0.7, max_length = 2048) -> HuggingFacePipeline:

        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, temperature=temperature, max_length=max_length)
        return HuggingFacePipeline(pipeline=pipe)

