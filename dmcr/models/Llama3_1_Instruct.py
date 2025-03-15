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
            quantization = False,
        ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.accelerator = Accelerator()

        if not quantization:
            self.model = AutoModelForCausalLM.from_pretrained(
                    path, 
                    device_map={"": self.accelerator.process_index},
                    torch_dtype=torch.bfloat16,

                )
        
        else:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                    path, 
                    device_map={"": self.accelerator.process_index},
                    torch_dtype=torch.bfloat16,
                    quantization_config=quantization_config
                )

    
    def run(self, prompt: str,  input: str, config_params: dict) -> str:

        messages = [
            {"role": "system", "content": "You are a coding generation tool that will solve a problem using Python"},
            {"role": "user", "content": prompt},
        ]

        pipe = pipeline("text-generation",
                        model = self.model,
                        tokenizer = self.tokenizer,
                        return_full_text=False,
                        eos_token_id=self.tokenizer.eos_token_id,
                        
         
                )
        
        output = pipe(messages, **config_params)
        return output
        

        
    

    def pipe(self, temperature: float = 0.7, max_length = 2048) -> HuggingFacePipeline:

        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, temperature=temperature, max_length=max_length)
        return HuggingFacePipeline(pipeline=pipe)

