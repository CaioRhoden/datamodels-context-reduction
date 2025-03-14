from dmcr.models import BaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from transformers import BitsAndBytesConfig
import torch
from accelerate import Accelerator
import os




class Llama3_1(BaseLLM):

    def __init__(
            self,
            path = "../../models/llms/Llama-3.1-8B",
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


        pipe = pipeline("text-generation",
                        model = self.model,
                        tokenizer = self.tokenizer,
                        
         
                )
        output = pipe(input, max_new_tokens=15, )

        return output[0]["generated_text"]

    def pipe(self, temperature: float = 0.7, max_length = 1024) -> HuggingFacePipeline:

        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, temperature=temperature, max_length=max_length)
        return HuggingFacePipeline(pipeline=pipe)

