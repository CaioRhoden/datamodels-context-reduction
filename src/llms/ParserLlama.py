from src.llms import BaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from transformers import BitsAndBytesConfig
import torch


from langchain.output_parsers import PydanticOutputParser

from accelerate import Accelerator
import os




class ParserLlama(BaseLLM):

    def __init__(
            self,
            parser_instruction: str,
            path = "../../models/llms/Llama-3.1-8B-Instruct",
            
        ) -> None:

        self.parser_instruction = parser_instruction
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
            {"role": "system", "content": f"You have to answer the question from the user that will be passed with some examples, not all of them necessarily useful. {self.parser_instruction}"},
            {"role": "user", "content": f"Question: {input}"},
        ]

        pipe = pipeline("text-generation",
                        model = self.model,
                        tokenizer = self.tokenizer,
                        return_full_text=False,
                         eos_token_id=self.tokenizer.eos_token_id
                        
         
                )
        
        try:
            output = pipe(messages, max_new_tokens=20, )
            result = output[0]["generated_text"]
        except:
            raise Exception("Output structure not as expected")
        

        return result
    

    def pipe(self, temperature: float = 0.7, max_length = 1024) -> HuggingFacePipeline:

        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, temperature=temperature, max_length=max_length)
        return HuggingFacePipeline(pipeline=pipe)

