from src.llms import BaseLLM
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
            path = "../../models/caio.rhoden/Meta-Llama-3.1-8B",
        ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        accelerator = Accelerator()
        self.model = AutoModelForCausalLM.from_pretrained(
                path, 
                quantization_config=BitsAndBytesConfig(load_in_4bit=True), 
                device_map={"": accelerator.process_index},
                torch_dtype=torch.float16,
                force_download=True,

            )

    
    def run(self, input: str) -> str:

        input_ids = self.tokenizer(input, return_tensors="pt").input_ids

        output = self.model.generate(input_ids, 
                                     max_new_tokens=25, 
                                     num_return_sequences=1,
                                     eos_token_id= self.tokenizer.eos_token_id,
                                     pad_token_id = self.tokenizer.eos_token_id,
                                     temperature=0.3
                                    )
        output = [tok_out[len(tok_in):] for tok_in, tok_out in zip(input_ids, output)] 
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def pipe(self, temperature: float = 0.7, max_length = 1024) -> HuggingFacePipeline:

        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, temperature=temperature, max_length=max_length)
        return HuggingFacePipeline(pipeline=pipe)

