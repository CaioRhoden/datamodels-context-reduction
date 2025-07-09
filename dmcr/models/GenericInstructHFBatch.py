from abc import ABC, abstractmethod
from typing import Any
from dmcr.models.BatchModel import BatchModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from transformers import BitsAndBytesConfig
import torch
from accelerate import Accelerator



class GenericInstructHFBatch(BatchModel):

    def __init__(self, path: str, quantization=False) -> None:
        
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.accelerator = Accelerator()

        if not quantization:
            self.model = AutoModelForCausalLM.from_pretrained(
                    path, 
                    device_map={"": self.accelerator.process_index},
                    torch_dtype=torch.bfloat16, 
                    attn_implementation="flash_attention_2",

                )
        
        else:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                    path, 
                    device_map={"": self.accelerator.process_index},
                    torch_dtype=torch.bfloat16,
                    quantization_config=quantization_config
                )
    

    def run(self, prompts: list[str],  instruction: str, config_params: dict) -> list[Any]:
        """
        Generate text for a list of prompts based on a given instruction.

        This method utilizes a text-generation pipeline to produce outputs for each prompt in the list.
        """
        messages = [[
            {"role": "system", "content": instruction},
            {"role": "user", "content": p},
        ]for p in prompts]

        pipe = pipeline("text-generation",
                        model = self.model,
                        tokenizer = self.tokenizer,
                        return_full_text=False,
                        eos_token_id=self.tokenizer.eos_token_id,
                        
         
                )
        
        output = pipe(messages, **config_params)

        if isinstance(output, list):
            return output

        raise Exception("Output structure not as expected")
        

    