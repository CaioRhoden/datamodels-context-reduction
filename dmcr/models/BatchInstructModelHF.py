from dmcr.models import BaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from transformers import BitsAndBytesConfig
import torch
from accelerate import Accelerator
import os




class BatchModelHF(BaseLLM):

    def __init__(
            self,
            path: str,
            quantization = False,
            attention_implementation = "sdpa"
        ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.accelerator = Accelerator()

        if not quantization:
            self.model = AutoModelForCausalLM.from_pretrained(
                    path, 
                    device_map={"": self.accelerator.process_index},
                    torch_dtype=torch.bfloat16, 
                    attn_implementation=attention_implementation,

                )
        
        else:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                    path, 
                    device_map={"": self.accelerator.process_index},
                    torch_dtype=torch.bfloat16,
                    quantization_config=quantization_config
                )

    
    def run(self, prompt: str,  instruction: str, config_params: dict) -> str:

        messages = [
            {"role": "system", "content": instruction},
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
    
    def delete_model(self):
        self.accelerator.wait_for_everyone()
        self.accelerator.free_memory()