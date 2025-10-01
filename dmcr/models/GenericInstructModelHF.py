from dmcr.models import BaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from transformers import BitsAndBytesConfig
import torch
from accelerate import Accelerator
import os




class GenericInstructModelHF(BaseLLM):

    def __init__(
            self,
            path: str,
            quantization = False,
            attn_implementation = "sdpa",
            thinking = False
        ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.accelerator = Accelerator()
        self.thinking = thinking

        if not quantization:
            self.model = AutoModelForCausalLM.from_pretrained(
                    path, 
                    device_map={"": self.accelerator.process_index},
                    dtype=torch.bfloat16, 
                    attn_implementation=attn_implementation,

                )
        
        else:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                    path, 
                    device_map={"": self.accelerator.process_index},
                    dtype=torch.bfloat16,
                    quantization_config=quantization_config
                )

    
    def run(self, prompt: str,  instruction: str, config_params: dict) -> str | list:

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
        
        if self.thinking:
            
            messages = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.thinking
            )

        output = pipe(messages, **config_params)
        return output
    
    def delete_model(self):
        self.accelerator.wait_for_everyone()
        self.accelerator.free_memory()