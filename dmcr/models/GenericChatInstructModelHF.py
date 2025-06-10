from dmcr.models import BaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from transformers import BitsAndBytesConfig
import torch
from accelerate import Accelerator
import os




class GenericChatInstructModelHF(BaseLLM):

    def __init__(
            self,
            path: str,
            quantization = False,
            thinking = False
        ) -> None:

        """
        Initialize the model with the given path and options.

        Args:
            path (str): The path to the model.
            quantization (bool, optional): Whether to use 8-bit quantization. Defaults to False.
            thinking (bool, optional): Whether to generate text with reasoning text prior the asnwer
                Defaults to False.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.accelerator = Accelerator()
        self.thinking = thinking

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

    
    def run(self, prompt: str,  instruction: str, config_params: dict):

        """
        Generate text based on a given instruction and input prompt.

        The instruction will be used as the system message and the prompt as the user message
        when calling the tokenizer apply_chat_template function. The generated text will be
        returned as a string. If the "thinking" is set to true will be a a tag <think> before the
        real answer

        Parameters:
        prompt (str): The input given by the user.
        instruction (str): The instruction given to the model.
        config_params (dict): A dictionary containing the configuration parameters for the model.

        Returns:
        str: The generated text.
        """
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ]

        formatted_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking
        )


        pipe = pipeline("text-generation",
                        model = self.model,
                        tokenizer = self.tokenizer,
                        return_full_text=False,
                        eos_token_id=self.tokenizer.eos_token_id,
                        
         
                )
        
        output = pipe(formatted_input, **config_params)
        return output
    
    def delete_model(self):
        self.accelerator.wait_for_everyone()
        self.accelerator.free_memory()