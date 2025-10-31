from abc import ABC, abstractmethod
from typing import Any, List, Dict # Assuming this is your abstract base class
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from dmcr.models.BatchModel import BatchModel 

class GenericVLLMBatch(BatchModel):
    """
    An instruction-following batch processing model powered by the vLLM engine for high-throughput inference.
    
    This class replaces the Hugging Face pipeline with vLLM to leverage its optimized performance,
    including paged attention and continuous batching.
    """

    def __init__(self, 
                path: str, 
                quantization: str = None,
                thinking: bool = False, 
                vllm_kwargs: dict = {}
            ) -> None:
        """
        Initializes the vLLM engine and tokenizer.

        Args:
            path (str): The path to the Hugging Face model repository or a local directory.
            quantization (str, optional): The quantization method to use (e.g., 'awq', 'gptq'). Defaults to None.
            vllm_kwargs (dict, optional): Additional keyword arguments to pass directly to the vLLM LLM constructor,
                                           such as `tensor_parallel_size`, `gpu_memory_utilization`, etc.
        """

        self.thinking = thinking

        # The vLLM engine replaces the HF model, pipeline, and accelerator
        self.llm = LLM(
            model=path,
            quantization=quantization,
            trust_remote_code=True,  # Recommended for many models
            **vllm_kwargs
        )
        self.tokenizer = self.llm.get_tokenizer()

    def run(self, prompts: List[str], instruction: str, config_params: Dict[str, Any], chat_template_kwargs: dict = {}) -> List[List[Dict[str, str]]]:
        """
        Generate text for a list of prompts using the vLLM engine.

        Args:
            prompts (List[str]): A list of user prompts to generate responses for.
            instruction (str): The system instruction to guide the model's behavior.
            config_params (Dict[str, Any]): Generation parameters (e.g., max_new_tokens, temperature).

        Returns:
            List[List[Dict[str, str]]]: A list where each item corresponds to a prompt and contains a list
                                        with a single dictionary: [{'generated_text': '...'}].
                                        This format mimics the original HF pipeline output.
        """
        messages = [[
            {"role": "system", "content": instruction},
            {"role": "user", "content": p},
        ] for p in prompts]

        formatted_prompts = [
            self.tokenizer.apply_chat_template(
                conversation=m,
                tokenize=False,
                add_generation_prompt=True 
            ) for m in messages
        ]


        sampling_config = config_params.copy()
        
        # vLLM uses 'max_tokens' instead of 'max_new_tokens'
        if "max_new_tokens" in sampling_config:
            sampling_config["max_tokens"] = sampling_config.pop("max_new_tokens")
        
        sampling_params = SamplingParams(**sampling_config)


        outputs = self.llm.generate(formatted_prompts, sampling_params)

        results = []

        for res in outputs:
            curr_output = []
            for output in res.outputs:
                generated_text = output.text
                curr_output.append({"generated_text": generated_text})
            results.append(curr_output)
        return results