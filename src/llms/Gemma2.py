from src.llms import BaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline






class Gemma2(BaseLLM):

    def __init__(
            self,
            path = "../models/inmetrics/gemma-2b",
        ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path)

    
    def run(self, input: str) -> str:

        input_ids = self.tokenizer(input, return_tensors="pt").input_ids
        output = self.model.generate(input_ids, max_new_tokens=25)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def pipe(self, temperature: float = 0.7, device = "cpu", max_length = 1024) -> HuggingFacePipeline:

        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, temperature=temperature, device=device, max_length=max_length)
        return HuggingFacePipeline(pipeline=pipe)

