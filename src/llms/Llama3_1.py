from src.llms import BaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from transformers import BitsAndBytesConfig






class Llama3_1(BaseLLM):

    def __init__(
            self,
            path = "../models/inmetrics/Meta-Llama-3.1-8B-Instruct",
        ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B", quantization_config=BitsAndBytesConfig(load_in_4bit=True))

    
    def run(self, input: str) -> str:

        input_ids = self.tokenizer(input).input_ids
        output = self.model.generate(input_ids, max_new_tokens=15, early_stopping=True, )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def pipe(self, temperature: float = 0.7, max_length = 1024) -> HuggingFacePipeline:

        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, temperature=temperature, max_length=max_length)
        return HuggingFacePipeline(pipeline=pipe)

