from dmcr.models import BaseLLM
from dmcr.pipelines import BasePipeline

import torch
from langchain.prompts import PromptTemplate
import pandas as pd
import json


class BaselinePipeline(BasePipeline):
    def __init__(self, path: str, llm: BaseLLM, instructions: str) -> None:
        self.llm = llm
        self.path = path
        self.instructions = json.load(open(instructions))

    def run(self, input: str, instruction: str) -> str:
        template = template = """
            Fill the expected Output according to the instruction
            Intruction: {instruction}

            User Input: 
            {input}

            Model Output:
        """

        prompt = PromptTemplate.from_template(template).format(
            instruction=instruction, input=input
        )
        return self.llm.run(prompt)

    def run_tests(
        self, data: pd.DataFrame, checkpoint: int, instruction: bool = True
    ) -> None:
        tasks, inputs, predicted = [], [], []

        for i in range(checkpoint, len(data)):
            print(f"Step {i} of {len(data)}")
            tasks.append(data.loc[i]["task"])
            inputs.append(data.loc[i]["input"])

            input = str(data.loc[i]["input"])

            if instruction:
                predicted.append(
                    self.run(input, self.instructions[data.loc[i]["task"]])
                )

            else:
                predicted.append(self.run(input, ""))

            df = pd.DataFrame({"task": tasks, "input": inputs, "predicted": predicted})
            df.to_feather(f"{self.path}/baseline_results.feather")
