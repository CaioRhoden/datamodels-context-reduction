
from src.llms import BaseLLM
from src.pipeline import BasePipeline

import torch
from langchain.prompts import PromptTemplate
import pandas as pd
import json





class BaselinePipeline(BasePipeline):

    
    def __init__(
                    self, 
                    path: str,
                    llm: BaseLLM,
                    instructions: str
                ) -> None:

        self.llm = llm
        self.path = path

        with open(instructions, "r") as f:
            self.instructions = json.load(f)

    
    
    def run(self, input: str, instruction: str) -> str:

        template = template = """
            Fill the expected Output according to the instruction
            Intruction: {instruction}

            User Input: 
            {input}

            Model Output:
        """

        
        prompt = PromptTemplate.from_template(template).format(instruction=instruction, input=input)
        return self.llm.run(prompt)
     
    
    def run_tests(self, data: pd.DataFrame, checkpoints_step: int, checkpoint: int) -> None:

        tasks, inputs, predicted = [], [], []

        for i in range(checkpoint, len(data)):
            print(f"Step {i} of {len(data)}")
            tasks.append( data.loc[i]["task"])
            inputs.append( data.loc[i]["input"])

            input = str(data.loc[i]["input"])

            predicted.append(self.run(input, self.instructions[data.loc[i]["task"]]))

            if i % checkpoints_step == 0 and i > checkpoint:

                df = pd.DataFrame({"task": tasks, "input": inputs,  "predicted": predicted})
                df.to_feather(f"{self.path}/{i - checkpoints_step}_{i}.feather")
                tasks, inputs, predicted = [], [], []

            elif i == len(data) - 1:
         
                df = pd.DataFrame({"task": tasks, "input": inputs, "predicted": predicted})
                df.to_feather(f"{self.path}/{min(0, i - checkpoints_step)}_{i}.feather")





