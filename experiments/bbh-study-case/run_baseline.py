from src.llms import GPT2
import argparse
from src.pipeline import BaselinePipeline
import pandas as pd

def main(llm: str, start: int = 0, checkpoint: int = 0):

    match llm:
        case "gpt2":
            llm = GPT2()
    

    path = "../../data/instruction-induction-data/baselines/GPT2/"
    instructions = "../../data/instruction-induction-data/instructions.json"

    pipeline = BaselinePipeline(path, llm, instructions)

    df = pd.read_csv("../../data/instruction-induction-data/datamodels/proportion_study/210_5/test_set.csv")

    pipeline.run_tests(df, checkpoint, start)

    print("All baseline tests completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass start and end index")
    parser.add_argument("--llm", type=str, help="LLM")
    parser.add_argument("-s", "--start", type=int, help="Start index")
    parser.add_argument("-c", "--checkpoint", type=int, help="Checkpoint")

    args = parser.parse_args()

    main(args.llm, args.start, args.checkpoint)

