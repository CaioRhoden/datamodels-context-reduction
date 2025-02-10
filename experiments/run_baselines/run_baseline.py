"""
This script runs all the baselines that we have implemented.

It takes as input a config file specifying which baselines to run and on which datasets.

It will then run the baselines and save the results in a folder specified in the config file.

The results will be saved in a folder with the following structure:

<results_folder>/<dataset_name>/<baseline_name>.csv

The csv files will contain the columns 'test_idx', 'task', 'input', 'output', 'is_correct'

"""



from src.llms import GPT2, Llama3_1_Instruct, ParserLlama, BaseLLM
from src.utils import BaselineConfig
import argparse
from src.pipeline import BaselinePipeline
import pandas as pd
import json


def main(config: BaselineConfig):
    """
    Run the baseline experiment.

    Args:
        config (BaselineConfig): Configuration for the experiment.
            The configuration should contain the following fields:
                - llm_type (str): Type of the LLM to use.
                    Should be one of ('gpt2', 'llama3-1-instruct', 'llama-parser').
                - dataset (str): Path to the dataset to use.
                - saving_path (str): Path to save the results.
                - instructions (str): Path to the instruction file.
                - start (int): Index to start from.
                - run_instructions (bool): Whether to run with instructions or not.
    """

    llm = _select_llm(config.llm_type)
    df = pd.read_feather(config.dataset).reset_index(drop=True)

    pipeline = BaselinePipeline(config.saving_path, llm, config.instructions)

    pipeline.run_tests(df, config.start, config.run_instructions)
    print("All baseline tests completed")


def _select_llm(llm_type: str) -> BaseLLM:
    """
    Selects the LLM to use based on the given type.

    Args:
        llm_type (str): Type of the LLM to use.
            Should be one of ('gpt2', 'llama3-1-instruct', 'llama-parser').

    Returns:
        BaseLLM: The selected LLM.

    Raises:
        ValueError: If the given type is unknown.
    """
    match llm_type:
        case "gpt2":
            llm = GPT2()

        case "llama3-1-instruct":
            llm = Llama3_1_Instruct()

        case "llama-parser":
            parser_instruction = "Return a JSON object with an 'answer' key that answers straight forward the user input"
            llm = ParserLlama(parser_instruction=parser_instruction)

        case _:
            raise ValueError(f"Unknown LLM: {llm_type}")

    return llm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass start and end index")
    parser.add_argument("--config", type=str, help="Config File")
    args = parser.parse_args()

    configs_dict = json.load(open(args.config))

    try:
        config = BaselineConfig(**configs_dict)

    except Exception as e:
        raise ValueError(f"Error loading config: {e}")

    main(config)
