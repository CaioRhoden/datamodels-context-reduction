from src.pipeline.BasePipeline import BasePipeline
from src.dataloader import PartialCSVDataLoader
from src.retriever import BM25
from src.llms import Llama3_1
import pandas as pd


def run_classical_retriever() -> None:

    train_dataloader = PartialCSVDataLoader('../../data/instruction-induction-data/raw/induce_tasks_examples.csv')

    pipe = BasePipeline(
        retriever=BM25(),
        dataloader=train_dataloader,
        llm=Llama3_1(),
        device="cuda:0",
    )

    test = pd.read_csv('../../data/instruction-induction-data/raw/execute_tasks_examples.csv')

    print("ALOHAAA")

    pipe.run_tests(
        test_data = test,
        checkpoint = 0,
        checkpoints_step= 50,
        k = 8,
        run_tag = "classical_run_20_09"
    )



run_classical_retriever()


