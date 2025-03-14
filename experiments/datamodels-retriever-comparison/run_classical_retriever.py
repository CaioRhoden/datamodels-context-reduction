from dmcr.pipelines.BasePipeline import BasePipeline
from dmcr.dataloaders import PartialCSVDataLoader
from dmcr.retrievers import BM25
from dmcr.models import Llama3_1
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


    pipe.run_tests(
        test_data = test,
        checkpoint = 0,
        checkpoints_step= 50,
        k = 8,
        run_tag = "classical_run_20_09"
    )



run_classical_retriever()


