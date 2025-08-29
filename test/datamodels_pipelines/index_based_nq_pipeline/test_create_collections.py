import pytest
import polars as pl
import numpy as np
from dmcr.datamodels.pipeline import DatamodelsIndexBasedNQPipeline
from dmcr.datamodels import DatamodelIndexBasedConfig
from dmcr.evaluators import Rouge_L_evaluator, JudgeEvaluator
from dmcr.utils.test_utils import clean_temp_folders
import tempfile
import os
import shutil


class TestIndexBasedNQPipelineCollectionCreation:

    @classmethod
    def setup_class(cls):

        ## CREATE TEMP DATA FOR PIPELINE
        clean_temp_folders()

        tmp_path = tempfile.mkdtemp(dir=os.getcwd())
        cls.datamodels_path = tmp_path


        train = {
            "text": ["train_text_1", "train_text_2", "train_text_3", "train_text_4", "train_text_5", "train_text_6", "train_text_7", "train_text_8", "train_text_9", "train_text_10"],
            "title": ["train_title_1", "train_title_2", "train_title_3", "train_title_4", "train_title_5", "train_title_6", "train_title_7", "train_title_8", "train_title_9", "train_title_10"],
        }

        test = {
            "question": ["test_question_1"],
            "answer": [["test_answer_1"]],
        }
        
        ## Create dfs

        train_df = pl.DataFrame(train).with_row_count("idx")
        test_df = pl.DataFrame(test).with_row_count("idx")

        ## Save temp data

        train_df.write_ipc(f"{tmp_path}/train_set.feather")
        test_df.write_ipc(f"{tmp_path}/test_set.feather")


        pre_collection = {
            "collection_idx": [0,1,2],
            "test_idx": [0,0,0],
            "input": [np.array([1,0,1]), np.array([1,0,0]), np.array([0,0,1])],
            "predicted_output": ["a", "b", "c"],
            "true_output": [["a"], ["b"], ["c", "d"]],
        }

        os.mkdir(f"{tmp_path}/pre_collections")
        os.mkdir(f"{tmp_path}/pre_collections/train")
        os.mkdir(f"{tmp_path}/pre_collections/test")

        df = pl.DataFrame(pre_collection)
        df.write_ipc(f"{tmp_path}/pre_collections/train/pre_collection_2.feather")
        df.write_ipc(f"{tmp_path}/pre_collections/test/pre_collection_2.feather")

        config = DatamodelIndexBasedConfig(
            k = 4,
            num_models= 1,
            datamodels_path = f"{tmp_path}",
            train_set_path= f"{tmp_path}/train_set.feather",
            test_set_path= f"{tmp_path}/test_set.feather",
        )

        pipe = DatamodelsIndexBasedNQPipeline(config, soft_test_flag=True)
        evaluator = Rouge_L_evaluator()

        def format_input(question, response):
            return f""""
            [System] 
            Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to a question displayed below. Your evaluation should consider factors such as relevance and accuracy. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".  
            [Question] 
            {question}  
            [The Start of Assistant’s Answer] 
            {response}
            [The End of Assistant’s Answer]
            """

        judge_evaluator = JudgeEvaluator(
            model_path=os.environ["DATAMODELS_TEST_MODEL"],
            model_configs = {
                "temperature": 0.5,
                "top_p": 0.9,
                "num_return_sequences": 5,
            },
            instruction="",
            format_template=format_input
        )

        pipe.create_collection(evaluator=evaluator, collection_name="unit_test", mode="train")
        pipe.create_collection(evaluator=evaluator, collection_name="unit_test", mode="test", checkpoint=2)

        pipe.create_collection(evaluator=judge_evaluator, collection_name="judge_test", mode="train")

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.datamodels_path)
        clean_temp_folders()

    def test_output_train_generated(self):
        assert os.path.exists(f"{self.datamodels_path}/collections/train/unit_test_0.feather")

    def test_output_test_generated_with_checkpoint(self):
        assert os.path.exists(f"{self.datamodels_path}/collections/test/unit_test_0.feather")
        assert os.path.exists(f"{self.datamodels_path}/collections/test/unit_test_2.feather")

    def test_length_train_generated(self):
        df = pl.read_ipc(f"{self.datamodels_path}/collections/train/unit_test_0.feather")
        assert len(df) == 3
    
    def test_judge_evaluator_output(self):
        df = pl.read_ipc(f"{self.datamodels_path}/collections/train/judge_test_0.feather")
        assert len(df) == 3
        assert "input" in df.columns
        assert "evaluation" in df.columns
        assert df["evaluation"].dtype == pl.Float64


    def test_collection_dtypes(self):
        df = pl.read_ipc(f"{self.datamodels_path}/collections/train/unit_test_0.feather")
        assert df["collection_idx"].dtype == pl.Int64
        assert df["test_idx"].dtype == pl.Int64
        assert df["input"].dtype == pl.Array
        assert df["evaluation"].dtype == pl.Float64	