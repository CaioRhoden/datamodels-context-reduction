import pytest
import polars as pl
import numpy as np
import h5py
from dmcr.datamodels.pipeline import DatamodelsNQPipeline
from dmcr.datamodels import DatamodelConfig
from dmcr.evaluators import Rouge_L_evaluator
from dmcr.models import GenericInstructModelHF
import tempfile
import os
import shutil


class TestNQPipelinePreCollectionCreation:

    @classmethod
    def setup_class(cls):

        ## CREATE TEMP DATA FOR PIPELINE

        tmp_path = tempfile.mkdtemp(dir=os.getcwd())
        cls.datamodels_path = tmp_path

        train = {
            "text": ["train_text_1", "train_text_2", "train_text_3", "train_text_4", "train_text_5", "train_text_6"],
            "title": ["train_title_1", "train_title_2", "train_title_3", "train_title_4", "train_title_5", "train_title_6"],
        }

        test = {
            "question": ["test_question_1"],
            "answer": ["test_answer_1"],
        }
        
        ## Create dfs

        train_df = pl.DataFrame(train).with_row_count("idx")
        test_df = pl.DataFrame(test).with_row_count("idx")

        ## Save temp data

        train_df.write_csv(f"{tmp_path}/train_set.csv")
        test_df.write_csv(f"{tmp_path}/test_set.csv")

        ## Create combinations for datamodel training
        with h5py.File(f"{tmp_path}/train_collection.h5", 'w') as hf:
            combinations = np.array([[0,1,2,3], [0,2,3,4]])
            hf.create_dataset('train_collection', data=combinations)

        with h5py.File(f"{tmp_path}/test_collection.h5", 'w') as hf:
            combinations = np.array([[1,2,3,4]])
            hf.create_dataset('test_collection', data=combinations)        


        ## INSTANTIATE PIPELINE
        model_configs = {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_length": 2048,
            "max_new_tokens": 10
        }

        config = DatamodelConfig(
            k = 4,
            num_models= 1,
            datamodels_path = f"{tmp_path}",
            train_collections_idx = None,
            test_collections_idx = None,
            test_set = None,
            train_set = None,
            instructions= "You are given a question and you MUST respond in 5 tokens, there are documents that can or cannot be helpful, and you MUST respond",
            llm = GenericInstructModelHF(os.environ["DATAMODELS_TEST_MODEL"], quantization=True),
            evaluator=Rouge_L_evaluator(),
            model_configs=model_configs
        )

        pipe = DatamodelsNQPipeline(config)

        pipe.create_pre_collection(start_idx = 0, end_idx = 1, type="train", log=False, log_config=None, checkpoint=10, output_column="answer")
        pipe.create_pre_collection(start_idx = 0, end_idx = 1, type="test", log=False, log_config=None, checkpoint=10, output_column="answer")

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.datamodels_path)

    def test_output_train_generated(self):
        assert os.path.exists(f"{self.datamodels_path}/pre_collections/train")
        assert os.path.exists(f"{self.datamodels_path}/pre_collections/train/pre_collection_0.feather")
        
    def test_output_test_generated(self):
        assert os.path.exists(f"{self.datamodels_path}/pre_collections/test")
        assert os.path.exists(f"{self.datamodels_path}/pre_collections/test/pre_collection_0.feather")

    def test_output_length(self):
        df = pl.read_ipc(f"{self.datamodels_path}/pre_collections/train/pre_collection_0.feather")
        assert len(df) == 1
    
    def test_checkpoint_stop(self):
        assert os.path.exists(f"{self.datamodels_path}/pre_collections/train/pre_collection_0.feather")
        assert not os.path.exists(f"{self.datamodels_path}/pre_collections/train/pre_collection_1.feather")


