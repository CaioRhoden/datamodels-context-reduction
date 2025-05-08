import pytest
import polars as pl
import numpy as np
import h5py
from dmcr.datamodels.pipeline import DatamodelsIndexBasedNQPipeline
from dmcr.datamodels import DatamodelIndexBasedConfig
from dmcr.models import GenericInstructModelHF
from dmcr.utils.test_utils import clean_temp_folders
import tempfile
import os
import shutil
import json


class TestIndexBasedNQPipelinePreCollectionCreation:

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

        ## Create combinations for datamodel training
        with h5py.File(f"{tmp_path}/train_collection.h5", 'w') as hf:
            combinations = np.array([[0,1,2,3], [0,2,3,4]])
            hf.create_dataset('train_collection', data=combinations)

        with h5py.File(f"{tmp_path}/test_collection.h5", 'w') as hf:
            combinations = np.array([[1,2,3,4]])
            hf.create_dataset('test_collection', data=combinations)

        indexes = {
            "0": [0 ,1, 2, 3, 4],
            "1": [5, 6, 7, 8, 9],
        }

        with open(f"{tmp_path}/indexes.json", "w") as f:
            json.dump(indexes, f)




        ## INSTANTIATE PIPELINE
        model_configs = {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_length": 2048,
            "max_new_tokens": 10
        }

        config = DatamodelIndexBasedConfig(
            k = 4,
            num_models= 1,
            datamodels_path = f"{tmp_path}",
            train_set_path= f"{tmp_path}/train_set.feather",
            test_set_path= f"{tmp_path}/test_set.feather",
        )

        pipe =DatamodelsIndexBasedNQPipeline(config)

        pipe.create_pre_collection( start_idx = 0, 
                                   end_idx = 1, 
                                   mode="train", 
                                   log=False, 
                                   output_column="answer", 
                                   model_configs=model_configs,
                                   instruction= "Test",
                                   rag_indexes_path=f"{tmp_path}/indexes.json",
                                  llm = GenericInstructModelHF(os.environ["DATAMODELS_TEST_MODEL"], quantization=True),
                                )
        
        pipe.create_pre_collection( start_idx = 0, 
                                   end_idx = 1, 
                                   mode="test", 
                                   log=False, 
                                   output_column="answer", 
                                   model_configs=model_configs,
                                   instruction= "Test",
                                   rag_indexes_path=f"{tmp_path}/indexes.json",
                                  llm = GenericInstructModelHF(os.environ["DATAMODELS_TEST_MODEL"], quantization=True),
                                )

    # @classmethod
    # def teardown_class(cls):
        # shutil.rmtree(cls.datamodels_path)
        # clean_temp_folders()


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

    def test_pre_collection_dtypes(self):
        df = pl.read_ipc(f"{self.datamodels_path}/pre_collections/train/pre_collection_0.feather")
        assert df["collection_idx"].dtype == pl.Int64
        assert df["test_idx"].dtype == pl.Int64
        assert df["input"].dtype == pl.Array
        assert df["predicted_output"].dtype == pl.String
        assert df["true_output"].dtype == pl.List(pl.String)

    


