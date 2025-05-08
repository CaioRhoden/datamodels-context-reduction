import pytest
import polars as pl
import numpy as np
import h5py
from dmcr.datamodels.pipeline import DatamodelsIndexBasedNQPipeline
from dmcr.datamodels import DatamodelIndexBasedConfig
from dmcr.evaluators import Rouge_L_evaluator
from dmcr.models import GenericInstructModelHF
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

        pipe = DatamodelsIndexBasedNQPipeline(config, test_flag=True)
        evaluator = Rouge_L_evaluator()

        pipe.create_collection(evaluator=evaluator, collection_name="unit_test", mode="train")
        pipe.create_collection(evaluator=evaluator, collection_name="unit_test", mode="test")
        
    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.datamodels_path)
        clean_temp_folders()

    def test_output_train_generated(self):
        assert os.path.exists(f"{self.datamodels_path}/collections/train/unit_test.feather")

    def test_output_test_generated(self):
        assert os.path.exists(f"{self.datamodels_path}/collections/test/unit_test.feather")

    def test_length_train_generated(self):
        df = pl.read_ipc(f"{self.datamodels_path}/collections/train/unit_test.feather")
        assert len(df) == 3

    def test_collection_dtypes(self):
        df = pl.read_ipc(f"{self.datamodels_path}/collections/train/unit_test.feather")
        assert df["collection_idx"].dtype == pl.Int64
        assert df["test_idx"].dtype == pl.Int64
        assert df["input"].dtype == pl.Array
        assert df["evaluation"].dtype == pl.Float64	