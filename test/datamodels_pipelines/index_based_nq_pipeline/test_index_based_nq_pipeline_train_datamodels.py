import pytest
import polars as pl
import numpy as np
from dmcr.datamodels.pipeline import DatamodelsIndexBasedNQPipeline
from dmcr.datamodels import DatamodelIndexBasedConfig
import torch

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
            "collection_idx": [i for i in range(10)],
            "test_idx": [0 for i in range(10)],
            "input": [np.array([1,0,1]) for i in range(10)],
            "evaluation": [0.5 for i in range(10)],
        }

        os.mkdir(f"{tmp_path}/collections")
        os.mkdir(f"{tmp_path}/collections/train")
        os.mkdir(f"{tmp_path}/collections/test")

        df = pl.DataFrame(pre_collection)
        df.write_ipc(f"{tmp_path}/collections/train/collections_test.feather")
        df.write_ipc(f"{tmp_path}/collections/test/collections_test.feather")

        config = DatamodelIndexBasedConfig(
            k = 4,
            num_models= 1,
            datamodels_path = f"{tmp_path}",
            train_set_path= f"{tmp_path}/train_set.feather",
            test_set_path= f"{tmp_path}/test_set.feather",
        )

        pipeline = DatamodelsIndexBasedNQPipeline(config, test_flag=True)
        pipeline.train_datamodels(
            collection_name = "collections_test",
            epochs=50,
            train_batches=1,
            val_batches=1,
            val_size=0.2,
            lr=0.001,
            patience=10,
            run_id="test",

        )



        
        
    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.datamodels_path)
        clean_temp_folders()

    def test_output_generation_train_datamodels(self):
        assert os.path.exists(f"{self.datamodels_path}/models/test")
        assert os.path.exists(f"{self.datamodels_path}/models/test/weights.pt")
        assert os.path.exists(f"{self.datamodels_path}/models/test/bias.pt")

    def test_output_sizes(self):
        weights = torch.load(f"{self.datamodels_path}/models/test/weights.pt", weights_only=True)
        bias = torch.load(f"{self.datamodels_path}/models/test/bias.pt", weights_only=True)
        assert weights.shape == (1, 3)
        assert bias.shape == (1,)
