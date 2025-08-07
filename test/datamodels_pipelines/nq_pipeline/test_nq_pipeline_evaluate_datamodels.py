import pytest
import polars as pl
import numpy as np
from dmcr.datamodels.pipeline import DatamodelsNQPipeline
from dmcr.datamodels import DatamodelConfig
import torch

from dmcr.utils.test_utils import clean_temp_folders
import tempfile
import os
import shutil
import torch


class TestNQPipelineCollectionCreation:

    @classmethod
    def setup_class(cls):

        ## CREATE TEMP DATA FOR PIPELINE
        clean_temp_folders()

        tmp_path = tempfile.mkdtemp(dir=os.getcwd())
        cls.datamodels_path = tmp_path

        collection = {
            "collection_idx": [i for i in range(2)],
            "test_idx": [0 for i in range(2)],
            "input": [np.array([1,0,0]) for i in range(2)],
            "evaluation": [1 for i in range(2)],
        }

        weights = torch.tensor([[1,0,0]])
        bias = torch.tensor([0])
        df = pl.DataFrame(collection)

        ## Save temp files
        os.mkdir(f"{tmp_path}/collections")
        os.mkdir(f"{tmp_path}/collections/test")
        os.mkdir(f"{tmp_path}/models")
        os.mkdir(f"{tmp_path}/models/test")
        df.write_ipc(f"{tmp_path}/collections/test/collections_test.feather")
        torch.save(weights, f"{tmp_path}/models/test/weights.pt")
        torch.save(bias, f"{tmp_path}/models/test/bias.pt")

        config = DatamodelConfig(
            k = 3,
            num_models= 1,
            datamodels_path = f"{tmp_path}",
        )

        pipeline = DatamodelsNQPipeline(config, hard_test_flag=True)

        pipeline.evaluate_test_collections(
            evaluation_id="test",
            collection_name="collections_test",
            model_id="test",
            log=False
        )


        
        
    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.datamodels_path)
        clean_temp_folders()

    def test_output_generation_train_datamodels(self):
        assert os.path.exists(f"{self.datamodels_path}/evaluations/test.feather")

    def test_output_results_correction(self):
        df = pl.read_ipc(f"{self.datamodels_path}/evaluations/test.feather")
        assert len(df) == 1
    

