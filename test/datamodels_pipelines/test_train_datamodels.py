import pytest
import polars as pl
import numpy as np
from dmcr.datamodels.pipeline import DatamodelsIndexBasedNQPipeline
from dmcr.datamodels import DatamodelIndexBasedConfig
from dmcr.datamodels.models import FactoryLASSOLinearRegressor
import torch

from dmcr.utils.test_utils import clean_temp_folders
import tempfile
import os
import shutil


class TestPreCollectionsMultipleInferences:



    @classmethod
    def setup_class(cls):

        ## CREATE TEMP DATA FOR PIPELINE
        """
        Set up a test pipeline with a toy dataset. This pipeline trains a few LASSO linear
        regressors on a toy dataset. The purpose of this test is to make sure that the
        pipeline is working correctly, and that the trained models are saved to disk.
        """
        clean_temp_folders()

        tmp_path = tempfile.mkdtemp(dir=os.getcwd())
        cls.datamodels_path = tmp_path

        pre_collection = {
            "collection_idx": [i for i in range(12)],
            "test_idx": [i%4 for i in range(1,13)],
            "input": [np.array([1,0,1]) for i in range(12)],
            "evaluation": [0.5 for i in range(12)],
        }

        os.mkdir(f"{tmp_path}/collections")
        os.mkdir(f"{tmp_path}/collections/train")
        os.mkdir(f"{tmp_path}/collections/test")

        df = pl.DataFrame(pre_collection)
        df.write_ipc(f"{tmp_path}/collections/train/collections_test.feather")
        df.write_ipc(f"{tmp_path}/collections/test/collections_test.feather")

        config = DatamodelIndexBasedConfig(
            k = 4,
            num_models= 4,
            datamodels_path = f"{tmp_path}",
            train_set_path= f"{tmp_path}/train_set.feather",
            test_set_path= f"{tmp_path}/test_set.feather",
        )

        pipeline = DatamodelsIndexBasedNQPipeline(config, hard_test_flag=True)
        model_factory = FactoryLASSOLinearRegressor(3, 1, device="cpu", **{"lambda_l1": 0.01})


        pipeline.train_datamodels(
            model_factory=model_factory,
            collection_name = "collections_test",
            epochs=10,
            train_batches=1,
            val_batches=1,
            val_size=0.5,
            lr=0.001,
            patience=10,
            run_id="test",
            start_idx=0,
            end_idx=4,
            checkpoint=2

        )



        
        
    @classmethod
    def teardown_class(cls):
        """
        Remove the temporary directory with the trained models and the associated data.
        Also clean temporary folders.
        """
        shutil.rmtree(cls.datamodels_path)
        clean_temp_folders()

    def test_output_generation_train_datamodels(self):
        """
        Test that the output files are correctly generated after training
        the datamodels. This includes checking the existence of the model
        directory and the weights and bias files.
        """

        assert os.path.exists(f"{self.datamodels_path}/models/test")
        assert os.path.exists(f"{self.datamodels_path}/models/test/0_1_weights.pt")
        assert os.path.exists(f"{self.datamodels_path}/models/test/0_1_bias.pt")
        assert os.path.exists(f"{self.datamodels_path}/models/test/2_3_weights.pt")
        assert os.path.exists(f"{self.datamodels_path}/models/test/2_3_bias.pt")

        

    def test_output_sizes(self):
        """
        Verify the output sizes of the weights and bias after training the datamodels.
        
        This test ensures that the weights and bias loaded from the specified paths
        have the expected shapes. The weights are expected to have a shape of (2, 3),
        and the bias is expected to have a shape of (2,).
        """

        weights = torch.load(f"{self.datamodels_path}/models/test/0_1_weights.pt", weights_only=True)
        bias = torch.load(f"{self.datamodels_path}/models/test/0_1_bias.pt", weights_only=True)
        assert weights.shape == (2, 3)
        assert bias.shape == (2,)
        