import pytest
import polars as pl
import numpy as np
from dmcr.datamodels.pipeline import DatamodelsIndexBasedNQPipeline
from dmcr.datamodels import DatamodelIndexBasedConfig
from dmcr.datamodels.models import FactoryLinearRegressor
import torch

from dmcr.utils.test_utils import clean_temp_folders
import tempfile
import os
import shutil


class TestIndexBasedNQPipelineSelectCollectionTraining:

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
            "collection_idx": [i for i in range(10)],
            "test_idx": [i%2 for i in range(1,11)],
            "input": [np.array([1,0,1]) for i in range(10)],
            "evaluation": [0.5 for i in range(10)],
        }

        os.mkdir(f"{cls.datamodels_path}/collections")
        os.mkdir(f"{cls.datamodels_path}/collections/train")
        os.mkdir(f"{cls.datamodels_path}/collections/test")

        df = pl.DataFrame(pre_collection)
        df.write_ipc(f"{cls.datamodels_path}/collections/train/collections_test.feather")
        df.write_ipc(f"{cls.datamodels_path}/collections/test/collections_test.feather")



        

    @classmethod
    def teardown_class(cls):
        """
        Remove the temporary directory with the trained models and the associated data.
        Also clean temporary folders.
        """
        shutil.rmtree(cls.datamodels_path)
        clean_temp_folders()

    
    def test_collection_not_found(self):
        """
        Test that the collection is found in the train folder.
        """
        
        config = DatamodelIndexBasedConfig(
            k = 4,
            num_models= 2,
            datamodels_path = f"{self.datamodels_path}",
            train_set_path= f"{self.datamodels_path}/train_set.feather",
            test_set_path= f"{self.datamodels_path}/test_set.feather",
        )

        pipeline = DatamodelsIndexBasedNQPipeline(config, test_flag=True)
        model_factory = FactoryLinearRegressor(3, 1, device="cpu")

        with pytest.raises(Exception, match="No collections found in train folder"):
            pipeline.train_datamodels(
                model_factory=model_factory,
                collection_name = "collections_not_found",
                epochs=10,
                train_batches=1,
                val_batches=1,
                val_size=0.2,
                lr=0.001,
                patience=10,
                run_id="test",
            )
        