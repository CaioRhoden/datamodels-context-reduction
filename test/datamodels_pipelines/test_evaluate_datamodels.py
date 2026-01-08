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
import torch


class TestIndexBasedNQPipelineCollectionCreation:

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

        config = DatamodelIndexBasedConfig(
            k = 4,
            num_models= 1,
            datamodels_path = f"{tmp_path}",
            train_set_path= f"{tmp_path}/train_set.feather",
            test_set_path= f"{tmp_path}/test_set.feather",
        )

        pipeline = DatamodelsIndexBasedNQPipeline(config, hard_test_flag=True)

        pipeline.evaluate_test_collections(
            evaluation_id="test",
            collection_name="collections_test",
            model_id="test",
            log=False,
            metric="R2Score"
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


class TestMultipleWeightsEvaluation:

    @classmethod
    def setup_class(cls):

        ## CREATE TEMP DATA FOR PIPELINE WITH MULTIPLE CHECKPOINTS
        clean_temp_folders()

        tmp_path = tempfile.mkdtemp(dir=os.getcwd())
        cls.datamodels_path = tmp_path
        cls.num_models = 3

        # Create collections for 3 different test models
        collection = {
            "collection_idx": [],
            "test_idx": [],
            "input": [],
            "evaluation": [],
        }

        # Add data for each test model
        for test_idx in range(cls.num_models):
            for collection_idx in range(2):
                collection["collection_idx"].append(collection_idx)
                collection["test_idx"].append(test_idx)
                collection["input"].append(np.array([1, 0, 0]))
                collection["evaluation"].append(1.0)

        df = pl.DataFrame(collection)

        # Create multiple checkpoint files (simulating checkpointed training)
        # Checkpoint 1: models 0-0
        weights_0 = torch.tensor([[1.0, 0.0, 0.0]])
        bias_0 = torch.tensor([0.0])
        
        # Checkpoint 2: models 1-1
        weights_1 = torch.tensor([[0.5, 0.5, 0.0]])
        bias_1 = torch.tensor([0.1])
        
        # Checkpoint 3: models 2-2
        weights_2 = torch.tensor([[0.3, 0.3, 0.4]])
        bias_2 = torch.tensor([0.2])

        ## Save temp files with checkpoint naming convention
        os.mkdir(f"{tmp_path}/collections")
        os.mkdir(f"{tmp_path}/collections/test")
        os.mkdir(f"{tmp_path}/models")
        os.mkdir(f"{tmp_path}/models/test_multi")
        
        df.write_ipc(f"{tmp_path}/collections/test/collections_test_multi.feather")
        
        # Save checkpoints with the new naming format
        torch.save(weights_0, f"{tmp_path}/models/test_multi/0_0_weights.pt")
        torch.save(bias_0, f"{tmp_path}/models/test_multi/0_0_bias.pt")
        torch.save(weights_1, f"{tmp_path}/models/test_multi/1_1_weights.pt")
        torch.save(bias_1, f"{tmp_path}/models/test_multi/1_1_bias.pt")
        torch.save(weights_2, f"{tmp_path}/models/test_multi/2_2_weights.pt")
        torch.save(bias_2, f"{tmp_path}/models/test_multi/2_2_bias.pt")

        config = DatamodelIndexBasedConfig(
            k=4,
            num_models=cls.num_models,
            datamodels_path=f"{tmp_path}",
            train_set_path=f"{tmp_path}/train_set.feather",
            test_set_path=f"{tmp_path}/test_set.feather",
        )

        pipeline = DatamodelsIndexBasedNQPipeline(config, hard_test_flag=True)

        pipeline.evaluate_test_collections(
            evaluation_id="test_multi",
            collection_name="collections_test_multi",
            model_id="test_multi",
            log=False,
            metric="R2Score"
        )

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.datamodels_path)
        clean_temp_folders()

    def test_output_generation_multiple_weights(self):
        assert os.path.exists(f"{self.datamodels_path}/evaluations/test_multi.feather")

    def test_output_results_count(self):
        df = pl.read_ipc(f"{self.datamodels_path}/evaluations/test_multi.feather")
        assert len(df) == self.num_models, f"Expected {self.num_models} evaluations, got {len(df)}"

    def test_all_test_indices_evaluated(self):
        df = pl.read_ipc(f"{self.datamodels_path}/evaluations/test_multi.feather")
        test_indices = df["test_idx"].to_list()
        expected_indices = list(range(self.num_models))
        assert sorted(test_indices) == expected_indices, f"Expected test indices {expected_indices}, got {test_indices}"

    def test_checkpoint_files_loaded_correctly(self):
        # Verify that all checkpoint files exist
        model_dir = f"{self.datamodels_path}/models/test_multi"
        weight_files = sorted([f for f in os.listdir(model_dir) if f.endswith("_weights.pt")])
        bias_files = sorted([f for f in os.listdir(model_dir) if f.endswith("_bias.pt")])
        
        assert len(weight_files) == self.num_models, f"Expected {self.num_models} weight files, found {len(weight_files)}"
        assert len(bias_files) == self.num_models, f"Expected {self.num_models} bias files, found {len(bias_files)}"
    

