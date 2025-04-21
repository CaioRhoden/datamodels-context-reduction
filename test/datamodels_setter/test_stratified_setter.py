import pytest
from dmcr.datamodels.setter.StratifiedSetter import StratifiedSetter
from dmcr.datamodels.setter.SetterConfig import StratifiedSetterConfig
import os
import h5py
import polars as pl

class TestStratifiedSetter:

    @pytest.fixture(autouse=True)
    def setup(self):
        
        """
        Setup and teardown for StratifiedSetter tests.

        This fixture is automatically invoked for every test in this class. It creates a StratifiedSetter
        instance, calls its set() method to generate train and test collections, and then yields control to the
        test. After the test is finished, it removes the created HDF5 files.

        Yields:
            None
        """
        setter = StratifiedSetter(
            load_path_target="toy_target.feather",
            load_path_random="toy_random.feather",
            save_path="results",
            k=3,
            n_samples_target=2,
            n_test_target=1,
            n_samples_mix=2,
            n_test_mix=1,
            n_samples_random=2,
            n_test_random=1,
            index_col="index",
            seed=42
        )

        setter.set()

        yield

        os.remove("results/train_collection.h5")
        os.remove("results/test_collection.h5")
    

    def test_stratified_setter_collection_creation(self):
        """
        Tests that the StratifiedSetter creates train and test collections and saves them as HDF5 files.
        """
        assert os.path.exists("results/train_collection.h5")
        assert os.path.exists("results/test_collection.h5")
        
    def test_stratified_setter_number_of_samples(self):

        """
        Verifies that the StratifiedSetter creates the correct number of samples
        in the train and test collections and saves them as HDF5 files.

        Asserts that the train collection consists of 6 samples and the test
        collection consists of 3 samples based on the configuration provided.
        """

        with h5py.File(f"results/train_collection.h5", 'r') as hf:
            train_collection = hf['train_collection'][:]
            assert len(train_collection) == 3

        with h5py.File(f"results/test_collection.h5", 'r') as hf:
            test_collection = hf['test_collection'][:]
            assert len(test_collection) == 3

    def test_indices_in_collection(self):
        with h5py.File(f"results/train_collection.h5", 'r') as hf:
            train_collection = hf['train_collection'][0]
            assert set(train_collection).issubset(set(range(0, 6)))

        with h5py.File(f"results/train_collection.h5", 'r') as hf:
            train_collection = hf['train_collection'][1]
            assert set(train_collection).issubset(set(range(6, 12)))

        
    


        



