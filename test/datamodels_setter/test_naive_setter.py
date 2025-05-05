import pytest
from dmcr.datamodels.setter.NaiveSetter import NaiveSetter
import os
import h5py

class TestNaiveSetter:

    @pytest.fixture(autouse=True)
    def setup(self):
        
       
        setter = NaiveSetter(
            load_path="toy_target.feather",
            save_path="results",
            k=3,
            index_col="index",
            train_samples=3,
            test_samples=1,
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
            assert len(test_collection) == 1

        
    


        



