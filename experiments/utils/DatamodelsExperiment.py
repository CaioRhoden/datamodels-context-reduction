
import os
from experiments.utils.experiment_modules.ExperimentConfig import DatamodelSchemeConfig
import pandas as pd

class DatamodelsExperiment:

    def __init__(self, folder_name: str,  k: int, folder_path: str, config: dict) -> None:
        """
        Initializes an experiment folder with the given name at the given path.

        Creates the following folders inside the experiment folder:
        - pre_collections
        - collections
        - datasets
        - estimations
        - logs

        Args:
            folder_name: str
                The name of the experiment folder.
            folder_path: str
                The path to the experiment folder.

        Returns:
            None
        """
        self.folder_name: str = folder_name
        self.k: int = k
        self.folder_path: str = folder_path
        
        self.config = config

        # Create auxiliary folders efficiently
        self.experiment_path: str = os.path.join(self.folder_path, self.folder_name)
        os.makedirs(self.experiment_path, exist_ok=True)
        
        subfolders: list[str] = ['pre_collections', 'collections', 'datasets', 'estimations', 'logs']
        for subfolder in subfolders:
            os.makedirs(os.path.join(self.experiment_path, subfolder), exist_ok=True)

        print(f"Experiment {self.folder_name} setup completed!")



        

    def create_scheme(self):
        
        from experiments.utils.experiment_modules.DatamodelsScheme import DatamodelsScheme

        ## Initialize Scheme config
        try:
            scheme_config = DatamodelSchemeConfig(**self.config["scheme"])
        except Exception as e:
            raise ValueError(f"Error creating scheme config: {e}")
        
        print("Initializing scheme creation")

        scheme = DatamodelsScheme(scheme_config)
        scheme(self)

        

        


        

    def create_pre_collection(self, config):
        pass

    def create_collection(self, config):
        pass

    def train_datamodels(self, config):
        pass

    def evaluate(self, config):
        pass

    def plot_results(self, config):
        pass

