import ABC, abstractmethod
from src.datamodels.modules.ModuleConfig import ModuleConfig
from src.datamodels.pipeline.BaseDatamodelsPipeline import BaseDatamodelsPipeline

class BaseModule(ABC):
    @abstractmethod
    def __init__(self, module_config: ModuleConfig):
        """
        Initializes a new instance of the BaseModule class to be used on the Datamodels pipeline.

        Parameters:
            module_config(ModuleConfig): The configuration for the Datamodels module.
        """
        pass
    

    @abstractmethod
    def __str__(self):
        """
        Returns a string representation of the handler.

        Returns:
            str: A string representation of the handler.
        """
        pass

    @abstractmethod
    def run(self, datamodel: BaseDatamodelsPipeline):
        """
        Executes the main functionality of the handler.

        This method needs to be implemented by subclasses to define
        the specific behavior of the handler.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """

        pass

