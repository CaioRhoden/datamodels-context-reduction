from abc import ABC, abstractmethod

class AbstractDataLoader(ABC):
    @abstractmethod
    def load_data(self):
        """Method to load data"""
        pass
    
    @abstractmethod
    def preprocess_data(self, data):
        """Method to preprocess the data"""
        pass
    
    @abstractmethod
    def get_data(self):
        """Method to retrieve the processed data"""
        pass