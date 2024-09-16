

class BaseDataloader():

    def __init__(
            self,
            path: str,
        ) -> None:

        self.path = path

    
    def load_data(self):
        """
        Method to load data
        """
        
        pass


    def preprocess_data(self):
        """Method to preprocess the data"""
        pass

    def get_data(self):
        """Method to retrieve the processed data"""
        pass	