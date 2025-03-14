from dmcr.dataloaders import BaseDataloader
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from typing import List, Any



class PartialCSVDataLoader(BaseDataloader):

    def __init__(
            self,
            path: str,
        ) -> None:

        super().__init__(path)

        ##
        self.data = self._preprocess_data(self._load_data())
        self.loader = DataFrameLoader(self.data, page_content_column="concat")


    
    def _load_data(self):
        """
        Loads the data from a CSV file at the specified path.

        Returns:
            pd.DataFrame: The loaded data.
        """

        return pd.read_csv(self.path)

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the input DataFrame by selecting only the 'input' and 'output' columns.

        Args:
            df (pd.DataFrame): The input DataFrame to be preprocessed.

        Returns:
            pd.DataFrame: The preprocessed DataFrame containing only the 'input' and 'output' columns.
        """
        df["concat"] = df["input"] + "\n" + df["output"]
        return df[["concat"]]

    

    def get_documents(self) -> List[Any]:
        """
        Retrieves the documents from the data loader.

        Returns:
            list: A list of documents loaded from the data loader.
        """

        return self.loader.load()


