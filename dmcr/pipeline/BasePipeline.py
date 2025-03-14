

from abc import ABC, abstractmethod
import pandas as pd


class BasePipeline(ABC):

    
    def __init__(
                    self, 
                ) -> None:
        pass


    @abstractmethod
    def run(self, input: str, k: int) -> str:

        pass

    @abstractmethod
    def run_tests(self, data: pd.DataFrame, checkpoint: int, k: int) -> None:
       pass
