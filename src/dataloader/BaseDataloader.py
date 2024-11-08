from abc import ABC
from typing import List, Any

from pyparsing import abstractmethod

class BaseDataloader(ABC):

    
    def __init__(
            self,
            path: str,
        ) -> None:

        self.path = path

    @abstractmethod
    def get_documents(self) -> List[Any]:
        pass
