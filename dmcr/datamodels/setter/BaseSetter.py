
from abc import ABC, abstractmethod
from typing import Any
class BaseSetter(ABC):

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set(self) -> None:
        pass