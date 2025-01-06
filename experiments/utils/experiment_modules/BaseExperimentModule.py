from abc import ABC, abstractmethod
from dataclasses import dataclass

class BaseExpeimentModule(ABC):

    @abstractmethod
    def __init__(self, config: dataclass) -> None:
        pass

    @abstractmethod
    def __call__(self) -> None:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass