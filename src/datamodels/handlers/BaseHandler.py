import ABC, abstractmethod

class BaseHandler(ABC):
    @abstractmethod
    def __init__(self):
        pass
    

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def handle(self, ):
        pass

