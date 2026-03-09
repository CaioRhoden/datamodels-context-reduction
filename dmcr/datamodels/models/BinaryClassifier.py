from abc import ABC, abstractmethod

class BinaryClassifier(ABC):

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def save_model(self, path):
        pass

    @abstractmethod
    def load_model(self, path):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def evaluate(self, X, y, metric):
        pass
