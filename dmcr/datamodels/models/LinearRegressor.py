from dmcr.datamodels.models.Regressor import Regressor
import torch.nn as nn
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


class LinearRegressor(Regressor):
    def __init__(self, model_configs: dict = None):
        model_configs = model_configs or {}
        self.model = LinearRegression(**model_configs)


    def predict(self, x):
        return self.model.predict(x)
    
    def predict_proba(self, x):
        return self.predict(x)

    
    def get_weights(self):
        if self.model.coef_ is not None:
            return self.model.coef_
        else:
            return None
    
    def get_bias(self):
        if self.model.intercept_ is not None:      
            return self.model.intercept_
        else:
            return None

    def train(self, x, y):
        self.model = self.model.fit(x, y)


    def evaluate(self, x, y, metric: str = "mse") -> float:
        predictions = self.predict(x)

        if metric == "mse":
            return self.model.score(x, y)

        elif metric == "R2Score":
            return r2_score(y, predictions)
        
        elif metric == "mae":
            return mean_absolute_error(y, predictions)
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
        