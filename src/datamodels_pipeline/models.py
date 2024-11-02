import torch.nn as nn
import torch

class LinearRegressor(nn.Module):
    def __init__(self, weights, bias):
        super(LinearRegressor, self).__init__()
        self.weights = weights
        self.bias = bias

    def forward(self, x):
        return (self.weights * x).unsqueeze(0).sum(1) + self.bias

    def evaluate(self, x, target):
        with torch.no_grad():  # Disable gradient calculation for evaluation
            predictions = self.forward(x)
            mse = nn.MSELoss()(predictions, target)
            return mse.item()  # Return MSE as a scalar