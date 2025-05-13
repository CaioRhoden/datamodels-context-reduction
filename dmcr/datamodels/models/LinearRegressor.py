import torch.nn as nn
import torch
from torcheval.metrics import R2Score

class LinearRegressor(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: str = "cuda:0"):
        super(LinearRegressor, self).__init__()
        self.device = device
        self.linear = nn.Linear(in_features, out_features, device=self.device)

    def forward(self, x):
        return self.linear(x).to(self.device)

    
    def get_weights(self):
        return self.linear.weight.to(self.device)
    
    def get_bias(self):
        return self.linear.bias.to(self.device)


    def evaluate(self, x, target, metric: str = "mse"):
        with torch.no_grad():  # Disable gradient calculation for evaluation
            predictions = self.linear(x).squeeze(1)

            if metric == "mse":
                mse = nn.MSELoss()(predictions, target)
                return mse.item()  # Return MSE as a scalar

            elif metric == "R2Score":
                metric = R2Score()
                metric.update(predictions, target)
                return metric.compute().item()
                
