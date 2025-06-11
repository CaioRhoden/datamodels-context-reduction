import torch.nn as nn
import torch
from torcheval.metrics import R2Score
import torch.optim as optim


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


    def evaluate(self, x, target, metric: str = "mse") -> float:
        with torch.no_grad():  # Disable gradient calculation for evaluation
            predictions = self.forward(x).squeeze(1).to(self.device)
            target = target.to(self.device)

            if metric == "mse":
                mse = nn.MSELoss()(predictions, target)
                return  float(mse.item())  # Return MSE as a scalar

            elif metric == "R2Score":
                evaluator = R2Score(device=self.device)
                evaluator.update(predictions, target)
                return float(evaluator.compute().item())
            
            else:
                raise ValueError(f"Unknown metric: {metric}")
                
    def optimize(self, y_pred, target, lr)    -> float:
        """
        Perform a single optimization step for the linear regression model.

        This function computes the loss using the specified criterion, performs
        backpropagation to calculate gradients, and updates the model parameters
        using the provided optimizer.

        Args:
            criterion: The loss function used to compute the difference between
                    predictions and target values.
            optimizer: The optimizer used to update the model parameters based
                    on the computed gradients.
            y_pred: The predicted values output by the model.
            target: The true target values to compare against predictions.

        Returns:
            float: The computed loss value as a scalar.
        """

        criterion = nn.MSELoss()
        
        optimizer = optim.SGD(self.parameters(), lr=lr)

        loss = criterion(y_pred, target).to(self.device).to(dtype=torch.float32)  # Add batch dimension to target

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return float(loss.item())
    
    def detach(self):
        """

        Returns:
            None
        """
        pass