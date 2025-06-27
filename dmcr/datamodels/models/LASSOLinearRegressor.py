
from torcheval.metrics import R2Score
from sklearn import linear_model
from dmcr.datamodels.models import LinearRegressor
import torch
import torch.nn as nn


class LASSOLinearRegressor(LinearRegressor):

    def __init__(self, in_features: int, out_features: int,  lambda_l1: float, device: str = "cuda:0"):
        """
        Initializes the LassoLinearRegressor with the given input and output features, 
        L1 penalty, and device.

        Parameters:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            l1_penalty (float): The L1 regularization penalty.
            device (str, optional): The device for computation. Defaults to "cuda:0".
        """

        super().__init__(in_features, out_features, device)
        self.lambda_l1 = lambda_l1
        self.weights = torch.empty(
            (in_features, out_features),
            dtype=torch.float32,
            device=device,
            requires_grad=True
        )
        # Correct bias shape: (out_features,)
        self.bias = torch.zeros(
            out_features,  # Shape: [1] for out_features=1
            dtype=torch.float32,
            requires_grad=True,
            device=device
        )

    def forward(self, x):
        x  = x.to(self.device)
        return (x @ self.weights + self.bias)
    



    def optimize(self, y_pred, target, lr)    -> float:

        """
        Perform a single optimization step for the LassoLinearRegressor model.

        This function computes the loss using the specified criterion, performs
        backpropagation to calculate gradients, and updates the model parameters
        using the provided optimizer.

        Parameters:
            y_pred (torch.tensor): The output of the model.
            target (torch.tensor): The true target values to compare against predictions.
            lr (float): The learning rate for optimization.

        Returns:
            float: The computed loss value as a scalar.
        """
        criterion = nn.MSELoss()
        mse_loss = criterion(y_pred, target).to(self.device).to(dtype=torch.float32)
        print(f"prediction: {y_pred}, target: {target}")
            
        # Compute L1 penalty (only on weights, not bias)
        l1_penalty = self.lambda_l1 * torch.sum(torch.abs(self.weights))
        
        # Total loss
        total_loss = mse_loss + l1_penalty
        
        # Compute gradients
        total_loss.backward()
        
        # Update parameters using gradient descent
        with torch.no_grad():
            self.weights -= lr * self.weights.grad
            self.bias -= lr * self.bias.grad
            
            # Zero gradients after updating
            self.weights.grad.zero_()
            self.bias.grad.zero_()
        
        return float(mse_loss.item())


    def get_weights(self):
        return self.weights.view(1,-1)

    def get_bias(self):
        return self.bias
    
    def detach(self):
        """
        Detach the model weights and bias from the computation graph.

        This function can be used to remove the model from the computation graph,
        which can be useful for saving the model or when the model is no longer
        needed.

        Returns:
            None
        """
        self.weights = self.weights.detach()
        self.bias = self.bias.detach()


    
