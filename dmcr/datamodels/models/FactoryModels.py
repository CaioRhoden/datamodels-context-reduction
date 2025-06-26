

from abc import abstractmethod, ABC
from dmcr.datamodels.models.LinearRegressor import LinearRegressor
from dmcr.datamodels.models.LASSOLinearRegressor import LASSOLinearRegressor



class FactoryBaseModel(ABC):
    """
    Base class for linear regression model factories.
    """

    @abstractmethod
    def __init__(self, in_features: int, out_features: int, device="cpu", **kwargs):
        """ Initialize the factory with input and output features.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            device (str): Device to run the model on (default: "cpu").
            **kwargs: Additional keyword arguments for model configuration.
        """
        pass

    @abstractmethod
    def create_model(self) -> LinearRegressor:
        """
        Create a linear regression model with the specified input and output features.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            **kwargs: Additional keyword arguments for model configuration.

        Returns:
            LinearRegressor: A PyTorch linear regression model.
        """
        pass

class FactoryLinearRegressor(FactoryBaseModel):
    
    def __init__(self, in_features: int, out_features: int, device="cpu", **kwargs):
        """
        Initialize the factory with input and output features.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            device (str): Device to run the model on (default: "cpu").
            **kwargs: Additional keyword arguments for model configuration.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.kwargs = kwargs

    def create_model(self) -> LinearRegressor:
        """
        Create and return a LinearRegressor model.

        Returns:
            LinearRegressor: A PyTorch linear regression model initialized with
            the specified input features, output features, device, and additional
            configuration options.
        """

        return LinearRegressor(
            in_features=self.in_features,
            out_features=self.out_features,
            device=self.device,
            **self.kwargs
        )


class FactoryLASSOLinearRegressor(FactoryLinearRegressor):

    def __init__(self, in_features: int, out_features: int, device="cpu", **kwargs):
        """
        Initialize the factory with input and output features, and L1 penalty.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            device (str): Device to run the model on (default: "cpu").
            **kwargs: Additional keyword arguments for model configuration.
                lambda_l1 (float): L1 regularization penalty (default: 0.01).
        """
        super().__init__(in_features, out_features, device, **kwargs)
        try:
            self.lambda_l1 = kwargs.get("lambda_l1", 0.01)
        except KeyError:
            raise KeyError("lambda_l1 not found in kwargs")

    def create_model(self) -> LASSOLinearRegressor:
        """
        Create a LASSO linear regression model with the specified input and output features.

        Returns:
            LASSOLinearRegressor: An instance of the LASSOLinearRegressor model initialized 
            with the given features, device, and L1 regularization penalty.
        """

        return LASSOLinearRegressor(
            in_features=self.in_features,
            out_features=self.out_features,
            device=self.device,
            lambda_l1=self.lambda_l1
        )
