import torch
from dmcr.datamodels.models import LinearRegressor

class TestLinearRegressor:
    
    @classmethod
    def setup_class(cls):
        device = "cpu"
        cls.model = LinearRegressor(10, 1).to(device)
        cls.x = torch.randn(5, 10).to(device)
        cls.target = torch.randn(5).to(device)
    
    def test_r2(self):
        r2 = self.model.evaluate(self.x, self.target, metric="R2Score")
        assert isinstance(r2, float)