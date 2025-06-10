import torch
from dmcr.datamodels.models import LASSOLinearRegressor

class TestLinearRegressor:
    
    @classmethod
    def setup_class(cls):
        device = "cpu"
        cls.model = LASSOLinearRegressor(10, 1, 0.01).to(device)
        cls.x = torch.randn(5, 10).to(device)
        cls.target = torch.randn(5).to(device)
    
    def test_r2(self):
        r2 = self.model.evaluate(self.x, self.target, metric="R2Score")
        assert isinstance(r2, float)
    
    def test_get_bias(self):
        bias = self.model.get_bias()
        assert bias.shape == (1,)
    
    def test_get_weights(self):
        weights = self.model.get_weights()
        assert weights.shape == (1,10)