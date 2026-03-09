import pytest
import numpy as np

pytest.importorskip("interpret")

from interpret.glassbox import ExplainableBoostingClassifier
from dmcr.datamodels.models.EBMClassifierModel import EBMClassifier


@pytest.fixture
def ebm():
    return EBMClassifier(in_features=10, out_features=1, device="cpu", ebm_configs={"n_estimators": 10})


class TestEBMClassifier:
    def test_init_creates_classifier_and_forwards_configs(self, ebm):
        assert isinstance(ebm.classifier, ExplainableBoostingClassifier)
        params = ebm.classifier.get_params()
        assert "n_estimators" in params

    def test_train_predict_and_get_weights(self, ebm):
        rng = np.random.RandomState(0)
        X = rng.randn(40, 10)
        y = rng.randint(0, 1, size=40)

        ebm.train(X, y)

        y_pred = ebm.classifier.predict_proba(X)
        assert len(y_pred) == len(X)

        weights = ebm.get_weights()
        assert hasattr(weights, "__len__")
        assert len(weights) <= 100
        assert all(isinstance(v, (int, float, np.floating, np.integer)) for v in weights)

    def test_evaluate_returns_auc_and_unsupported_raises(self, ebm):
        rng = np.random.RandomState(1)
        X = rng.randn(50, 10)
        y = rng.randint(0, 2, size=50)

        ebm.train(X, y)
        auc = ebm.evaluate(X, y, metric="roc_auc")
        assert isinstance(auc, float)
        assert 0.0 <= auc <= 1.0

        with pytest.raises(ValueError):
            ebm.evaluate(X, y, metric="not_a_metric")

    def test_save_model_creates_file(self, ebm, tmp_path):
        out = tmp_path / "model.pkl"
        ebm.save_model(str(out))
        assert out.exists()
