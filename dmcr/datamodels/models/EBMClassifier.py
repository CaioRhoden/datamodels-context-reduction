from interpret.glassbox import ExplainableBoostingClassifier
import joblib
from sklearn.metrics import roc_auc_score


class EBMClassifier():
    def __init__(self, in_features: int, out_features: int, device: str = "cpu", ebm_configs: dict = None):
        self.device = device
        self.classifier = ExplainableBoostingClassifier(**(ebm_configs or {}))

    
    def get_weights(self):
        return self.classifier.explain_global().data()["scores"][:100]
    
    def save_model(self, path: str):
        joblib.dump(self.classifier, path)

    def train(self, x, target):
        self.classifier.fit(x, target)


    def evaluate(self, x, target, metric: str = "roc_auc") -> float:
        y_pred = self.classifier.predict_proba(x)
        if metric == "roc_auc":
            return roc_auc_score(target, y_pred)
        else:
            raise ValueError("Unsupported metric")