from dmcr.evaluators import GenericEvaluator
from dmcr.evaluators import SquadV2Evaluator
import numpy as np

def test_generic_evaluator():
    evaluator = GenericEvaluator("rouge", "rougeL")
    preds = np.array(["hello", "world"])
    references = np.array([["hello"], ["world"]])
    results =  evaluator.evaluate(references, preds)
    assert results.shape == (2,)
    assert results.dtype == float

def test_squadv2_evaluator():
    evaluator = SquadV2Evaluator(squadv2_key="best_f1")
    preds = np.array(["hello"])
    references = np.array([["hello", "art"]])
    results =  evaluator.evaluate(references, preds)
    assert results.shape == (1,)
    assert results.dtype == float
    assert results[0] == 1