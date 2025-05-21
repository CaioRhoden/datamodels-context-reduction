from dmcr.evaluators import GenericEvaluator
import numpy as np

def test_generic_evaluator():
    evaluator = GenericEvaluator("rouge", "rougeL")
    preds = np.array(["hello", "world"])
    references = np.array([["hello"], ["world"]])
    results =  evaluator.evaluate(references, preds)
    assert results.shape == (2,)
    assert results.dtype == float