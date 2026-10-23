"""
Unit Tests for Validation-Based Hybrid Ensemble.
"""

import pytest
import numpy as np
from src.hybrid_ids.models.ensemble import HybridGBDTEnsemble
from src.hybrid_ids.models.calibration import MulticlassProbabilityCalibrator


class DummyModel:
    """Mock classifier emitting fixed synthetic probabilities for testing."""
    def __init__(self, probas: np.ndarray):
        self.probas = probas

    def predict_proba(self, X):
        return self.probas

    def predict(self, X):
        return np.argmax(self.probas, axis=1)


def test_ensemble_uniform_averaging():
    """Verify uniform probability blending across models."""
    p1 = np.array([[0.8, 0.2], [0.1, 0.9]])
    p2 = np.array([[0.6, 0.4], [0.3, 0.7]])
    p3 = np.array([[0.7, 0.3], [0.2, 0.8]])

    ensemble = HybridGBDTEnsemble(
        catboost_model=DummyModel(p1),
        xgboost_model=DummyModel(p2),
        lightgbm_model=DummyModel(p3),
        strategy="uniform",
        class_names=["Benign", "Attack"],
        calibrate=False
    )

    blended = ensemble.predict_proba(np.zeros((2, 2)))
    expected = (p1 + p2 + p3) / 3.0
    np.testing.assert_allclose(blended, expected, atol=1e-5)
    np.testing.assert_allclose(blended.sum(axis=1), 1.0, atol=1e-5)


def test_ensemble_weight_optimization():
    """Verify validation loss minimization produces non-negative simplex weights."""
    p1 = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]])
    p2 = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    p3 = np.array([[0.2, 0.8], [0.8, 0.2], [0.3, 0.7]])
    y_val = np.array([0, 1, 0])

    ensemble = HybridGBDTEnsemble(
        catboost_model=DummyModel(p1),
        xgboost_model=DummyModel(p2),
        lightgbm_model=DummyModel(p3),
        strategy="validation_weighted",
        class_names=["Benign", "Attack"],
        calibrate=False
    )

    weights = ensemble.optimize_weights_on_validation(p1, p2, p3, y_val)
    assert len(weights) == 3
    assert np.all(weights >= 0.0)
    assert abs(np.sum(weights) - 1.0) < 1e-4
    # CatBoost should have the highest weight because it has the lowest cross-entropy on y_val
    assert weights[0] > weights[2]


def test_probability_calibrator_simplex():
    """Verify calibrated probabilities sum to 1."""
    rng = np.random.RandomState(42)
    raw_p = rng.dirichlet(np.ones(3), size=50)
    y_val = np.random.choice([0, 1, 2], size=50)

    calib = MulticlassProbabilityCalibrator(method="sigmoid")
    calib.fit(raw_p, y_val)
    transformed = calib.transform(raw_p)

    np.testing.assert_allclose(transformed.sum(axis=1), 1.0, atol=1e-4)
