"""
Unit Tests for Base GBDT Models (CatBoost, XGBoost, LightGBM).
"""

import pytest
import numpy as np
import pandas as pd
from src.hybrid_ids.models.catboost_model import CatBoostModel
from src.hybrid_ids.models.xgboost_model import XGBoostModel
from src.hybrid_ids.models.lightgbm_model import LightGBMModel


@pytest.fixture
def synthetic_multiclass_data():
    """Generates synthetic 3-class separable tabular dataset."""
    rng = np.random.RandomState(42)
    n = 60
    X = np.vstack([
        rng.normal(loc=-2.0, scale=0.5, size=(20, 4)),
        rng.normal(loc=0.0, scale=0.5, size=(20, 4)),
        rng.normal(loc=2.0, scale=0.5, size=(20, 4))
    ]).astype(np.float32)
    y = np.array([0] * 20 + [1] * 20 + [2] * 20)
    return X, y


def test_catboost_fit_predict(synthetic_multiclass_data):
    X, y = synthetic_multiclass_data
    model = CatBoostModel(iterations=10, depth=3, learning_rate=0.1, verbose=0)
    model.fit(X, y)
    preds = model.predict(X)
    probas = model.predict_proba(X)

    assert len(preds) == len(y)
    assert probas.shape == (len(y), 3)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)


def test_xgboost_fit_predict(synthetic_multiclass_data):
    X, y = synthetic_multiclass_data
    model = XGBoostModel(n_estimators=10, max_depth=3, learning_rate=0.1)
    model.fit(X, y)
    preds = model.predict(X)
    probas = model.predict_proba(X)

    assert len(preds) == len(y)
    assert probas.shape == (len(y), 3)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)


def test_lightgbm_fit_predict(synthetic_multiclass_data):
    X, y = synthetic_multiclass_data
    model = LightGBMModel(n_estimators=10, max_depth=3, learning_rate=0.1, verbose=-1)
    model.fit(X, y)
    preds = model.predict(X)
    probas = model.predict_proba(X)

    assert len(preds) == len(y)
    assert probas.shape == (len(y), 3)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)
