"""
Golden Regression Contract Test.
Verifies predictions on versioned fixtures against known expected outputs, schemas, and numerical tolerances.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from src.hybrid_ids.inference.model_registry import ModelRegistry
from src.hybrid_ids.config import get_project_root


def test_golden_prediction_fixture():
    root = get_project_root()
    fixture_path = root / "tests" / "fixtures" / "golden_predictions.json"
    assert fixture_path.exists(), f"Golden fixture missing at {fixture_path}"

    with open(fixture_path, "r", encoding="utf-8") as f:
        fixture = json.load(f)

    # 1. Verify schema contract
    records = fixture["records"]
    df = pd.DataFrame(records)
    assert len(df) == fixture["sample_count"]

    registry = ModelRegistry()
    with open(root / "models" / "registry" / "cleaner_edge_iiot.pkl", "rb") as f:
        cleaner = pickle.load(f)
    with open(root / "models" / "registry" / "feature_selector_edge_iiot.pkl", "rb") as f:
        selector = pickle.load(f)
    model = registry.load_artifact("catboost_model", "v2")

    clean_df = cleaner.transform(df)
    X = selector.transform(clean_df)

    assert list(X.columns) == fixture["expected_schema"], "Feature order or columns do not match expected schema!"

    # 2. Verify prediction contracts
    probs = model.predict_proba(X)
    preds = model.predict(X)

    expected_shape = tuple(fixture["expected_shape"])
    assert probs.shape == expected_shape, f"Shape mismatch: got {probs.shape}, expected {expected_shape}"

    # Probabilities sum to 1.0 within tolerance
    row_sums = np.sum(probs, axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-4, err_msg="Probabilities do not sum to 1.0")

    # Probabilities non-negative and finite
    assert np.all(probs >= 0.0)
    assert np.all(np.isfinite(probs))

    # Reference predictions match
    expected_preds = fixture["reference_predictions"]
    np.testing.assert_array_equal(preds, expected_preds, err_msg="Predicted class indices deviated from golden baseline!")
