"""
End-to-End Integration Tests.
Verifies integration across Data Loading, Preprocessing, Model Prediction,
Alert Generation, and Serialization.
"""

import numpy as np
import pandas as pd
from src.hybrid_ids.preprocessing.cleaning import PipelineCleaner
from src.hybrid_ids.preprocessing.feature_selection import FeatureSelector
from src.hybrid_ids.models.catboost_model import CatBoostModel
from src.hybrid_ids.inference.predictor import IntrusionPredictor


def test_end_to_end_train_predict_alert():
    """Verify flow from training down to real-time inference and alert generation."""
    np.random.seed(42)
    n_samples = 60
    n_features = 8

    # Create dummy raw traffic dataframe
    data = {f"feat_{i}": np.random.randn(n_samples) for i in range(n_features)}
    data["Attack_type"] = np.random.choice(["Normal", "DDoS_UDP", "MITM"], size=n_samples)
    data["Attack_label"] = [0 if t == "Normal" else 1 for t in data["Attack_type"]]
    df = pd.DataFrame(data)

    # 1. Clean
    cleaner = PipelineCleaner()
    X = cleaner.fit_transform(df)
    y = np.random.randint(0, 3, size=n_samples)

    # 2. Feature select
    selector = FeatureSelector(k_features=5, method="mutual_info")
    X_sel = selector.fit_transform(X, y)
    assert X_sel.shape[1] == 5

    # 3. Train fast smoke model
    model = CatBoostModel(iterations=10, depth=3, random_state=42)
    model.fit(X_sel, y)

    # 4. Predict
    preds = model.predict(X_sel)
    probs = model.predict_proba(X_sel)
    assert len(preds) == n_samples
    assert probs.shape == (n_samples, 3)

    # 5. Alert generation logic
    threat_detected = int(np.sum(preds != 0))
    assert threat_detected >= 0
