"""
Unit Tests for Data Cleaning and Feature Selection.
"""

import pytest
import pandas as pd
import numpy as np
from src.hybrid_ids.preprocessing.cleaning import PipelineCleaner
from src.hybrid_ids.preprocessing.feature_selection import FeatureSelector


def test_pipeline_cleaner_imputation_and_leakage_dropping():
    """Verify cleaner fits imputers on train data and removes metadata drop columns."""
    df_train = pd.DataFrame({
        "frame.time": ["12:00:00", "12:00:01", "12:00:02"],
        "ip.src_host": ["192.168.1.1", "192.168.1.2", "192.168.1.3"],
        "tcp.len": [100.0, np.nan, 300.0],
        "tcp.flags.ack": [1.0, 1.0, 0.0],
        "Attack_type": ["Normal", "DDoS_TCP", "Normal"]
    })

    df_test = pd.DataFrame({
        "frame.time": ["12:00:03"],
        "ip.src_host": ["192.168.1.4"],
        "tcp.len": [np.nan],
        "tcp.flags.ack": [1.0],
        "Attack_type": ["DDoS_TCP"]
    })

    cleaner = PipelineCleaner(drop_metadata=True)
    X_train = cleaner.fit_transform(df_train, target_cols=["Attack_type"])
    X_test = cleaner.transform(df_test)

    # Verify IP and timestamp are dropped
    assert "frame.time" not in X_train.columns
    assert "ip.src_host" not in X_train.columns
    assert "frame.time" not in X_test.columns
    assert "ip.src_host" not in X_test.columns

    # Verify NaN was imputed with training median (200.0)
    assert X_test["tcp.len"].iloc[0] == 200.0


def test_feature_selector_mutual_info():
    """Verify feature selector selects k best informative features."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame({
        "informative": np.array([0.0, 0.1, 0.9, 1.0, 0.05, 0.95]),
        "noise": rng.randn(6),
        "constant": [1.0] * 6
    })
    y = np.array([0, 0, 1, 1, 0, 1])

    selector = FeatureSelector(method="mutual_info", k_features=1)
    X_sel = selector.fit_transform(X, y)

    assert X_sel.shape[1] == 1
    assert "informative" in X_sel.columns
