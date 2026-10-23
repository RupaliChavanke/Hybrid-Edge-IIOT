"""
Data and Target Leakage Contract Tests.
Guarantees zero-leakage between train, validation, and locked test partitions,
and validates feature transformations do not leak target information.
"""

import numpy as np
import pandas as pd
from src.hybrid_ids.data.splitting import DatasetSplitter
from src.hybrid_ids.preprocessing.cleaning import PipelineCleaner
from src.hybrid_ids.preprocessing.feature_selection import FeatureSelector
from src.hybrid_ids.models.ensemble import HybridGBDTEnsemble


def test_zero_index_overlap_in_splits():
    """Verify train, validation, and test sets are mutually exclusive with zero index overlap."""
    np.random.seed(42)
    n_samples = 300
    df = pd.DataFrame({
        "feat1": np.random.randn(n_samples),
        "feat2": np.random.randn(n_samples),
        "label": np.random.choice(["Normal", "DDoS", "Port_Scan"], size=n_samples, p=[0.7, 0.2, 0.1]),
    })

    train_df, val_df, test_df = DatasetSplitter.train_val_test_split(
        df, target_column="label", train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42, stratify=True, reset_index=False
    )

    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    test_idx = set(test_df.index)

    assert len(train_idx.intersection(val_idx)) == 0, "Leakage detected: Train and Val share indices!"
    assert len(train_idx.intersection(test_idx)) == 0, "Leakage detected: Train and Test share indices!"
    assert len(val_idx.intersection(test_idx)) == 0, "Leakage detected: Val and Test share indices!"
    assert len(train_idx) + len(val_idx) + len(test_idx) == n_samples, "Partition count mismatch!"


def test_no_target_columns_in_features():
    """Verify target columns are dropped from feature matrices."""
    df = pd.DataFrame({
        "frame.time_delta": [0.01, 0.02, 0.03],
        "tcp.flags": [2, 16, 2],
        "Attack_type": ["Normal", "DDoS", "Normal"],
        "Attack_label": [0, 1, 0],
    })

    cleaner = PipelineCleaner()
    X = cleaner.fit_transform(df)

    assert "Attack_type" not in X.columns, "Target leakage: Attack_type found in feature set!"
    assert "Attack_label" not in X.columns, "Target leakage: Attack_label found in feature set!"
    assert "label" not in X.columns, "Target leakage: label found in feature set!"


def test_feature_selector_fit_isolation():
    """Verify FeatureSelector fit only depends on training data and retains fixed order."""
    np.random.seed(42)
    X_train = pd.DataFrame({
        "informative": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "noise1": np.random.randn(10),
        "noise2": np.random.randn(10),
    })
    y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    selector = FeatureSelector(k_features=2, method="mutual_info")
    selector.fit(X_train, y_train)

    assert "informative" in selector.selected_features_

    X_test = pd.DataFrame({
        "informative": [1, 2],
        "noise1": [0.1, 0.2],
        "noise2": [0.3, 0.4],
    })
    X_test_trans = selector.transform(X_test)
    assert list(X_test_trans.columns) == selector.selected_features_
    assert len(X_test_trans.columns) == 2
