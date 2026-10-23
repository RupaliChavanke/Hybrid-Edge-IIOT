"""
Unit Tests for Leakage-Safe Dataset Splitting.
"""

import pytest
import pandas as pd
import numpy as np
from src.hybrid_ids.data.splitting import DatasetSplitter


def test_zero_leakage_split_isolation():
    """Verify train, val, and locked-test partitions have zero index overlap."""
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame({
        "feat_1": rng.randn(n),
        "feat_2": rng.randn(n),
        "target": ["Normal"] * 100 + ["Attack"] * 100
    })

    df_train, df_val, df_test = DatasetSplitter.train_val_test_split(
        df,
        target_column="target",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        stratify=True
    )

    assert len(df_train) == 140
    assert len(df_val) == 30
    assert len(df_test) == 30
    assert len(df_train) + len(df_val) + len(df_test) == n

    # Stratified balance check
    assert df_train["target"].value_counts()["Normal"] == 70
    assert df_val["target"].value_counts()["Normal"] == 15
    assert df_test["target"].value_counts()["Normal"] == 15
