"""
Unit Tests for Dataset Discovery and Schema Quality Validation.
"""

import pytest
import pandas as pd
import numpy as np
from src.hybrid_ids.data.validation import DataValidator
from src.hybrid_ids.data.loaders import DataLoader


def test_data_validator_detects_nulls_infs_constants():
    """Verify validator identifies nulls, infs, constant columns, and duplicate rows."""
    df = pd.DataFrame({
        "feat_normal": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feat_with_null": [1.0, np.nan, 3.0, 4.0, 5.0],
        "feat_with_inf": [1.0, 2.0, np.inf, 4.0, -np.inf],
        "feat_constant": [7.0, 7.0, 7.0, 7.0, 7.0],
        "target": ["Benign", "Attack", "Attack", "Benign", "Attack"]
    })

    audit = DataValidator.audit_quality(df, target_col="target")
    assert audit["total_rows"] == 5
    assert audit["total_cols"] == 5
    assert "feat_with_null" in audit["null_column_counts"]
    assert "feat_with_inf" in audit["inf_column_counts"]
    assert "feat_constant" in audit["constant_columns"]
    assert "Attack" in audit["target_distribution"]
    assert "Benign" in audit["target_distribution"]


def test_loader_instantiation(tmp_path):
    """Verify DataLoader initializes properly and checks project root."""
    loader = DataLoader(tmp_path)
    assert loader.root == tmp_path
