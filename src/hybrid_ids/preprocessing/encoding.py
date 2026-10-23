"""
Target Label Encoding and Categorical Feature Encoding.
Provides aligned, deterministic class-to-integer mappings with unknown class handling.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


class TargetLabelEncoder:
    """Robust label encoder with fixed class order and out-of-vocabulary handling."""

    def __init__(self, predefined_classes: Optional[List[str]] = None):
        self.classes_: List[str] = sorted(list(set(predefined_classes))) if predefined_classes else []
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes_)}
        self.idx_to_class: Dict[int, str] = {i: c for i, c in enumerate(self.classes_)}
        self.is_fitted: bool = bool(self.classes_)

    def fit(self, y: pd.Series | np.ndarray) -> "TargetLabelEncoder":
        """Fits classes from training target series."""
        unique_classes = sorted(list(set(pd.Series(y).dropna().astype(str).unique())))
        self.classes_ = unique_classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        self.idx_to_class = {i: c for i, c in enumerate(self.classes_)}
        self.is_fitted = True
        return self

    def transform(self, y: pd.Series | np.ndarray) -> np.ndarray:
        """Transforms labels to integer indices. Unseen classes are mapped to -1."""
        if not self.is_fitted:
            raise RuntimeError("TargetLabelEncoder is not fitted!")
        series = pd.Series(y).astype(str)
        return series.map(lambda c: self.class_to_idx.get(c, -1)).values.astype(np.int64)

    def inverse_transform(self, indices: np.ndarray) -> np.ndarray:
        """Transforms integer indices back to class labels."""
        if not self.is_fitted:
            raise RuntimeError("TargetLabelEncoder is not fitted!")
        return np.array([self.idx_to_class.get(int(idx), "Unknown") for idx in indices])

    def fit_transform(self, y: pd.Series | np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)

    @property
    def num_classes(self) -> int:
        return len(self.classes_)
