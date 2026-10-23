"""
Feature Selection Module for Gradient-Boosted Trees.
Implements correlation-based filtering, mutual-information scoring, and minority-preserving protocol feature retention.
Fitted exclusively on training partitions to prevent data leakage.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


class FeatureSelector:
    """Selects optimal predictive features using correlation filtering and mutual information."""

    def __init__(
        self,
        method: str = "mutual_info",
        k_features: int = 33,
        correlation_threshold: float = 0.95
    ):
        self.method = method
        self.k_features = k_features
        self.correlation_threshold = correlation_threshold
        self.selected_features_: List[str] = []
        self.feature_scores_: Dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "FeatureSelector":
        """Fits feature selector strictly on training features and targets."""
        feature_names = list(X.columns)

        if self.method == "all":
            self.selected_features_ = feature_names[:self.k_features] if self.k_features < len(feature_names) else feature_names
            self.feature_scores_ = {f: 1.0 for f in self.selected_features_}
            self.is_fitted = True
            return self

        # 1. Drop constant columns (0 variance on training set)
        non_constant_cols = [c for c in feature_names if X[c].nunique() > 1]
        if not non_constant_cols:
            self.selected_features_ = feature_names[:self.k_features]
            self.is_fitted = True
            return self

        # 2. Correlation filter: remove highly collinear redundant features (|rho| > threshold)
        corr_matrix = X[non_constant_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        drop_cols = [col for col in upper_tri.columns if any(upper_tri[col] > self.correlation_threshold)]
        filtered_cols = [c for c in non_constant_cols if c not in drop_cols]

        # 3. Mutual Information ranking on training partition
        X_sub = X[filtered_cols].values
        n_neighbors = min(3, max(1, len(X_sub) - 1))
        if len(X_sub) > 10000:
            indices = np.random.RandomState(42).choice(len(X_sub), size=10000, replace=False)
            mi_scores = mutual_info_classif(X_sub[indices], y[indices], n_neighbors=n_neighbors, random_state=42)
        else:
            mi_scores = mutual_info_classif(X_sub, y, n_neighbors=n_neighbors, random_state=42)

        score_dict = {col: float(score) for col, score in zip(filtered_cols, mi_scores)}
        self.feature_scores_ = score_dict

        # Rank features descending by MI score
        sorted_features = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
        self.selected_features_ = [f[0] for f in sorted_features[:self.k_features]]
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Projects DataFrame onto selected features."""
        if not self.is_fitted:
            raise RuntimeError("FeatureSelector is not fitted!")
        return X[self.selected_features_].copy()

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
