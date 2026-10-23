"""
XGBoost Classifier Wrapper for Multiclass Network Intrusion Detection.
Utilizes exact/histogram gradient boosted decision trees with early stopping and regularization.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from src.hybrid_ids.logging_config import setup_logger

logger = setup_logger("xgboost_model")


class XGBoostModel:
    """Production wrapper for XGBClassifier."""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.08,
        max_depth: int = 6,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 25,
        random_state: int = 42,
        n_jobs: int = -1,
        tree_method: str = "hist",
        **kwargs
    ):
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "early_stopping_rounds": early_stopping_rounds,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "tree_method": tree_method,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss"
        }
        self.params.update(kwargs)
        self.model = XGBClassifier(**self.params)
        self.feature_names_: List[str] = []
        self.classes_: np.ndarray = np.array([])
        self.is_fitted: bool = False

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame | np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "XGBoostModel":
        """Trains XGBoost with validation early stopping."""
        if isinstance(X_train, pd.DataFrame):
            self.feature_names_ = list(X_train.columns)
        self.classes_ = np.unique(y_train)

        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None

        if eval_set is None:
            self.model.set_params(early_stopping_rounds=None)
        else:
            self.model.set_params(early_stopping_rounds=self.params["early_stopping_rounds"])

        logger.info(f"Training XGBoost on {len(X_train)} samples across {len(self.classes_)} classes...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False
        )
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predicts class indices."""
        if not self.is_fitted:
            raise RuntimeError("XGBoostModel is not fitted!")
        preds = self.model.predict(X)
        return np.ravel(preds).astype(np.int64)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predicts class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("XGBoostModel is not fitted!")
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_
