"""
LightGBM Classifier Wrapper for Multiclass Network Intrusion Detection.
Utilizes fast histogram-based gradient boosting with leaf-wise tree growth and early stopping.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from src.hybrid_ids.logging_config import setup_logger

logger = setup_logger("lightgbm_model")


class LightGBMModel:
    """Production wrapper for LGBMClassifier."""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.08,
        num_leaves: int = 31,
        max_depth: int = 6,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        early_stopping_round: int = 25,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = -1,
        **kwargs
    ):
        self.early_stopping_round = early_stopping_round
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "objective": "multiclass",
            "verbose": verbose
        }
        self.params.update(kwargs)
        self.model = LGBMClassifier(**self.params)
        self.feature_names_: List[str] = []
        self.classes_: np.ndarray = np.array([])
        self.is_fitted: bool = False

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame | np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "LightGBMModel":
        """Trains LightGBM with validation early stopping."""
        if isinstance(X_train, pd.DataFrame):
            self.feature_names_ = list(X_train.columns)
        self.classes_ = np.unique(y_train)

        callbacks = []
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            callbacks = [
                early_stopping(stopping_rounds=self.early_stopping_round, verbose=False),
                log_evaluation(period=0)
            ]

        logger.info(f"Training LightGBM on {len(X_train)} samples across {len(self.classes_)} classes...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks
        )
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predicts class indices."""
        if not self.is_fitted:
            raise RuntimeError("LightGBMModel is not fitted!")
        preds = self.model.predict(X)
        return np.ravel(preds).astype(np.int64)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predicts class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("LightGBMModel is not fitted!")
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_
