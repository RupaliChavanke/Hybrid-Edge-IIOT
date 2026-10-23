"""
CatBoost Classifier Wrapper for Multiclass Network Intrusion Detection.
Utilizes symmetric oblivious decision trees, early stopping, and multiclass probabilities.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from src.hybrid_ids.logging_config import setup_logger

logger = setup_logger("catboost_model")


class CatBoostModel:
    """Production wrapper for CatBoostClassifier."""

    def __init__(
        self,
        iterations: int = 300,
        learning_rate: float = 0.08,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        early_stopping_rounds: int = 25,
        random_seed: int = 42,
        thread_count: int = -1,
        verbose: int = 0,
        **kwargs
    ):
        if "random_state" in kwargs:
            random_seed = kwargs.pop("random_state")
        self.params = {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "l2_leaf_reg": l2_leaf_reg,
            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",
            "early_stopping_rounds": early_stopping_rounds,
            "random_seed": random_seed,
            "thread_count": thread_count,
            "verbose": verbose
        }
        self.params.update(kwargs)
        self.model = CatBoostClassifier(**self.params)
        self.feature_names_: List[str] = []
        self.classes_: np.ndarray = np.array([])
        self.is_fitted: bool = False

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame | np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        cat_features: Optional[List[int | str]] = None
    ) -> "CatBoostModel":
        """Trains CatBoost with validation early stopping."""
        if isinstance(X_train, pd.DataFrame):
            self.feature_names_ = list(X_train.columns)
        self.classes_ = np.unique(y_train)

        eval_set = (X_val, y_val) if X_val is not None and y_val is not None else None

        logger.info(f"Training CatBoost on {len(X_train)} samples across {len(self.classes_)} classes...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            cat_features=cat_features,
            use_best_model=True if eval_set is not None else False
        )
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predicts class indices."""
        if not self.is_fitted:
            raise RuntimeError("CatBoostModel is not fitted!")
        preds = self.model.predict(X)
        return np.ravel(preds).astype(np.int64)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predicts calibrated class probability distribution."""
        if not self.is_fitted:
            raise RuntimeError("CatBoostModel is not fitted!")
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.get_feature_importance()
