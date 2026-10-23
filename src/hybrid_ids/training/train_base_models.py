"""
Base Model Training Coordinator for CatBoost, XGBoost, and LightGBM.
Fits all three gradient-boosted tree architectures under identical training and validation budgets.
"""

from typing import Dict, List, Optional, Tuple, Any
import time
import numpy as np
import pandas as pd
from src.hybrid_ids.models.catboost_model import CatBoostModel
from src.hybrid_ids.models.xgboost_model import XGBoostModel
from src.hybrid_ids.models.lightgbm_model import LightGBMModel
from src.hybrid_ids.evaluation.metrics import MetricsCalculator
from src.hybrid_ids.utils.resource_monitor import benchmark_timer
from src.hybrid_ids.logging_config import setup_logger

logger = setup_logger("train_base_models")


class BaseModelsTrainer:
    """Trains and benchmarks the three core gradient-boosted tree classifiers."""

    def __init__(
        self,
        catboost_params: Optional[Dict[str, Any]] = None,
        xgboost_params: Optional[Dict[str, Any]] = None,
        lightgbm_params: Optional[Dict[str, Any]] = None
    ):
        self.catboost = CatBoostModel(**(catboost_params or {}))
        self.xgboost = XGBoostModel(**(xgboost_params or {}))
        self.lightgbm = LightGBMModel(**(lightgbm_params or {}))
        self.training_history: Dict[str, Any] = {}

    def fit_all(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Fits CatBoost, XGBoost, and LightGBM sequentially with early stopping and profiling."""
        results = {}

        # 1. Train CatBoost
        logger.info("--- Starting CatBoost Training ---")
        with benchmark_timer() as t_cat:
            self.catboost.fit(X_train, y_train, X_val, y_val)
        val_pred_cat = self.catboost.predict(X_val)
        val_proba_cat = self.catboost.predict_proba(X_val)
        metrics_cat = MetricsCalculator.compute_all_metrics(y_val, val_pred_cat, val_proba_cat, class_names)
        results["catboost"] = {
            "model": self.catboost,
            "val_probas": val_proba_cat,
            "metrics": metrics_cat,
            "timing": t_cat
        }
        logger.info(f"CatBoost Validation Macro-F1: {metrics_cat['macro_f1']} (Duration: {t_cat['duration_sec']}s)")

        # 2. Train XGBoost
        logger.info("--- Starting XGBoost Training ---")
        with benchmark_timer() as t_xgb:
            self.xgboost.fit(X_train, y_train, X_val, y_val)
        val_pred_xgb = self.xgboost.predict(X_val)
        val_proba_xgb = self.xgboost.predict_proba(X_val)
        metrics_xgb = MetricsCalculator.compute_all_metrics(y_val, val_pred_xgb, val_proba_xgb, class_names)
        results["xgboost"] = {
            "model": self.xgboost,
            "val_probas": val_proba_xgb,
            "metrics": metrics_xgb,
            "timing": t_xgb
        }
        logger.info(f"XGBoost Validation Macro-F1: {metrics_xgb['macro_f1']} (Duration: {t_xgb['duration_sec']}s)")

        # 3. Train LightGBM
        logger.info("--- Starting LightGBM Training ---")
        with benchmark_timer() as t_lgb:
            self.lightgbm.fit(X_train, y_train, X_val, y_val)
        val_pred_lgb = self.lightgbm.predict(X_val)
        val_proba_lgb = self.lightgbm.predict_proba(X_val)
        metrics_lgb = MetricsCalculator.compute_all_metrics(y_val, val_pred_lgb, val_proba_lgb, class_names)
        results["lightgbm"] = {
            "model": self.lightgbm,
            "val_probas": val_proba_lgb,
            "metrics": metrics_lgb,
            "timing": t_lgb
        }
        logger.info(f"LightGBM Validation Macro-F1: {metrics_lgb['macro_f1']} (Duration: {t_lgb['duration_sec']}s)")

        self.training_history = results
        return results
