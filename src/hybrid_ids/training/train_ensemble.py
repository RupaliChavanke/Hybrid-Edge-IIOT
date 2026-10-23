"""
Validation-Based Ensemble Training & Model Selection Coordinator.
Evaluates CatBoost, XGBoost, LightGBM, Uniform Averaging, SLSQP Validation-Weighted Blending,
and Calibration on validation evidence before locking the final ensemble.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from src.hybrid_ids.models.ensemble import HybridGBDTEnsemble
from src.hybrid_ids.evaluation.metrics import MetricsCalculator
from src.hybrid_ids.logging_config import setup_logger

logger = setup_logger("train_ensemble")


class EnsembleTrainer:
    """Trains and optimizes the hybrid ensemble using validation and out-of-fold predictions."""

    @staticmethod
    def train_and_select_ensemble(
        catboost_model: Any,
        xgboost_model: Any,
        lightgbm_model: Any,
        val_probas_cat: np.ndarray,
        val_probas_xgb: np.ndarray,
        val_probas_lgb: np.ndarray,
        y_val: np.ndarray,
        class_names: Optional[List[str]] = None,
        calibrate: bool = True
    ) -> Tuple[HybridGBDTEnsemble, Dict[str, Any]]:
        """Compares ensemble candidates and constructs the optimal validation-selected ensemble."""
        logger.info("Evaluating validation ensemble strategies...")
        comparison = {}

        # 1. Uniform Probability Averaging
        unif_probas = (val_probas_cat + val_probas_xgb + val_probas_lgb) / 3.0
        unif_preds = np.argmax(unif_probas, axis=1)
        metrics_unif = MetricsCalculator.compute_all_metrics(y_val, unif_preds, unif_probas, class_names)
        comparison["uniform_average"] = metrics_unif
        logger.info(f"Uniform Averaging Validation Macro-F1: {metrics_unif['macro_f1']}")

        # 2. Validation-Optimized Weighted Blending (SLSQP on Simplex)
        ensemble_weighted = HybridGBDTEnsemble(
            catboost_model=catboost_model,
            xgboost_model=xgboost_model,
            lightgbm_model=lightgbm_model,
            strategy="validation_weighted",
            class_names=class_names,
            calibrate=calibrate,
            calibration_method="temperature"
        )
        optimal_weights = ensemble_weighted.optimize_weights_on_validation(
            val_probas_cat, val_probas_xgb, val_probas_lgb, y_val
        )
        weighted_probas = (
            optimal_weights[0] * val_probas_cat +
            optimal_weights[1] * val_probas_xgb +
            optimal_weights[2] * val_probas_lgb
        )
        if calibrate:
            ensemble_weighted.fit_calibration(weighted_probas, y_val)
            weighted_probas = ensemble_weighted.calibrator.transform(weighted_probas)

        weighted_preds = np.argmax(weighted_probas, axis=1)
        metrics_weighted = MetricsCalculator.compute_all_metrics(y_val, weighted_preds, weighted_probas, class_names)
        comparison["validation_weighted"] = {
            **metrics_weighted,
            "optimal_weights": {
                "catboost": round(float(optimal_weights[0]), 4),
                "xgboost": round(float(optimal_weights[1]), 4),
                "lightgbm": round(float(optimal_weights[2]), 4)
            },
            "temperature": round(float(getattr(ensemble_weighted.calibrator, "temperature_", 1.0)), 4) if calibrate else 1.0
        }
        logger.info(f"SLSQP Weighted Blending Validation Macro-F1: {metrics_weighted['macro_f1']}")

        best_strategy = "validation_weighted"
        if metrics_unif["macro_f1"] > metrics_weighted["macro_f1"] + 1e-4:
            best_strategy = "uniform"
            ensemble_final = HybridGBDTEnsemble(
                catboost_model=catboost_model,
                xgboost_model=xgboost_model,
                lightgbm_model=lightgbm_model,
                strategy="uniform",
                class_names=class_names,
                calibrate=calibrate,
                calibration_method="temperature"
            )
            if calibrate:
                ensemble_final.fit_calibration(unif_probas, y_val)
        else:
            ensemble_final = ensemble_weighted

        logger.info(f"Selected Winning Ensemble Architecture: {best_strategy}")
        return ensemble_final, comparison
