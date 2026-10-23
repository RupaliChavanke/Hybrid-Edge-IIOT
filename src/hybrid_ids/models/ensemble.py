"""
Validation-Based Hybrid Ensemble Architecture.
Combines CatBoost, XGBoost, and LightGBM with SLSQP simplex weight optimization,
out-of-fold stacking meta-classification, and post-hoc probability calibration.
"""

from typing import Dict, List, Optional, Tuple, Any
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from src.hybrid_ids.logging_config import setup_logger
from src.hybrid_ids.models.calibration import MulticlassProbabilityCalibrator

logger = setup_logger("ensemble")


class HybridGBDTEnsemble:
    """Hybrid Gradient-Boosted Decision Tree Ensemble (CatBoost + XGBoost + LightGBM)."""

    def __init__(
        self,
        catboost_model: Any = None,
        xgboost_model: Any = None,
        lightgbm_model: Any = None,
        strategy: str = "validation_weighted",
        class_names: Optional[List[str]] = None,
        calibrate: bool = True,
        calibration_method: str = "temperature"
    ):
        self.catboost = catboost_model
        self.xgboost = xgboost_model
        self.lightgbm = lightgbm_model
        self.strategy = strategy
        self.class_names = class_names or []
        self.weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        self.meta_classifier: Optional[Any] = None
        self.calibrator = MulticlassProbabilityCalibrator(method=calibration_method) if calibrate else None
        self.is_fitted: bool = False
        self.version: str = "2.1.0"

    def optimize_weights_on_validation(
        self,
        val_probas_cat: np.ndarray,
        val_probas_xgb: np.ndarray,
        val_probas_lgb: np.ndarray,
        y_val: np.ndarray
    ) -> np.ndarray:
        """Finds non-negative simplex weights w = [w_cat, w_xgb, w_lgb] via SLSQP minimizing multiclass log-loss on validation."""
        eps = 1e-15

        def loss_func(w):
            blended_p = (
                w[0] * val_probas_cat +
                w[1] * val_probas_xgb +
                w[2] * val_probas_lgb
            )
            blended_p = np.clip(blended_p, eps, 1.0 - eps)
            blended_p = blended_p / blended_p.sum(axis=1, keepdims=True)
            return float(log_loss(y_val, blended_p))

        init_w = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        constraints = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
        res = minimize(loss_func, init_w, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 200})

        optimal_w = np.maximum(0.0, res.x)
        if np.sum(optimal_w) > 0:
            optimal_w = optimal_w / np.sum(optimal_w)
        else:
            optimal_w = init_w

        self.weights = optimal_w
        self.is_fitted = True
        logger.info(f"Optimal SLSQP ensemble weights on validation: CatBoost={self.weights[0]:.4f}, XGBoost={self.weights[1]:.4f}, LightGBM={self.weights[2]:.4f}")
        return self.weights

    def fit_stacking_meta_classifier(
        self,
        oof_probas_cat: np.ndarray,
        oof_probas_xgb: np.ndarray,
        oof_probas_lgb: np.ndarray,
        y_train: np.ndarray,
        regularization_C: float = 1.0
    ) -> None:
        """Trains regularized stacking meta-classifier on out-of-fold probabilities."""
        X_meta = np.hstack([oof_probas_cat, oof_probas_xgb, oof_probas_lgb])
        self.meta_classifier = LogisticRegression(
            C=regularization_C,
            solver="lbfgs",
            max_iter=500,
            random_state=42
        )
        self.meta_classifier.fit(X_meta, y_train)
        self.is_fitted = True
        logger.info("Stacking meta-classifier fitted on out-of-fold predictions.")

    def fit_calibration(self, val_probas: np.ndarray, y_val: np.ndarray) -> None:
        """Fits probability calibration strictly on held-out validation predictions."""
        if self.calibrator is not None:
            self.calibrator.fit(val_probas, y_val)
            self.is_fitted = True
            logger.info(f"Probability calibrator ({self.calibrator.method}) fitted on validation predictions.")

    def predict_proba_components(self, X: pd.DataFrame | np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns base probabilities from all three classifiers."""
        p_cat = self.catboost.predict_proba(X)
        p_xgb = self.xgboost.predict_proba(X)
        p_lgb = self.lightgbm.predict_proba(X)
        return p_cat, p_xgb, p_lgb

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Returns combined, calibrated ensemble probability distribution."""
        p_cat, p_xgb, p_lgb = self.predict_proba_components(X)

        if self.strategy == "stacking" and self.meta_classifier is not None:
            X_meta = np.hstack([p_cat, p_xgb, p_lgb])
            blended = self.meta_classifier.predict_proba(X_meta)
        elif self.strategy == "uniform":
            blended = (p_cat + p_xgb + p_lgb) / 3.0
        else:
            blended = (
                self.weights[0] * p_cat +
                self.weights[1] * p_xgb +
                self.weights[2] * p_lgb
            )

        if self.calibrator is not None and self.calibrator.is_fitted:
            blended = self.calibrator.transform(blended)

        return blended

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Returns predicted class indices."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def predict_event(self, feature_row: pd.Series | Dict[str, Any] | np.ndarray) -> Dict[str, Any]:
        """Real-time production inference interface returning prediction, confidence, latency, and alert severity."""
        t_start = time.perf_counter()

        if isinstance(feature_row, dict):
            X_input = pd.DataFrame([feature_row])
        elif isinstance(feature_row, pd.Series):
            X_input = pd.DataFrame([feature_row.to_dict()])
        elif isinstance(feature_row, np.ndarray):
            X_input = feature_row.reshape(1, -1) if feature_row.ndim == 1 else feature_row
        else:
            X_input = feature_row

        probas = self.predict_proba(X_input)[0]
        class_idx = int(np.argmax(probas))
        confidence = float(probas[class_idx])
        pred_label = self.class_names[class_idx] if class_idx < len(self.class_names) else str(class_idx)

        t_end = time.perf_counter()
        latency_ms = round((t_end - t_start) * 1000.0, 4)

        is_attack = str(pred_label).lower() not in ("benign", "normal", "benigntraffic")
        if not is_attack:
            severity = "INFORMATIONAL"
        elif confidence >= 0.85:
            severity = "CRITICAL"
        elif confidence >= 0.65:
            severity = "HIGH"
        else:
            severity = "MEDIUM"

        return {
            "predicted_class_idx": class_idx,
            "predicted_label": pred_label,
            "confidence": round(confidence, 4),
            "is_attack": is_attack,
            "severity": severity,
            "latency_ms": latency_ms,
            "model_version": self.version,
            "class_probabilities": {
                (self.class_names[i] if i < len(self.class_names) else str(i)): round(float(p), 4)
                for i, p in enumerate(probas)
            }
        }
