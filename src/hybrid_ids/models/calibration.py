"""
Probability Calibration Module for Intrusion Detection Models.
Calibrates tree-based ensemble probabilities on validation data using Temperature Scaling,
Platt scaling (sigmoid), or Isotonic Regression.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


class MulticlassProbabilityCalibrator:
    """Calibrates multiclass predicted probabilities on held-out validation sets."""

    def __init__(self, method: str = "temperature"):
        self.method = method
        self.calibrators_: List[Any] = []
        self.temperature_: float = 1.0
        self.is_fitted: bool = False

    def fit(self, probas: np.ndarray, y_true: np.ndarray) -> "MulticlassProbabilityCalibrator":
        """Fits Temperature, Platt, or Isotonic calibrators strictly on validation probabilities.
        
        Args:
            probas: Matrix of shape (N, C) containing predicted validation probabilities.
            y_true: Ground truth class integer indices of shape (N,).
        """
        n_samples, n_classes = probas.shape
        self.calibrators_ = []

        if self.method == "temperature":
            eps = 1e-12
            log_p = np.log(np.clip(probas, eps, 1.0))

            def nll(T):
                t = max(0.05, float(T[0]))
                scaled = softmax(log_p / t, axis=1)
                return float(log_loss(y_true, scaled))

            res = minimize(nll, [1.0], method="Nelder-Mead")
            self.temperature_ = float(max(0.05, res.x[0]))
            self.is_fitted = True
            return self

        for c in range(n_classes):
            binary_y = (y_true == c).astype(int)
            p_c = probas[:, c]
            if self.method == "isotonic":
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(p_c, binary_y)
                self.calibrators_.append(iso)
            else:
                eps = 1e-7
                log_odds = np.log(np.clip(p_c, eps, 1.0 - eps) / np.clip(1.0 - p_c, eps, 1.0 - eps)).reshape(-1, 1)
                lr = LogisticRegression(C=1.0, solver="lbfgs")
                lr.fit(log_odds, binary_y)
                self.calibrators_.append(lr)

        self.is_fitted = True
        return self

    def transform(self, probas: np.ndarray) -> np.ndarray:
        """Applies calibration and normalizes probabilities to sum to 1."""
        if not self.is_fitted:
            return probas

        if self.method == "temperature":
            eps = 1e-12
            log_p = np.log(np.clip(probas, eps, 1.0))
            return softmax(log_p / getattr(self, "temperature_", 1.0), axis=1)

        n_samples, n_classes = probas.shape
        calibrated = np.zeros_like(probas)

        for c in range(n_classes):
            p_c = probas[:, c]
            if self.method == "isotonic":
                calibrated[:, c] = self.calibrators_[c].predict(p_c)
            else:
                eps = 1e-7
                log_odds = np.log(np.clip(p_c, eps, 1.0 - eps) / np.clip(1.0 - p_c, eps, 1.0 - eps)).reshape(-1, 1)
                calibrated[:, c] = self.calibrators_[c].predict_proba(log_odds)[:, 1]

        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return calibrated / row_sums
