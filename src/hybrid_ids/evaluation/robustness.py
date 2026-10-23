"""
Robustness and Perturbation Evaluation Module.
Tests model stability under Gaussian noise injection and feature dropout perturbations.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


class RobustnessEvaluator:
    """Evaluates classifier resilience against adversarial or degraded operational inputs."""

    @staticmethod
    def evaluate_noise_perturbation(
        model: Any,
        X_test: pd.DataFrame | np.ndarray,
        y_test: np.ndarray,
        noise_levels: List[float] = [0.01, 0.05, 0.10],
        random_seed: int = 42
    ) -> List[Dict[str, Any]]:
        """Evaluates predictive degradation under additive Gaussian noise."""
        rng = np.random.RandomState(random_seed)
        X_arr = X_test.values if isinstance(X_test, pd.DataFrame) else X_test.copy()
        feat_stds = np.std(X_arr, axis=0, keepdims=True)
        feat_stds[feat_stds == 0] = 1.0

        results = []
        # Baseline clean performance
        base_preds = model.predict(X_arr)
        results.append({
            "perturbation": "None (Clean)",
            "noise_level": 0.0,
            "accuracy": round(float(accuracy_score(y_test, base_preds)), 4),
            "macro_f1": round(float(f1_score(y_test, base_preds, average="macro", zero_division=0)), 4)
        })

        for sigma in noise_levels:
            noise = rng.normal(0.0, sigma, size=X_arr.shape) * feat_stds
            X_perturbed = X_arr + noise
            preds = model.predict(X_perturbed)
            results.append({
                "perturbation": f"Gaussian Noise (sigma={sigma})",
                "noise_level": sigma,
                "accuracy": round(float(accuracy_score(y_test, preds)), 4),
                "macro_f1": round(float(f1_score(y_test, preds, average="macro", zero_division=0)), 4)
            })

        return results

    @staticmethod
    def evaluate_feature_dropout(
        model: Any,
        X_test: pd.DataFrame | np.ndarray,
        y_test: np.ndarray,
        dropout_rates: List[float] = [0.05, 0.10, 0.20],
        random_seed: int = 42
    ) -> List[Dict[str, Any]]:
        """Evaluates predictive degradation under random missing feature masking (dropout)."""
        rng = np.random.RandomState(random_seed)
        X_arr = X_test.values if isinstance(X_test, pd.DataFrame) else X_test.copy()

        results = []
        for rate in dropout_rates:
            mask = rng.binomial(1, 1.0 - rate, size=X_arr.shape)
            X_dropped = X_arr * mask
            preds = model.predict(X_dropped)
            results.append({
                "perturbation": f"Feature Dropout (rate={rate})",
                "dropout_rate": rate,
                "accuracy": round(float(accuracy_score(y_test, preds)), 4),
                "macro_f1": round(float(f1_score(y_test, preds, average="macro", zero_division=0)), 4)
            })

        return results
