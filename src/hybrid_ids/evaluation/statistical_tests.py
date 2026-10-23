"""
Statistical Validity and Significance Testing Module.
Implements paired bootstrap confidence intervals and McNemar's test for classifier comparisons.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from scipy import stats


class StatisticalTester:
    """Performs statistical tests and confidence interval estimation on test predictions."""

    @staticmethod
    def bootstrap_metric_ci(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bootstraps: int = 500,
        confidence_level: float = 0.95,
        random_seed: int = 42
    ) -> Dict[str, Tuple[float, float]]:
        """Calculates non-parametric bootstrap confidence intervals for Accuracy and Macro-F1."""
        rng = np.random.RandomState(random_seed)
        n_samples = len(y_true)

        acc_bootstraps = []
        f1_bootstraps = []

        for _ in range(n_bootstraps):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            y_t_boot = y_true[indices]
            y_p_boot = y_pred[indices]

            acc_bootstraps.append(accuracy_score(y_t_boot, y_p_boot))
            f1_bootstraps.append(f1_score(y_t_boot, y_p_boot, average="macro", zero_division=0))

        alpha = (1.0 - confidence_level) / 2.0
        acc_ci = (
            round(float(np.percentile(acc_bootstraps, 100 * alpha)), 4),
            round(float(np.percentile(acc_bootstraps, 100 * (1.0 - alpha))), 4)
        )
        f1_ci = (
            round(float(np.percentile(f1_bootstraps, 100 * alpha)), 4),
            round(float(np.percentile(f1_bootstraps, 100 * (1.0 - alpha))), 4)
        )

        return {
            "accuracy_ci": acc_ci,
            "macro_f1_ci": f1_ci
        }

    @staticmethod
    def mcnemar_paired_test(
        y_true: np.ndarray,
        y_pred_a: np.ndarray,
        y_pred_b: np.ndarray
    ) -> Dict[str, Any]:
        """Runs McNemar's test with Edwards continuity correction using scipy.stats.
        
        Tests H0: Classifier A and Classifier B have identical error rates on held-out test instances.
        """
        correct_a = (y_pred_a == y_true)
        correct_b = (y_pred_b == y_true)

        # b = A correct, B incorrect
        b = int(np.sum(correct_a & (~correct_b)))
        # c = A incorrect, B correct
        c = int(np.sum((~correct_a) & correct_b))

        table = [[int(np.sum(correct_a & correct_b)), b], [c, int(np.sum((~correct_a) & (~correct_b)))]]

        if (b + c) == 0:
            stat = 0.0
            pval = 1.0
        else:
            # Edwards continuity correction
            stat = float((abs(b - c) - 1.0) ** 2 / (b + c))
            pval = float(1.0 - stats.chi2.cdf(stat, df=1))

        is_significant = pval < 0.05
        return {
            "contingency_table": table,
            "statistic": round(stat, 4),
            "p_value": round(pval, 6),
            "is_significant_at_05": is_significant
        }
