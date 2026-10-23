"""
Comprehensive Evaluation Metrics Module for Multiclass Intrusion Detection.
Calculates Accuracy, Balanced Accuracy, Macro/Weighted F1, Per-Class Metrics,
Confusion Matrices, ROC-AUC, PR-AUC, One-vs-Rest Macro FPR/FNR, True Binary Benign-to-Attack FPR/FNR,
Log-Loss, Multiclass Brier Score, and Expected Calibration Error (ECE).
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    log_loss
)


class MetricsCalculator:
    """Calculates publication-standard multiclass intrusion detection metrics."""

    @staticmethod
    def compute_calibration_metrics(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
        """Computes Multiclass Log-Loss, Multiclass Brier Score, ECE, and MCE."""
        n_samples, n_classes = y_proba.shape
        y_oh = np.eye(n_classes)[y_true]
        brier = float(np.mean(np.sum((y_proba - y_oh) ** 2, axis=1)))
        ll = float(log_loss(y_true, y_proba, labels=np.arange(n_classes)))

        confidences = np.max(y_proba, axis=1)
        predictions = np.argmax(y_proba, axis=1)
        accuracies = (predictions == y_true).astype(float)

        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece, mce = 0.0, 0.0
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = float(np.mean(in_bin))
            if prop_in_bin > 0:
                diff = abs(float(np.mean(accuracies[in_bin])) - float(np.mean(confidences[in_bin])))
                ece += diff * prop_in_bin
                mce = max(mce, diff)

        return {
            "log_loss": round(ll, 5),
            "brier_score": round(brier, 5),
            "ece": round(float(ece), 5),
            "mce": round(float(mce), 5)
        }

    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Computes comprehensive suite of classification and calibration metrics."""
        unique_classes = np.unique(y_true)
        n_classes = len(class_names) if class_names is not None else len(unique_classes)
        names = class_names or [str(i) for i in range(n_classes)]

        acc = float(accuracy_score(y_true, y_pred))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))

        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=np.arange(len(names)), zero_division=0
        )

        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(names)))
        with np.errstate(divide="ignore", invalid="ignore"):
            norm_cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            norm_cm = np.nan_to_num(norm_cm)

        # One-vs-Rest (OvR) False Positive Rate (FPR) and False Negative Rate (FNR)
        fp_per_class = cm.sum(axis=0) - np.diag(cm)
        fn_per_class = cm.sum(axis=1) - np.diag(cm)
        tp_per_class = np.diag(cm)
        tn_per_class = cm.sum() - (fp_per_class + fn_per_class + tp_per_class)

        with np.errstate(divide="ignore", invalid="ignore"):
            fpr_per_class = np.nan_to_num(fp_per_class / (fp_per_class + tn_per_class))
            fnr_per_class = np.nan_to_num(fn_per_class / (fn_per_class + tp_per_class))

        macro_fpr = float(np.mean(fpr_per_class))
        macro_fnr = float(np.mean(fnr_per_class))

        # True Binary Benign-to-Attack FPR and FNR (identifying Normal/Benign index explicitly)
        benign_idx = 0
        for idx, nm in enumerate(names):
            if str(nm).lower() in ("normal", "benign", "benigntraffic"):
                benign_idx = idx
                break

        benign_mask = (y_true == benign_idx)
        attack_mask = (y_true != benign_idx)
        fp_bin = int(np.sum((y_pred != benign_idx) & benign_mask))
        tn_bin = int(np.sum((y_pred == benign_idx) & benign_mask))
        fn_bin = int(np.sum((y_pred == benign_idx) & attack_mask))
        tp_bin = int(np.sum((y_pred != benign_idx) & attack_mask))

        binary_fpr = float(fp_bin / (fp_bin + tn_bin)) if (fp_bin + tn_bin) > 0 else 0.0
        binary_fnr = float(fn_bin / (fn_bin + tp_bin)) if (fn_bin + tp_bin) > 0 else 0.0

        roc_auc = None
        pr_auc = None
        cal_metrics = {}
        if y_proba is not None and y_proba.shape[1] >= len(unique_classes):
            try:
                roc_auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
            except Exception:
                roc_auc = None
            try:
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(y_true, classes=range(y_proba.shape[1]))
                pr_auc = float(average_precision_score(y_bin, y_proba, average="macro"))
            except Exception:
                pr_auc = None
            try:
                cal_metrics = MetricsCalculator.compute_calibration_metrics(y_true, y_proba)
            except Exception:
                cal_metrics = {}

        per_class_dict = {}
        for idx in range(len(names)):
            c_name = names[idx]
            per_class_dict[c_name] = {
                "precision": round(float(per_p[idx]), 4),
                "recall": round(float(per_r[idx]), 4),
                "f1_score": round(float(per_f1[idx]), 4),
                "support": int(per_sup[idx]),
                "fpr": round(float(fpr_per_class[idx]), 4),
                "fnr": round(float(fnr_per_class[idx]), 4)
            }

        return {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "macro_fpr": round(macro_fpr, 4),
            "macro_fnr": round(macro_fnr, 4),
            "binary_fpr": round(binary_fpr, 4),
            "binary_fnr": round(binary_fnr, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
            **cal_metrics,
            "per_class": per_class_dict,
            "confusion_matrix": cm.tolist(),
            "normalized_confusion_matrix": norm_cm.tolist()
        }
