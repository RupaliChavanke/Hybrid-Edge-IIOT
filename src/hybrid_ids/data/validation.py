"""
Data Quality Validation Module.
Audits incoming network telemetry DataFrames for nulls, infs, constant features, duplicates, and class imbalance.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np


class DataValidator:
    """Validates data hygiene and detects potential anomalies or leaks."""

    @staticmethod
    def audit_quality(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        """Runs comprehensive data quality checks on a DataFrame."""
        total_rows, total_cols = df.shape

        # Null / NaN counts
        null_counts = df.isnull().sum()
        cols_with_null = null_counts[null_counts > 0].to_dict()

        # Infinite values in numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        inf_counts = {}
        for col in num_cols:
            n_inf = np.isinf(df[col]).sum()
            if n_inf > 0:
                inf_counts[col] = int(n_inf)

        # Constant and near-constant columns
        constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

        # Duplicate row count
        num_duplicates = int(df.duplicated().sum())

        # Class balance if target specified
        class_distribution = {}
        minority_classes = []
        if target_col and target_col in df.columns:
            counts = df[target_col].value_counts().to_dict()
            total_targets = sum(counts.values())
            for cls_name, cnt in counts.items():
                ratio = cnt / total_targets
                class_distribution[str(cls_name)] = {
                    "count": int(cnt),
                    "percentage": round(ratio * 100, 3)
                }
                if ratio < 0.02:
                    minority_classes.append(str(cls_name))

        # Memory usage
        mem_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

        return {
            "total_rows": total_rows,
            "total_cols": total_cols,
            "memory_mb": mem_mb,
            "null_column_counts": cols_with_null,
            "inf_column_counts": inf_counts,
            "constant_columns": constant_cols,
            "duplicate_rows": num_duplicates,
            "target_distribution": class_distribution,
            "minority_classes": minority_classes
        }
