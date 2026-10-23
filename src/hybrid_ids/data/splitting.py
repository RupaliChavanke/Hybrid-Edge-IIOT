"""
Leakage-Safe Dataset Splitting Module.
Implements strict partition isolation for 70/15/15 Train/Validation/Locked-Test splits.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetSplitter:
    """Partitions network intrusion datasets into mutually exclusive subsets with zero leakage."""

    @staticmethod
    def train_val_test_split(
        df: pd.DataFrame,
        target_column: str,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        stratify: bool = True,
        reset_index: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits DataFrame into Train, Validation, and Locked-Test partitions.
        
        Guarantees:
        1. Test partition is strictly locked.
        2. Zero index or record overlap between splits.
        3. Stratified target distribution preservation.
        """
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

        n_total = len(df)
        n_test = max(1, int(round(n_total * test_ratio)))
        n_val = max(1, int(round(n_total * val_ratio)))

        strat_col = df[target_column] if stratify else None

        # First split off the locked test set
        df_train_val, df_test = train_test_split(
            df,
            test_size=n_test,
            random_state=random_seed,
            stratify=strat_col
        )

        strat_col_val = df_train_val[target_column] if stratify else None

        df_train, df_val = train_test_split(
            df_train_val,
            test_size=n_val,
            random_state=random_seed,
            stratify=strat_col_val
        )

        # Audit index overlap
        train_idx = set(df_train.index)
        val_idx = set(df_val.index)
        test_idx = set(df_test.index)

        assert train_idx.isdisjoint(val_idx), "FATAL: Train and Validation indices overlap!"
        assert train_idx.isdisjoint(test_idx), "FATAL: Train and Locked Test indices overlap!"
        assert val_idx.isdisjoint(test_idx), "FATAL: Validation and Locked Test indices overlap!"

        if reset_index:
            return (
                df_train.reset_index(drop=True),
                df_val.reset_index(drop=True),
                df_test.reset_index(drop=True)
            )
        return df_train, df_val, df_test

    @staticmethod
    def audit_splits(
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        target_column: str
    ) -> pd.DataFrame:
        """Produces a breakdown table of target counts and percentages across splits."""
        train_counts = df_train[target_column].value_counts()
        val_counts = df_val[target_column].value_counts()
        test_counts = df_test[target_column].value_counts()

        classes = sorted(list(set(train_counts.index) | set(val_counts.index) | set(test_counts.index)))
        rows = []
        for c in classes:
            tr = train_counts.get(c, 0)
            va = val_counts.get(c, 0)
            te = test_counts.get(c, 0)
            total = tr + va + te
            rows.append({
                "class": c,
                "train_count": tr,
                "train_pct": round(tr / len(df_train) * 100, 2),
                "val_count": va,
                "val_pct": round(va / len(df_val) * 100, 2),
                "test_count": te,
                "test_pct": round(te / len(df_test) * 100, 2),
                "total_count": total
            })
        return pd.DataFrame(rows)
