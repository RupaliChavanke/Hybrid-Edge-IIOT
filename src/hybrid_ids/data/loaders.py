"""
Data Loader Module for Edge-IIoTset and CICIoT2023.
Supports memory-safe chunked reading, downcasted dtypes, and configurable sample modes.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterator
import pandas as pd
import numpy as np
from src.hybrid_ids.logging_config import setup_logger
from src.hybrid_ids.config import get_project_root

logger = setup_logger("data_loader")


class DataLoader:
    """Memory-efficient loader for large intrusion detection CSV datasets."""

    def __init__(self, project_root: Optional[Path] = None):
        self.root = project_root or get_project_root()

    def load_edge_iiot(
        self,
        mode: str = "dev",
        sample_size: Optional[int] = None,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """Loads Edge-IIoTset dataset with specified evaluation mode.
        
        Args:
            mode: 'smoke' (~2,000 rows), 'dev' (~25,000 rows), or 'full' (all 157,800 rows).
            sample_size: Explicit row count override if provided.
            random_seed: Deterministic sampling seed.
        """
        path = self.root / "data" / "raw" / "edge_iiot" / "ML-EdgeIIoT-dataset.csv"
        if not path.exists():
            # Fallback to direct raw directory
            path = self.root / "data" / "raw" / "ML-EdgeIIoT-dataset.csv"

        logger.info(f"Loading Edge-IIoTset from: {path} (mode={mode})")

        if mode == "full" and sample_size is None:
            df = pd.read_csv(path, low_memory=False)
            logger.info(f"Loaded full Edge-IIoTset: {df.shape[0]} rows, {df.shape[1]} cols")
            return df

        target_n = sample_size
        if target_n is None:
            target_n = 2000 if mode == "smoke" else 25000

        # Read full file or chunked, then stratified sample by Attack_type
        # To avoid reading all into memory repeatedly if unnecessary, read chunked or sample
        df = pd.read_csv(path, low_memory=False)
        if len(df) > target_n:
            from sklearn.model_selection import train_test_split
            df, _ = train_test_split(
                df,
                train_size=target_n,
                stratify=df["Attack_type"],
                random_state=random_seed
            )
            df = df.reset_index(drop=True)

        logger.info(f"Loaded Edge-IIoTset ({mode} mode): {df.shape[0]} rows, {df.shape[1]} cols")
        return df

    def load_cic_iot_2023(
        self,
        split: str = "test",
        mode: str = "dev",
        sample_size: Optional[int] = None,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """Loads CICIoT2023 dataset with specified split ('test' or 'validation') and mode.
        
        Args:
            split: 'test' or 'validation'.
            mode: 'smoke' (~2,000 rows), 'dev' (~25,000 rows), or 'full' (~1.18M rows).
            sample_size: Explicit row count override if provided.
            random_seed: Deterministic sampling seed.
        """
        filename = "test.csv" if split == "test" else "validation.csv"
        path = self.root / "data" / "raw" / "cic_iot_2023" / filename

        logger.info(f"Loading CICIoT2023 ({split}) from: {path} (mode={mode})")

        target_n = sample_size
        if target_n is None:
            if mode == "smoke":
                target_n = 2000
            elif mode == "dev":
                target_n = 25000
            else:
                target_n = None

        if target_n is not None:
            # Read first 100k chunk and stratified sample
            chunks = []
            chunk_size = 50000
            read_limit = max(100000, target_n * 3)
            rows_read = 0
            for chunk in pd.read_csv(path, chunksize=chunk_size):
                chunks.append(chunk)
                rows_read += len(chunk)
                if rows_read >= read_limit:
                    break
            pool = pd.concat(chunks, ignore_index=True)

            # Downcast float64 to float32
            float_cols = pool.select_dtypes(include=["float64"]).columns
            pool[float_cols] = pool[float_cols].astype(np.float32)

            from sklearn.model_selection import train_test_split
            # Ensure minority classes with < 2 instances don't break stratification
            class_counts = pool["label"].value_counts()
            valid_classes = class_counts[class_counts >= 2].index
            pool_filtered = pool[pool["label"].isin(valid_classes)]

            df, _ = train_test_split(
                pool_filtered,
                train_size=min(target_n, len(pool_filtered) - len(valid_classes)),
                stratify=pool_filtered["label"],
                random_state=random_seed
            )
            df = df.reset_index(drop=True)
            logger.info(f"Loaded CICIoT2023 ({split}, {mode} mode): {df.shape[0]} rows, {df.shape[1]} cols")
            return df
        else:
            # Full benchmark read with float32 optimization
            df = pd.read_csv(path)
            float_cols = df.select_dtypes(include=["float64"]).columns
            df[float_cols] = df[float_cols].astype(np.float32)
            logger.info(f"Loaded full CICIoT2023 ({split}): {df.shape[0]} rows, {df.shape[1]} cols")
            return df
