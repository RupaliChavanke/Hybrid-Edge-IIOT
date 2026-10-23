"""
Leakage-Safe Data Preparation and Preprocessing Script.
Loads raw data, performs stratified train/val/locked-test splitting (70/15/15),
fits cleaner, encoder, and feature selector strictly on training partition, and persists processed artifacts.
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.hybrid_ids.data.loaders import DataLoader
from src.hybrid_ids.data.splitting import DatasetSplitter
from src.hybrid_ids.preprocessing.cleaning import PipelineCleaner
from src.hybrid_ids.preprocessing.encoding import TargetLabelEncoder
from src.hybrid_ids.preprocessing.feature_selection import FeatureSelector
from src.hybrid_ids.inference.model_registry import ModelRegistry
from src.hybrid_ids.config import get_project_root
from src.hybrid_ids.logging_config import setup_logger

logger = setup_logger("prepare_data")


def prepare_edge_iiot_pipeline(
    mode: str = "full",
    sample_size: int = None,
    k_features: int = 33
):
    root = get_project_root()
    loader = DataLoader(root)
    registry = ModelRegistry()
    proc_dir = root / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Preparing Edge-IIoTset dataset (mode={mode})...")
    df_raw = loader.load_edge_iiot(mode=mode, sample_size=sample_size)

    # 1. Leakage-safe train/val/test splitting
    logger.info("Executing 70/15/15 stratified train/val/locked-test split...")
    df_train, df_val, df_test = DatasetSplitter.train_val_test_split(
        df_raw,
        target_column="Attack_type",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        stratify=True
    )
    logger.info(f"Split sizes: Train={len(df_train)}, Val={len(df_val)}, Locked-Test={len(df_test)}")

    # Save split audit
    split_audit = DatasetSplitter.audit_splits(df_train, df_val, df_test, "Attack_type")
    split_audit.to_csv(root / "reports" / "split_audit.csv", index=False)

    # 2. Fit label encoder ONLY on training labels
    logger.info("Fitting TargetLabelEncoder on training labels...")
    label_encoder = TargetLabelEncoder()
    y_train = label_encoder.fit_transform(df_train["Attack_type"])
    y_val = label_encoder.transform(df_val["Attack_type"])
    y_test = label_encoder.transform(df_test["Attack_type"])

    # 3. Fit cleaner ONLY on training features
    logger.info("Fitting PipelineCleaner on training features...")
    cleaner = PipelineCleaner(drop_metadata=True)
    cleaner.fit(df_train, target_cols=["Attack_type", "Attack_label"])

    X_train_clean = cleaner.transform(df_train)
    X_val_clean = cleaner.transform(df_val)
    X_test_clean = cleaner.transform(df_test)

    # 4. Fit feature selector ONLY on training features and training targets
    logger.info(f"Fitting FeatureSelector (k={k_features}) on training partition...")
    selector = FeatureSelector(method="mutual_info", k_features=k_features)
    selector.fit(X_train_clean, y_train)

    X_train_sel = selector.transform(X_train_clean)
    X_val_sel = selector.transform(X_val_clean)
    X_test_sel = selector.transform(X_test_clean)

    logger.info(f"Selected {len(selector.selected_features_)} features: {selector.selected_features_}")

    # 5. Persist processed datasets and transformers
    np.savez_compressed(
        proc_dir / "edge_iiot_processed.npz",
        X_train=X_train_sel.values,
        y_train=y_train,
        X_val=X_val_sel.values,
        y_val=y_val,
        X_test=X_test_sel.values,
        y_test=y_test,
        feature_names=np.array(selector.selected_features_),
        class_names=np.array(label_encoder.classes_)
    )

    registry.save_artifact(cleaner, "cleaner", version="edge_iiot")
    registry.save_artifact(label_encoder, "label_encoder", version="edge_iiot")
    registry.save_artifact(selector, "feature_selector", version="edge_iiot")

    logger.info(f"Data preparation complete. Artifacts saved to {proc_dir} and models/registry/")


def main():
    parser = argparse.ArgumentParser(description="Leakage-Safe Data Preparation")
    parser.add_argument("--mode", type=str, default="full", choices=["smoke", "dev", "full"], help="Processing mode")
    parser.add_argument("--sample-size", type=int, default=None, help="Explicit sample row count override")
    parser.add_argument("--k-features", type=int, default=33, help="Number of selected features")
    args = parser.parse_args()

    prepare_edge_iiot_pipeline(mode=args.mode, sample_size=args.sample_size, k_features=args.k_features)


if __name__ == "__main__":
    main()
