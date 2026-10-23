"""
Training Script for CatBoost, XGBoost, LightGBM, and Validation-Based Hybrid Ensemble.
Trains the three gradient-boosted tree architectures, optimizes ensemble weights on validation data,
calibrates probabilities, and exports serialized artifacts.
"""

import argparse
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from src.hybrid_ids.training.train_base_models import BaseModelsTrainer
from src.hybrid_ids.training.train_ensemble import EnsembleTrainer
from src.hybrid_ids.training.experiment_tracker import ExperimentTracker
from src.hybrid_ids.inference.model_registry import ModelRegistry
from src.hybrid_ids.utils.reproducibility import set_seed
from src.hybrid_ids.config import get_project_root, load_yaml
from src.hybrid_ids.logging_config import setup_logger

logger = setup_logger("train_models")


def main():
    parser = argparse.ArgumentParser(description="Train GBDT Models & Hybrid Ensemble")
    parser.add_argument("--data-file", type=str, default="data/processed/edge_iiot_processed.npz")
    parser.add_argument("--calibrate", action="store_true", default=True, help="Enable Platt calibration")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.random_seed)
    root = get_project_root()
    registry = ModelRegistry()
    tracker = ExperimentTracker()

    data_path = root / args.data_file
    if not data_path.exists():
        logger.error(f"Processed dataset not found at {data_path}. Run scripts/prepare_data.py first!")
        return

    logger.info(f"Loading preprocessed partitions from: {data_path}")
    data = np.load(data_path, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    feature_names = list(data["feature_names"])
    class_names = list(data["class_names"])

    logger.info(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Features: {len(feature_names)}, Classes: {len(class_names)}")

    # Load model hyperparameter configurations
    models_cfg = load_yaml("configs/models.yaml")
    cat_cfg = models_cfg.get("catboost", {})
    xgb_cfg = models_cfg.get("xgboost", {})
    lgb_cfg = models_cfg.get("lightgbm", {})

    t_start = time.perf_counter()

    # 1. Train Base GBDT Models
    trainer = BaseModelsTrainer(cat_cfg, xgb_cfg, lgb_cfg)
    base_results = trainer.fit_all(X_train, y_train, X_val, y_val, class_names=class_names)

    # 2. Train & Optimize Validation-Based Hybrid Ensemble
    logger.info("--- Constructing Validation-Based Hybrid Ensemble ---")
    ensemble, ensemble_val_comparison = EnsembleTrainer.train_and_select_ensemble(
        catboost_model=trainer.catboost,
        xgboost_model=trainer.xgboost,
        lightgbm_model=trainer.lightgbm,
        val_probas_cat=base_results["catboost"]["val_probas"],
        val_probas_xgb=base_results["xgboost"]["val_probas"],
        val_probas_lgb=base_results["lightgbm"]["val_probas"],
        y_val=y_val,
        class_names=class_names,
        calibrate=args.calibrate
    )

    t_total = round(time.perf_counter() - t_start, 2)
    logger.info(f"All training completed in {t_total} seconds.")

    # 3. Save Artifacts to Model Registry and checkpoints/
    logger.info("Registering model artifacts...")
    p_cat, h_cat = registry.save_artifact(trainer.catboost, "catboost_model", version="v2")
    p_xgb, h_xgb = registry.save_artifact(trainer.xgboost, "xgboost_model", version="v2")
    p_lgb, h_lgb = registry.save_artifact(trainer.lightgbm, "lightgbm_model", version="v2")
    p_ens, h_ens = registry.save_artifact(ensemble, "hybrid_ensemble", version="v2")

    # Save validation summary metrics
    val_summary = {
        "catboost": {
            "macro_f1": base_results["catboost"]["metrics"]["macro_f1"],
            "accuracy": base_results["catboost"]["metrics"]["accuracy"],
            "duration_sec": base_results["catboost"]["timing"]["duration_sec"]
        },
        "xgboost": {
            "macro_f1": base_results["xgboost"]["metrics"]["macro_f1"],
            "accuracy": base_results["xgboost"]["metrics"]["accuracy"],
            "duration_sec": base_results["xgboost"]["timing"]["duration_sec"]
        },
        "lightgbm": {
            "macro_f1": base_results["lightgbm"]["metrics"]["macro_f1"],
            "accuracy": base_results["lightgbm"]["metrics"]["accuracy"],
            "duration_sec": base_results["lightgbm"]["timing"]["duration_sec"]
        },
        "ensemble_validation_comparison": ensemble_val_comparison,
        "total_training_duration_sec": t_total
    }

    metrics_dir = root / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(val_summary, f, indent=2)

    logger.info("Validation summary saved to artifacts/metrics/validation_summary.json")


if __name__ == "__main__":
    main()
