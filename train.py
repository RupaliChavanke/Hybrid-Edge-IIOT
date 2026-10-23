"""
Standalone Training Entry Point for Hybrid Intrusion Detection Framework.
Trains CatBoost, XGBoost, LightGBM, and Validation-Based Hybrid Ensemble.
"""

import sys
from pathlib import Path
import argparse

# Ensure project root in path
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from scripts.prepare_data import prepare_edge_iiot_pipeline
from scripts.train_models import main as train_main


def main():
    parser = argparse.ArgumentParser(description="Hybrid IDS Training Pipeline")
    parser.add_argument("--mode", type=str, default="full", choices=["smoke", "dev", "full"], help="Data mode")
    parser.add_argument("--k-features", type=int, default=25, help="Number of selected features")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip data preparation if already processed")
    args = parser.parse_args()

    processed_path = root / "data" / "processed" / "edge_iiot_processed.npz"
    if not processed_path.exists() or not args.skip_prepare:
        print(f"Executing data preparation in '{args.mode}' mode...")
        prepare_edge_iiot_pipeline(mode=args.mode, k_features=args.k_features)

    print("Executing GBDT base models and ensemble training...")
    train_main()


if __name__ == "__main__":
    main()
