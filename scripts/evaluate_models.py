"""
Locked-Test Evaluation & Cross-Dataset Generalization Benchmark Script.
Evaluates CatBoost, XGBoost, LightGBM, and Hybrid Ensemble strictly on locked test partitions.
Runs statistical significance tests, bootstrap confidence intervals, robustness evaluations,
and cross-dataset generalization experiments (A, B, C, D, E, F).
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.hybrid_ids.inference.model_registry import ModelRegistry
from src.hybrid_ids.evaluation.metrics import MetricsCalculator
from src.hybrid_ids.evaluation.statistical_tests import StatisticalTester
from src.hybrid_ids.evaluation.robustness import RobustnessEvaluator
from src.hybrid_ids.evaluation.cross_dataset import CrossDatasetBenchmark
from src.hybrid_ids.data.loaders import DataLoader
from src.hybrid_ids.models.lightgbm_model import LightGBMModel
from src.hybrid_ids.config import get_project_root
from src.hybrid_ids.logging_config import setup_logger

logger = setup_logger("evaluate_models")


def main():
    parser = argparse.ArgumentParser(description="Locked-Test Evaluation Suite")
    parser.add_argument("--data-file", type=str, default="data/processed/edge_iiot_processed.npz")
    parser.add_argument("--cross-dataset", action="store_true", default=True, help="Run cross-dataset experiments")
    parser.add_argument("--robustness", action="store_true", default=True, help="Run perturbation robustness tests")
    args = parser.parse_args()

    root = get_project_root()
    registry = ModelRegistry()
    metrics_dir = root / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    data_path = root / args.data_file
    logger.info(f"Loading preprocessed locked test partition from: {data_path}")
    data = np.load(data_path, allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = list(data["feature_names"])
    class_names = list(data["class_names"])

    logger.info(f"Locked Test Set: {len(X_test)} samples, {len(feature_names)} features, {len(class_names)} classes")

    # Load trained models from registry
    logger.info("Loading registered models from models/registry/...")
    cat_model = registry.load_artifact("catboost_model", version="v2")
    xgb_model = registry.load_artifact("xgboost_model", version="v2")
    lgb_model = registry.load_artifact("lightgbm_model", version="v2")
    ensemble = registry.load_artifact("hybrid_ensemble", version="v2")

    benchmark_results = {}

    # 1. CatBoost Locked Test Evaluation
    logger.info("Evaluating CatBoost on Locked Test...")
    preds_cat = cat_model.predict(X_test)
    probas_cat = cat_model.predict_proba(X_test)
    metrics_cat = MetricsCalculator.compute_all_metrics(y_test, preds_cat, probas_cat, class_names)
    ci_cat = StatisticalTester.bootstrap_metric_ci(y_test, preds_cat, n_bootstraps=300)
    benchmark_results["catboost"] = {**metrics_cat, "confidence_intervals": ci_cat}

    # 2. XGBoost Locked Test Evaluation
    logger.info("Evaluating XGBoost on Locked Test...")
    preds_xgb = xgb_model.predict(X_test)
    probas_xgb = xgb_model.predict_proba(X_test)
    metrics_xgb = MetricsCalculator.compute_all_metrics(y_test, preds_xgb, probas_xgb, class_names)
    ci_xgb = StatisticalTester.bootstrap_metric_ci(y_test, preds_xgb, n_bootstraps=300)
    benchmark_results["xgboost"] = {**metrics_xgb, "confidence_intervals": ci_xgb}

    # 3. LightGBM Locked Test Evaluation
    logger.info("Evaluating LightGBM on Locked Test...")
    preds_lgb = lgb_model.predict(X_test)
    probas_lgb = lgb_model.predict_proba(X_test)
    metrics_lgb = MetricsCalculator.compute_all_metrics(y_test, preds_lgb, probas_lgb, class_names)
    ci_lgb = StatisticalTester.bootstrap_metric_ci(y_test, preds_lgb, n_bootstraps=300)
    benchmark_results["lightgbm"] = {**metrics_lgb, "confidence_intervals": ci_lgb}

    # 4. Hybrid Ensemble Locked Test Evaluation
    logger.info("Evaluating Validation-Trained Hybrid Ensemble on Locked Test...")
    preds_ens = ensemble.predict(X_test)
    probas_ens = ensemble.predict_proba(X_test)
    metrics_ens = MetricsCalculator.compute_all_metrics(y_test, preds_ens, probas_ens, class_names)
    ci_ens = StatisticalTester.bootstrap_metric_ci(y_test, preds_ens, n_bootstraps=300)
    benchmark_results["hybrid_ensemble"] = {**metrics_ens, "confidence_intervals": ci_ens}

    # 5. Statistical Significance Tests (McNemar's test vs Base Models)
    logger.info("Computing McNemar's paired tests for ensemble vs base models...")
    mcnemar_cat = StatisticalTester.mcnemar_paired_test(y_test, preds_ens, preds_cat)
    mcnemar_xgb = StatisticalTester.mcnemar_paired_test(y_test, preds_ens, preds_xgb)
    mcnemar_lgb = StatisticalTester.mcnemar_paired_test(y_test, preds_ens, preds_lgb)

    significance_tests = {
        "ensemble_vs_catboost": mcnemar_cat,
        "ensemble_vs_xgboost": mcnemar_xgb,
        "ensemble_vs_lightgbm": mcnemar_lgb
    }
    benchmark_results["significance_tests"] = significance_tests

    # 6. Robustness Perturbation Evaluation
    if args.robustness:
        logger.info("Running robustness and noise perturbation evaluations...")
        noise_results = RobustnessEvaluator.evaluate_noise_perturbation(
            ensemble, X_test, y_test, noise_levels=[0.01, 0.05, 0.10]
        )
        dropout_results = RobustnessEvaluator.evaluate_feature_dropout(
            ensemble, X_test, y_test, dropout_rates=[0.05, 0.10, 0.20]
        )
        benchmark_results["robustness"] = {
            "gaussian_noise": noise_results,
            "feature_dropout": dropout_results
        }

    # 7. Cross-Dataset Generalization Benchmark (Experiments A, B, C, D, E, F)
    if args.cross_dataset:
        logger.info("Executing Cross-Dataset Generalization Suite across Edge-IIoTset & CICIoT2023...")
        loader = DataLoader(root)
        df_edge_raw = loader.load_edge_iiot(mode="dev", sample_size=15000)
        df_cic_raw = loader.load_cic_iot_2023(split="validation", mode="dev", sample_size=15000)

        # Split Edge and CIC into train and test partitions
        from sklearn.model_selection import train_test_split
        edge_tr, edge_te = train_test_split(df_edge_raw, test_size=0.3, random_state=42)
        cic_tr, cic_te = train_test_split(df_cic_raw, test_size=0.3, random_state=42)

        def train_lgb_proxy(X_tr, y_tr, X_va, y_va):
            m = LightGBMModel(n_estimators=100, learning_rate=0.1, max_depth=5, n_jobs=-1, verbose=-1)
            m.fit(X_tr, y_tr, X_va, y_va)
            return m

        cd_bench = CrossDatasetBenchmark()
        cross_dataset_results = cd_bench.run_experiments(edge_tr, edge_te, cic_tr, cic_te, train_lgb_proxy)
        benchmark_results["cross_dataset_generalization"] = cross_dataset_results

    # Save comprehensive machine-readable metrics
    with open(metrics_dir / "locked_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    # 8. Generate reports/benchmark_report.md
    logger.info("Generating reports/benchmark_report.md...")
    with open(reports_dir / "benchmark_report.md", "w", encoding="utf-8") as f:
        f.write("# Comprehensive Intrusion Detection Benchmark Report\n\n")
        f.write(f"**Evaluation Date:** {pd.Timestamp.now().isoformat()}\n")
        f.write(f"**Locked Test Samples:** {len(X_test):,}\n")
        f.write(f"**Feature Count:** {len(feature_names)}\n")
        f.write(f"**Target Classes:** {len(class_names)}\n\n")

        f.write("## 1. Locked Test Classifier Performance Comparison\n\n")
        f.write("| Model Architecture | Accuracy | Balanced Acc | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Macro FPR | Macro FNR | 95% CI (Macro F1) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")

        for m_name, display_title in [
            ("catboost", "CatBoost Classifier"),
            ("xgboost", "XGBoost Classifier"),
            ("lightgbm", "LightGBM Classifier"),
            ("hybrid_ensemble", "**Validation Hybrid Ensemble**")
        ]:
            res = benchmark_results[m_name]
            ci = res.get("confidence_intervals", {}).get("macro_f1_ci", (0.0, 0.0))
            f.write(f"| {display_title} | {res['accuracy'] * 100:.2f}% | {res['balanced_accuracy'] * 100:.2f}% | "
                    f"{res['macro_precision'] * 100:.2f}% | {res['macro_recall'] * 100:.2f}% | "
                    f"**{res['macro_f1'] * 100:.2f}%** | {res['weighted_f1'] * 100:.2f}% | "
                    f"{res['macro_fpr'] * 100:.2f}% | {res['macro_fnr'] * 100:.2f}% | "
                    f"[{ci[0] * 100:.2f}%, {ci[1] * 100:.2f}%] |\n")

        f.write("\n## 2. Statistical Significance Analysis (McNemar's Paired Test)\n\n")
        f.write("| Comparison | Chi-Square Stat | p-value | Significant (p < 0.05)? |\n| :--- | :--- | :--- | :--- |\n")
        for comp_name, test_res in significance_tests.items():
            f.write(f"| {comp_name.replace('_', ' ').title()} | {test_res['statistic']} | {test_res['p_value']} | {'YES (p < 0.05)' if test_res['is_significant_at_05'] else 'NO'} |\n")

        if "cross_dataset_generalization" in benchmark_results:
            cd = benchmark_results["cross_dataset_generalization"]
            f.write("\n## 3. Cross-Dataset Generalization Results (Edge-IIoTset <-> CICIoT2023)\n\n")
            f.write("| Experiment Code & Direction | Macro F1 | Accuracy | Balanced Accuracy |\n| :--- | :--- | :--- | :--- |\n")
            f.write(f"| **Exp A:** Train Edge-IIoTset -> Test Edge-IIoTset | **{cd['experiment_a_edge_to_edge']['macro_f1'] * 100:.2f}%** | {cd['experiment_a_edge_to_edge']['accuracy'] * 100:.2f}% | {cd['experiment_a_edge_to_edge']['balanced_accuracy'] * 100:.2f}% |\n")
            f.write(f"| **Exp B:** Train CICIoT2023 -> Test CICIoT2023 | **{cd['experiment_b_cic_to_cic']['macro_f1'] * 100:.2f}%** | {cd['experiment_b_cic_to_cic']['accuracy'] * 100:.2f}% | {cd['experiment_b_cic_to_cic']['balanced_accuracy'] * 100:.2f}% |\n")
            f.write(f"| **Exp C:** Pooled Train -> Pooled Test | **{cd['experiment_c_pooled_to_pooled']['macro_f1'] * 100:.2f}%** | {cd['experiment_c_pooled_to_pooled']['accuracy'] * 100:.2f}% | {cd['experiment_c_pooled_to_pooled']['balanced_accuracy'] * 100:.2f}% |\n")
            f.write(f"| **Exp D:** Train Edge-IIoTset -> Test CICIoT2023 | **{cd['experiment_d_edge_to_cic']['macro_f1'] * 100:.2f}%** | {cd['experiment_d_edge_to_cic']['accuracy'] * 100:.2f}% | {cd['experiment_d_edge_to_cic']['balanced_accuracy'] * 100:.2f}% |\n")
            f.write(f"| **Exp E:** Train CICIoT2023 -> Test Edge-IIoTset | **{cd['experiment_e_cic_to_edge']['macro_f1'] * 100:.2f}%** | {cd['experiment_e_cic_to_edge']['accuracy'] * 100:.2f}% | {cd['experiment_e_cic_to_edge']['balanced_accuracy'] * 100:.2f}% |\n\n")

            comp_f = cd["experiment_f_comparison"]
            f.write(f"### Domain Shift Analysis (Exp F):\n")
            f.write(f"- **Edge -> CIC Transfer Drop:** {comp_f['domain_shift_delta_edge_to_cic'] * 100:.2f}% F1 drop due to sensor network distribution shift.\n")
            f.write(f"- **CIC -> Edge Transfer Drop:** {comp_f['domain_shift_delta_cic_to_edge'] * 100:.2f}% F1 drop due to domain shift.\n")

    logger.info(f"Evaluation complete. Reports and metrics written to {reports_dir / 'benchmark_report.md'}")


if __name__ == "__main__":
    main()
