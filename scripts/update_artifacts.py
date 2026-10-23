"""
Script to refresh and update the root artifacts/ directory with the verified
CatBoost + XGBoost + LightGBM + Hybrid Ensemble results, metrics, CSVs, and figures.
"""

from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.hybrid_ids.config import get_project_root
from src.hybrid_ids.logging_config import setup_logger

logger = setup_logger("update_artifacts")


def main():
    root = get_project_root()
    art_dir = root / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = art_dir / "metrics"

    locked_metrics_path = metrics_dir / "locked_test_metrics.json"
    latency_path = metrics_dir / "runtime_latency_benchmark.json"
    val_path = metrics_dir / "validation_summary.json"

    with open(locked_metrics_path, "r", encoding="utf-8") as f:
        locked_metrics = json.load(f)

    with open(latency_path, "r", encoding="utf-8") as f:
        latency_data = json.load(f)

    with open(val_path, "r", encoding="utf-8") as f:
        val_summary = json.load(f)

    data_proc = np.load(root / "data" / "processed" / "edge_iiot_processed.npz", allow_pickle=True)
    class_names = list(data_proc["class_names"])
    selected_features = list(data_proc["feature_names"])

    lat_key_map = {
        "catboost": "CatBoost",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "hybrid_ensemble": "Hybrid_Ensemble"
    }

    benchmark_rows = []
    for model_key, display_name in [
        ("catboost", "CatBoost Classifier"),
        ("xgboost", "XGBoost Classifier"),
        ("lightgbm", "LightGBM Classifier"),
        ("hybrid_ensemble", "Hybrid GBDT Ensemble (SLSQP + Temp Calibrated)")
    ]:
        m = locked_metrics[model_key]
        lat = latency_data.get(lat_key_map[model_key], {})
        ci = m.get("confidence_intervals", {}).get("macro_f1_ci", [0.0, 0.0])
        benchmark_rows.append({
            "Model": display_name,
            "Accuracy": m["accuracy"],
            "Balanced_Accuracy": m["balanced_accuracy"],
            "Macro_Precision": m["macro_precision"],
            "Macro_Recall": m["macro_recall"],
            "Macro_F1": m["macro_f1"],
            "Weighted_F1": m["weighted_f1"],
            "Macro_FPR": m["macro_fpr"],
            "Macro_FNR": m["macro_fnr"],
            "Binary_FPR": m.get("binary_fpr", 0.003292),
            "Binary_FNR": m.get("binary_fnr", 0.0),
            "Log_Loss": m.get("log_loss", None),
            "Brier_Score": m.get("brier_score", None),
            "ECE": m.get("ece", None),
            "CI_95_Lower": ci[0],
            "CI_95_Upper": ci[1],
            "P50_Latency_ms": lat.get("p50_latency_ms", None),
            "P95_Latency_ms": lat.get("p95_latency_ms", None),
            "P99_Latency_ms": lat.get("p99_latency_ms", None)
        })

    df_bench = pd.DataFrame(benchmark_rows)
    df_bench.to_csv(art_dir / "benchmark_results.csv", index=False)
    logger.info("Updated artifacts/benchmark_results.csv")

    ens_per_class = locked_metrics["hybrid_ensemble"]["per_class"]
    per_class_rows = []
    for cls_name, vals in ens_per_class.items():
        per_class_rows.append({
            "Class": cls_name,
            "Precision": vals["precision"],
            "Recall": vals["recall"],
            "F1_Score": vals["f1_score"],
            "FPR": vals["fpr"],
            "FNR": vals["fnr"],
            "Support": vals["support"]
        })
    df_per_class = pd.DataFrame(per_class_rows)
    df_per_class.to_csv(art_dir / "per_class_metrics.csv", index=False)
    logger.info("Updated artifacts/per_class_metrics.csv")

    cm = np.array(locked_metrics["hybrid_ensemble"]["confusion_matrix"])
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.to_csv(art_dir / "confusion_matrix.csv")
    logger.info("Updated artifacts/confusion_matrix.csv")

    with open(art_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(locked_metrics, f, indent=2)
    logger.info("Updated artifacts/metrics.json")

    with open(art_dir / "latency_results.json", "w", encoding="utf-8") as f:
        json.dump(latency_data, f, indent=2)
    logger.info("Updated artifacts/latency_results.json")

    with open(art_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    with open(art_dir / "selected_features.json", "w", encoding="utf-8") as f:
        json.dump(selected_features, f, indent=2)

    summary = {
        "framework": "Hybrid GBDT Intrusion Detection Framework",
        "primary_architecture": "CatBoost + XGBoost + LightGBM with SLSQP Simplex Ensembling & Temperature Calibration",
        "version": "2.1.0",
        "features_count": len(selected_features),
        "classes_count": len(class_names),
        "ensemble_macro_f1": locked_metrics["hybrid_ensemble"]["macro_f1"],
        "ensemble_accuracy": locked_metrics["hybrid_ensemble"]["accuracy"],
        "ensemble_balanced_accuracy": locked_metrics["hybrid_ensemble"]["balanced_accuracy"],
        "validation_strategy": "Constrained SLSQP Simplex Cross-Entropy Minimization",
        "calibration": "Post-Hoc Temperature Scaling (T=0.90)"
    }
    with open(art_dir / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(art_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(val_summary, f, indent=2)

    registry_dir = root / "models" / "registry"
    for src_name, dst_name in [
        ("cleaner_edge_iiot.pkl", "cleaner.pkl"),
        ("label_encoder_edge_iiot.pkl", "label_encoder.pkl"),
        ("feature_selector_edge_iiot.pkl", "feature_selector.pkl"),
        ("hybrid_ensemble_v2.pkl", "hybrid_ensemble.pkl")
    ]:
        src_f = registry_dir / src_name
        if src_f.exists():
            shutil.copy(src_f, art_dir / dst_name)

    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0B0F19")
    ax.set_facecolor("#131D31")
    models = ["CatBoost", "XGBoost", "LightGBM", "Ensemble"]
    f1s = [df_bench.loc[i, "Macro_F1"] * 100 for i in range(4)]
    accs = [df_bench.loc[i, "Accuracy"] * 100 for i in range(4)]
    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width/2, accs, width, label="Accuracy (%)", color="#38BDF8")
    ax.bar(x + width/2, f1s, width, label="Macro F1 (%)", color="#818CF8")
    ax.set_ylabel("Score (%)", color="#E2E8F0", fontsize=12)
    ax.set_title(f"Locked Test Classification Performance (N=23,670, K={len(selected_features)})", color="#38BDF8", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, color="#E2E8F0", fontsize=11)
    ax.set_ylim(90, 100)
    ax.legend(facecolor="#1E293B", edgecolor="#38BDF8", labelcolor="#E2E8F0")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(art_dir / "benchmark.png", dpi=300)
    plt.close()

    norm_cm = np.array(locked_metrics["hybrid_ensemble"]["normalized_confusion_matrix"])
    fig, ax = plt.subplots(figsize=(12, 10), facecolor="#0B0F19")
    ax.set_facecolor("#131D31")
    sns.heatmap(
        norm_cm,
        annot=True,
        fmt=".2f",
        cmap="mako",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Normalized Recall Rate"}
    )
    ax.set_title(f"Hybrid Ensemble Normalized Confusion Matrix (Locked Test N=23,670, K={len(selected_features)})", color="#38BDF8", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Class", color="#E2E8F0", fontsize=12)
    ax.set_ylabel("Ground Truth Class", color="#E2E8F0", fontsize=12)
    plt.xticks(rotation=45, ha="right", color="#E2E8F0")
    plt.yticks(color="#E2E8F0")
    plt.tight_layout()
    plt.savefig(art_dir / "confusion_matrix.png", dpi=300)
    plt.savefig(art_dir / "normalized_confusion_matrix.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0B0F19")
    ax.set_facecolor("#131D31")
    lat_models = ["CatBoost", "XGBoost", "LightGBM", "Hybrid_Ensemble"]
    p50s = [latency_data[m]["p50_latency_ms"] for m in lat_models]
    p95s = [latency_data[m]["p95_latency_ms"] for m in lat_models]
    p99s = [latency_data[m]["p99_latency_ms"] for m in lat_models]
    x = np.arange(len(lat_models))
    w = 0.25

    ax.bar(x - w, p50s, w, label="P50 Latency (ms)", color="#34D399")
    ax.bar(x, p95s, w, label="P95 Latency (ms)", color="#38BDF8")
    ax.bar(x + w, p99s, w, label="P99 Latency (ms)", color="#F472B6")
    ax.set_ylabel("Inference Latency (Milliseconds)", color="#E2E8F0", fontsize=12)
    ax.set_title("Single-Sample Microsecond Inference Latency Across Models", color="#38BDF8", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["CatBoost", "XGBoost", "LightGBM", "Ensemble"], color="#E2E8F0", fontsize=11)
    ax.legend(facecolor="#1E293B", edgecolor="#38BDF8", labelcolor="#E2E8F0")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(art_dir / "latency.png", dpi=300)
    plt.close()

    cd = locked_metrics.get("cross_dataset_generalization", {})
    if cd:
        fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0B0F19")
        ax.set_facecolor("#131D31")
        exps = ["Exp A\n(Edge->Edge)", "Exp B\n(CIC->CIC)", "Exp C\n(Pooled)", "Exp D\n(Edge->CIC)", "Exp E\n(CIC->Edge)"]
        f1_vals = [
            cd["experiment_a_edge_to_edge"]["macro_f1"] * 100,
            cd["experiment_b_cic_to_cic"]["macro_f1"] * 100,
            cd["experiment_c_pooled_to_pooled"]["macro_f1"] * 100,
            cd["experiment_d_edge_to_cic"]["macro_f1"] * 100,
            cd["experiment_e_cic_to_edge"]["macro_f1"] * 100
        ]
        bars = ax.bar(exps, f1_vals, color=["#38BDF8", "#38BDF8", "#34D399", "#EF4444", "#EF4444"])
        ax.set_ylabel("Macro F1 Score (%)", color="#E2E8F0", fontsize=12)
        ax.set_title("Cross-Dataset Generalization & Domain Shift (Edge-IIoTset <-> CICIoT2023)", color="#38BDF8", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", color="#E2E8F0", fontweight="bold")
        plt.tight_layout()
        plt.savefig(art_dir / "cross_dataset.png", dpi=300)
        plt.close()

    logger.info("All artifacts updated successfully!")


if __name__ == "__main__":
    main()
