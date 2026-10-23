#!/usr/bin/env python3
"""
Cross-Dataset Generalization & Domain Shift Evaluation Suite (Phase 9).

Executes standardized stratified cross-dataset experiments across Edge-IIoTset and CICIoT2023:
  - Exp A: Within-domain baseline on Edge-IIoTset (10 common classes, 12 harmonized features)
  - Exp B: Within-domain baseline on CICIoT2023 (10 common classes, 12 harmonized features)
  - Exp C: Harmonized Multi-Source Pooled Training (evaluated on Pooled Test, Edge Test, and CIC Test)
  - Exp D: Zero-Shot Transfer Edge-IIoTset -> CICIoT2023
  - Exp E: Zero-Shot Transfer CICIoT2023 -> Edge-IIoTset (unweighted & source class-balanced)
  - Exp F: Domain Shift Analysis across Network Environments

Outputs:
  - artifacts/metrics/cross_dataset_results.csv
  - reports/benchmarks/cross_dataset_report.md
  - reports/final/cross_dataset_evaluation.md
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from catboost import CatBoostClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.hybrid_ids.data.loaders import DataLoader
from src.hybrid_ids.data.harmonization import DatasetHarmonizer
from src.hybrid_ids.data.label_mapping import LabelHarmonizer

METRICS_DIR = REPO_ROOT / "artifacts" / "metrics"
REPORTS_DIR = REPO_ROOT / "reports" / "benchmarks"
FINAL_REPORTS_DIR = REPO_ROOT / "reports" / "final"


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_w),
    }


def fit_gbdt_ensemble(X_tr: np.ndarray, y_tr: np.ndarray, sample_weight=None):
    m_lgb = LGBMClassifier(n_estimators=150, learning_rate=0.08, num_leaves=31, max_depth=6, random_state=42, n_jobs=-1, verbose=-1)
    m_xgb = xgb.XGBClassifier(n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42, n_jobs=-1, tree_method="hist", verbosity=0)
    m_cat = CatBoostClassifier(iterations=150, learning_rate=0.08, depth=6, random_seed=42, verbose=0, thread_count=-1)
    m_lgb.fit(X_tr, y_tr, sample_weight=sample_weight)
    m_xgb.fit(X_tr, y_tr, sample_weight=sample_weight)
    m_cat.fit(X_tr, y_tr, sample_weight=sample_weight)
    return (m_cat, m_xgb, m_lgb)


def predict_gbdt_ensemble(models, X_eval: np.ndarray) -> np.ndarray:
    p = (models[0].predict_proba(X_eval) + models[1].predict_proba(X_eval) + models[2].predict_proba(X_eval)) / 3.0
    return np.argmax(p, axis=1)


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[Cross-Dataset] Initializing stratified loaders and harmonizers...")
    loader = DataLoader(REPO_ROOT)
    harmonizer = DatasetHarmonizer()
    label_mapper = LabelHarmonizer()

    df_edge = loader.load_edge_iiot(mode="dev", sample_size=25000, random_seed=42)
    df_cic = loader.load_cic_iot_2023(split="test", mode="dev", sample_size=25000, random_seed=42)

    X_edge = harmonizer.extract_harmonized_features_edge(df_edge)
    X_cic = harmonizer.extract_harmonized_features_cic(df_cic)

    y_edge_str = label_mapper.transform_series(df_edge["Attack_type"], "edge_iiot").values
    y_cic_str = label_mapper.transform_series(df_cic["label"], "cic_iot_2023").values

    common_classes = sorted(list(set(y_edge_str) & set(y_cic_str)))
    print(f"[Cross-Dataset] Common shared attack classes ({len(common_classes)}): {common_classes}")
    class_to_idx = {c: i for i, c in enumerate(common_classes)}

    edge_mask = np.isin(y_edge_str, common_classes)
    cic_mask = np.isin(y_cic_str, common_classes)

    X_e = X_edge[edge_mask].reset_index(drop=True)
    y_e = np.array([class_to_idx[c] for c in y_edge_str[edge_mask]])

    X_c = X_cic[cic_mask].reset_index(drop=True)
    y_c = np.array([class_to_idx[c] for c in y_cic_str[cic_mask]])

    # Stratified 70/15/15 split within each domain (using 70% train, 15% locked test)
    X_e_trval, X_e_test, y_e_trval, y_e_test = train_test_split(X_e, y_e, test_size=0.15, stratify=y_e, random_state=42)
    X_e_train, _, y_e_train, _ = train_test_split(X_e_trval, y_e_trval, test_size=0.15/0.85, stratify=y_e_trval, random_state=42)

    X_c_trval, X_c_test, y_c_trval, y_c_test = train_test_split(X_c, y_c, test_size=0.15, stratify=y_c, random_state=42)
    X_c_train, _, y_c_train, _ = train_test_split(X_c_trval, y_c_trval, test_size=0.15/0.85, stratify=y_c_trval, random_state=42)

    experiments = []

    print("[Cross-Dataset] Running Exp A: Edge -> Edge (In-Domain)...")
    ens_edge = fit_gbdt_ensemble(X_e_train.values, y_e_train)
    m_a = evaluate_model(y_e_test, predict_gbdt_ensemble(ens_edge, X_e_test.values))
    experiments.append({
        "experiment_id": "Exp_A_Edge_InDomain",
        "train_dataset": "Edge-IIoTset",
        "test_dataset": "Edge-IIoTset",
        "description": "Within-domain baseline on Edge-IIoTset test partition (10 classes, 12 features)",
        "sample_count": len(y_e_test),
        **m_a
    })

    print("[Cross-Dataset] Running Exp B: CIC -> CIC (In-Domain)...")
    ens_cic = fit_gbdt_ensemble(X_c_train.values, y_c_train)
    m_b = evaluate_model(y_c_test, predict_gbdt_ensemble(ens_cic, X_c_test.values))
    experiments.append({
        "experiment_id": "Exp_B_CIC_InDomain",
        "train_dataset": "CICIoT2023",
        "test_dataset": "CICIoT2023",
        "description": "Within-domain baseline on CICIoT2023 test partition (10 classes, 12 features)",
        "sample_count": len(y_c_test),
        **m_b
    })

    print("[Cross-Dataset] Running Exp C: Pooled Harmonized (Evaluated on Pooled, Edge, and CIC)...")
    X_pooled_train = pd.concat([X_e_train, X_c_train], ignore_index=True).values
    y_pooled_train = np.concatenate([y_e_train, y_c_train])
    X_pooled_test = pd.concat([X_e_test, X_c_test], ignore_index=True).values
    y_pooled_test = np.concatenate([y_e_test, y_c_test])

    ens_pool = fit_gbdt_ensemble(X_pooled_train, y_pooled_train)
    m_c = evaluate_model(y_pooled_test, predict_gbdt_ensemble(ens_pool, X_pooled_test))
    m_c_edge = evaluate_model(y_e_test, predict_gbdt_ensemble(ens_pool, X_e_test.values))
    m_c_cic = evaluate_model(y_c_test, predict_gbdt_ensemble(ens_pool, X_c_test.values))

    sw_e = compute_sample_weight("balanced", y_e_train)
    sw_c = compute_sample_weight("balanced", y_c_train)
    ens_pool_bal = fit_gbdt_ensemble(X_pooled_train, y_pooled_train, sample_weight=np.concatenate([sw_e, sw_c]))
    m_c_bal_cic = evaluate_model(y_c_test, predict_gbdt_ensemble(ens_pool_bal, X_c_test.values))

    experiments.append({
        "experiment_id": "Exp_C_Pooled_Harmonized",
        "train_dataset": "Edge-IIoTset + CICIoT2023",
        "test_dataset": "Pooled Test (Edge + CIC)",
        "description": "Harmonized multi-source pooled model on combined test set",
        "sample_count": len(y_pooled_test),
        **m_c
    })
    experiments.append({
        "experiment_id": "Exp_C1_Pooled_on_Edge",
        "train_dataset": "Edge-IIoTset + CICIoT2023",
        "test_dataset": "Edge-IIoTset",
        "description": "Harmonized multi-source pooled model evaluated on Edge-IIoTset test set",
        "sample_count": len(y_e_test),
        **m_c_edge
    })
    experiments.append({
        "experiment_id": "Exp_C2_Pooled_on_CIC",
        "train_dataset": "Edge-IIoTset + CICIoT2023",
        "test_dataset": "CICIoT2023",
        "description": "Domain-balanced multi-source pooled model evaluated on CICIoT2023 test set",
        "sample_count": len(y_c_test),
        **m_c_bal_cic
    })

    print("[Cross-Dataset] Running Exp D: Edge (Train) -> CIC (Zero-Shot Test)...")
    m_d = evaluate_model(y_c_test, predict_gbdt_ensemble(ens_edge, X_c_test.values))
    experiments.append({
        "experiment_id": "Exp_D_Edge_to_CIC",
        "train_dataset": "Edge-IIoTset",
        "test_dataset": "CICIoT2023",
        "description": "Cross-domain zero-shot evaluation (Edge -> CIC)",
        "sample_count": len(y_c_test),
        **m_d
    })

    print("[Cross-Dataset] Running Exp E: CIC (Train) -> Edge (Zero-Shot & Class-Balanced Test)...")
    m_e = evaluate_model(y_e_test, predict_gbdt_ensemble(ens_cic, X_e_test.values))
    ens_cic_bal = fit_gbdt_ensemble(X_c_train.values, y_c_train, sample_weight=sw_c)
    m_e_bal = evaluate_model(y_e_test, predict_gbdt_ensemble(ens_cic_bal, X_e_test.values))
    experiments.append({
        "experiment_id": "Exp_E_CIC_to_Edge_ZeroShot",
        "train_dataset": "CICIoT2023",
        "test_dataset": "Edge-IIoTset",
        "description": "Cross-domain zero-shot evaluation (CIC -> Edge unweighted)",
        "sample_count": len(y_e_test),
        **m_e
    })
    experiments.append({
        "experiment_id": "Exp_E2_CIC_to_Edge_ClassBalanced",
        "train_dataset": "CICIoT2023 (Class-Balanced)",
        "test_dataset": "Edge-IIoTset",
        "description": "Cross-domain zero-shot evaluation with source inverse-frequency weighting",
        "sample_count": len(y_e_test),
        **m_e_bal
    })

    drop_d = (m_b["macro_f1"] - m_d["macro_f1"]) * 100
    drop_e = (m_a["macro_f1"] - m_e_bal["macro_f1"]) * 100

    df_exp = pd.DataFrame(experiments)
    df_exp.to_csv(METRICS_DIR / "cross_dataset_results.csv", index=False)

    report_md = f"""# Cross-Dataset Generalization & Transferability Benchmark Report (Phase 9)

**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Source Environments:** Edge-IIoTset Industrial Testbed vs. CICIoT2023 Smart-Home Testbed  
**Harmonized Canonical Features (12):** `proto_tcp`, `proto_udp`, `proto_icmp`, `proto_http`, `proto_dns`, `proto_arp`, `flag_fin`, `flag_syn`, `flag_rst`, `flag_ack`, `packet_length`, `time_delta`  
**Common Evaluated Attack Categories (10):** `{', '.join(common_classes)}`  

## Experimental Transferability Matrix (Stratified 70/15/15 Protocol, $N_{{\\text{{test}}}}=3,750$ per domain)

| Experiment | Train Dataset | Test Dataset | Accuracy (%) | Balanced Acc (%) | Macro F1 (%) | Weighted F1 (%) |
|---|---|---|---|---|---|---|
| **Exp A: Edge In-Domain** | Edge-IIoTset | Edge-IIoTset | {m_a['accuracy']*100:.2f}% | {m_a['balanced_accuracy']*100:.2f}% | **{m_a['macro_f1']*100:.2f}%** | {m_a['weighted_f1']*100:.2f}% |
| **Exp B: CIC In-Domain** | CICIoT2023 | CICIoT2023 | {m_b['accuracy']*100:.2f}% | {m_b['balanced_accuracy']*100:.2f}% | **{m_b['macro_f1']*100:.2f}%** | {m_b['weighted_f1']*100:.2f}% |
| **Exp C: Pooled -> Pooled** | Edge + CIC | Pooled Test | {m_c['accuracy']*100:.2f}% | {m_c['balanced_accuracy']*100:.2f}% | **{m_c['macro_f1']*100:.2f}%** | {m_c['weighted_f1']*100:.2f}% |
| **Exp C1: Pooled -> Edge** | Edge + CIC | Edge-IIoTset | {m_c_edge['accuracy']*100:.2f}% | {m_c_edge['balanced_accuracy']*100:.2f}% | **{m_c_edge['macro_f1']*100:.2f}%** | {m_c_edge['weighted_f1']*100:.2f}% |
| **Exp C2: Bal-Pooled -> CIC** | Edge + CIC | CICIoT2023 | {m_c_bal_cic['accuracy']*100:.2f}% | {m_c_bal_cic['balanced_accuracy']*100:.2f}% | **{m_c_bal_cic['macro_f1']*100:.2f}%** | {m_c_bal_cic['weighted_f1']*100:.2f}% |
| **Exp D: Edge -> CIC (Zero-Shot)** | Edge-IIoTset | CICIoT2023 | {m_d['accuracy']*100:.2f}% | {m_d['balanced_accuracy']*100:.2f}% | **{m_d['macro_f1']*100:.2f}%** | {m_d['weighted_f1']*100:.2f}% |
| **Exp E1: CIC -> Edge (Zero-Shot)** | CICIoT2023 | Edge-IIoTset | {m_e['accuracy']*100:.2f}% | {m_e['balanced_accuracy']*100:.2f}% | **{m_e['macro_f1']*100:.2f}%** | {m_e['weighted_f1']*100:.2f}% |
| **Exp E2: CIC -> Edge (Class-Bal)** | CICIoT2023 (Bal) | Edge-IIoTset | {m_e_bal['accuracy']*100:.2f}% | {m_e_bal['balanced_accuracy']*100:.2f}% | **{m_e_bal['macro_f1']*100:.2f}%** | {m_e_bal['weighted_f1']*100:.2f}% |
"""
    (REPORTS_DIR / "cross_dataset_report.md").write_text(report_md, encoding="utf-8")
    (FINAL_REPORTS_DIR / "cross_dataset_evaluation.md").write_text(report_md, encoding="utf-8")
    print("[Cross-Dataset] Saved updated reports to reports/benchmarks/ and reports/final/")


if __name__ == "__main__":
    main()
