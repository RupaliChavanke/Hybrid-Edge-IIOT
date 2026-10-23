# Comprehensive Intrusion Detection Benchmark Report (Reconciled K=33 & K=25)

**Evaluation Date:** 2026-10-23T23:52:51.334252
**Locked Test Samples:** 23,670
**Feature Count:** 33 zero-leakage protocol features (K=25 baseline also preserved)
**Target Classes:** 15

## 1. Locked Test Classifier Performance Comparison (K=33 Zero-Leakage Protocol Features)

| Model Architecture | Accuracy | Balanced Acc | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | OvR Macro FPR | Binary Benign FPR | Log-Loss | ECE | 95% CI (Macro F1) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| CatBoost Classifier (K=33) | 99.18% | 98.19% | 97.54% | 98.19% | **97.66%** | 99.20% | 0.06% | 0.33% | 0.0279 | 0.705% | [97.24%, 98.10%] |
| XGBoost Classifier (K=33) | 99.18% | 98.18% | 97.63% | 98.18% | **97.70%** | 99.19% | 0.06% | 0.33% | 0.0222 | 0.089% | [97.28%, 98.14%] |
| LightGBM Classifier (K=33) | 99.19% | 98.20% | 97.64% | 98.20% | **97.71%** | 99.21% | 0.06% | 0.33% | 0.0215 | 0.059% | [97.30%, 98.15%] |
| **Calibrated SLSQP Hybrid Ensemble (K=33)** | 99.20% | 98.20% | 97.65% | 98.20% | **97.72%** | 99.21% | 0.06% | 0.33% | 0.0215 | 0.067% | [97.31%, 98.16%] |

## 2. Statistical Significance Analysis (McNemar's Paired Test on K=33)

| Comparison | Chi-Square Stat | p-value | Significant (p < 0.05)? |
| :--- | :--- | :--- | :--- |
| Ensemble Vs Catboost | 1.125 | 0.288844 | NO |
| Ensemble Vs Xgboost | 1.7778 | 0.182422 | NO |
| Ensemble Vs Lightgbm | 0.5 | 0.4795 | NO |

## 3. Cross-Dataset Generalization Results (Edge-IIoTset <-> CICIoT2023, 10 Common Classes, 12 Features)

| Experiment Code & Direction | Ensemble Macro F1 | Accuracy | Balanced Accuracy |
| :--- | ---: | ---: | ---: |
| **Exp A:** Train Edge-IIoTset -> Test Edge-IIoTset | **59.05%** | 62.85% | 56.37% |
| **Exp B:** Train CICIoT2023 -> Test CICIoT2023 | **66.16%** | 98.53% | 63.78% |
| **Exp C:** Natural Pooled Train -> Pooled Test | **66.10%** | 80.29% | 64.54% |
| **Exp C1:** Natural Pooled -> Test Edge-IIoTset | **57.28%** | 61.71% | 55.07% |
| **Exp C2:** Natural Pooled -> Test CICIoT2023 | **69.85%** | 98.88% | 70.03% |
| **Exp C-DB2:** Domain-Balanced Pooled -> Test CICIoT2023 | **70.32%** | 98.77% | 71.80% |
| **Exp D:** Zero-Shot Train Edge-IIoTset -> Test CICIoT2023 | **27.66%** | 56.69% | 28.77% |
| **Exp E:** Zero-Shot Train CICIoT2023 -> Test Edge-IIoTset | **9.34%** | 13.60% | 13.31% |
