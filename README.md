# Hybrid Edge-IIoTset + CICIoT2023 Intrusion Detection Framework
### *CatBoost + XGBoost + LightGBM with Constrained SLSQP Simplex Ensembling & Post-Hoc Temperature Calibration*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2%2B-yellow.svg)](https://catboost.ai/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-green.svg)](https://lightgbm.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 1. Executive Summary & Core Architecture

This repository implements a forensically audited, zero-leakage, reproducible hybrid intrusion detection system (IDS) engineered for heterogeneous Internet of Things (IoT) and Industrial IoT (IIoT) environments.

The framework combines three distinct gradient-boosted decision tree (GBDT) classifiers:
- **CatBoost:** Symmetric oblivious decision trees with branchless $O(D)$ evaluation (`0.073 ms` single-event P50 latency; `1,000,584 events/s` batched throughput).
- **XGBoost:** Depth-wise second-order gradient histogram boosting with sparsity-aware splitting and L1/L2 tree regularization.
- **LightGBM:** Leaf-wise histogram tree growth with Gradient-Based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB).

The three models are unified through a **Constrained SLSQP Validation-Weighted Simplex Ensemble** ($\mathbf{w}^* = [0.0000, 0.4610, 0.5390]^T$) followed by **Post-Hoc Temperature Scaling** ($T = 0.90$) fitted strictly on the held-out validation split ($N_{\text{val}} = 23,670$).

```
Hybrid Telemetry Ingestion (Edge-IIoTset + CICIoT2023)
  │
  ▼
Schema & Canonical Label Harmonization (15 Native IIoT Classes / 10 Shared Cross-Domain Classes)
  │
  ▼
Strict Zero-Leakage 70/15/15 Stratified Partitioning (Locked Test Set N=23,670)
  │
  ▼
Training-Fitted Sanitization (6 Identifiers Purged) & Protocol Feature Selection (K=33 Features)
  │
  ├────────────────────────┬────────────────────────┐
  ▼                        ▼                        ▼
CatBoost Model           XGBoost Model            LightGBM Model
  │                        │                        │
  └────────────────────────┼────────────────────────┘
                           ▼
        Constrained SLSQP Validation-Weighted Ensemble
         (Simplex Blending + Temperature Scaling T=0.90)
                           │
                           ▼
     Real-Time Threat Detection & Micro-Batched Redpanda Alerting
```

---

## 2. Verified Empirical Benchmark Results (Locked Test Partition: $N = 23,670$)

All metrics are computed directly from `y_true` and `y_pred` on the strictly locked test partition ($N = 23,670$ records) with zero data or target leakage. Both the **Frozen Baseline ($K=25$)** and **Optimized Full Protocol Feature System ($K=33$)** are preserved in `models/registry/`:

| Model Architecture | Feature Set | Accuracy | Balanced Acc | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | OvR Macro FPR | Binary Benign FPR | Log-Loss | ECE | P50 Latency |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **CatBoost (Baseline)** | $K=25$ | 98.93% | 94.19% | 98.87% | 94.19% | 95.47% | 98.82% | 0.08% | 0.33% | 0.0462 | 1.384% | **0.075 ms** |
| **XGBoost (Baseline)** | $K=25$ | 98.93% | 94.19% | 98.97% | 94.19% | 95.51% | 98.82% | 0.08% | 0.33% | 0.0289 | 0.105% | 0.264 ms |
| **LightGBM (Baseline)** | $K=25$ | 98.96% | 94.22% | 99.00% | 94.22% | 95.54% | 98.85% | 0.07% | 0.33% | 0.0283 | 0.106% | 0.205 ms |
| **Hybrid Ensemble (Baseline)** | $K=25$ | 98.94% | 94.20% | 98.98% | 94.20% | 95.53% | 98.84% | 0.08% | 0.33% | 0.0286 | 0.130% | 1.337 ms |
| **CatBoost (Optimized)** | **$K=33$** | 99.18% | 98.18% | 97.96% | 98.18% | **97.67%** | 99.18% | 0.06% | 0.33% | 0.0277 | 0.700% | **0.073 ms** |
| **XGBoost (Optimized)** | **$K=33$** | 99.18% | 98.19% | 98.00% | 98.19% | **97.70%** | 99.18% | 0.06% | 0.33% | 0.0222 | 0.082% | 0.235 ms |
| **LightGBM (Optimized)** | **$K=33$** | 99.20% | 98.21% | 98.01% | 98.21% | **97.72%** | 99.20% | 0.06% | 0.33% | **0.0215** | **0.068%** | 0.187 ms |
| **Calibrated SLSQP Ensemble** | **$K=33$** | **99.21%** | **98.21%** | **98.03%** | **98.21%** | **97.73%** | **99.20%** | **0.06%** | **0.33%** | **0.0215** | **0.073%** | **1.316 ms** |

### Key Scientific Highlights:
- **Resolving `MITM` Collisions ($K=25 \to K=33$):** Retaining all 33 non-constant zero-leakage protocol features preserves `arp.opcode`, `arp.hw.size`, `dns.qry.qu`, and `dns.qry.type`, reducing irreducible test-set feature collisions from 532 rows (Bayes ceiling $99.17\%$) to 348 rows (Bayes ceiling $99.43\%$), lifting `MITM` recall from **$36.26\%$ to $100.00\%$** and overall Ensemble Macro-$F_1$ from **$95.53\%$ to $97.73\%$** ($\chi^2_{\text{McNemar}} = 18.42, p = 1.77 \times 10^{-5}$; paired bootstrap Wilcoxon $p = 8.41 \times 10^{-32}$).
- **Within-Fixed-$K$ Convergence to Bayes Ceiling:** Within $K=33$, all three GBDTs and the ensemble operate within $0.22\%$ of the empirical Bayes upper bound ($99.4254\%$), with pairwise McNemar $p \ge 0.099$. The calibrated ensemble provides a $10\times$ reduction in Expected Calibration Error over CatBoost ($0.700\% \to 0.073\%$).

---

## 3. Cross-Dataset Generalization Findings (10 Shared Classes, 12 Features)

Evaluated across **Edge-IIoTset** (Industrial IoT) and **CICIoT2023** (Consumer IoT) using stratified 70/15/15 splits ($N_{\text{test}} = 3,750$ per domain):

| Experiment Code | Protocol & Transfer Direction | Ensemble Macro F1 | Single LightGBM Macro F1 | Ensemble Accuracy | Ensemble Balanced Acc |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **Experiment A** | Train Edge-IIoTset $\to$ Test Edge-IIoTset (In-Domain) | **59.05%** | 59.45% | 62.24% | 59.09% |
| **Experiment B** | Train CICIoT2023 $\to$ Test CICIoT2023 (In-Domain) | **66.16%** | 67.82% | 98.48% | 65.87% |
| **Experiment C** | Natural Pooled Train $\to$ Pooled Test ($N=7,500$) | **66.10%** | 66.77% | 79.96% | 63.76% |
| **Experiment C-DB1** | **Domain-Balanced Pooled $\to$ Test Edge-IIoTset** | **55.66%** | — | 52.88% | **59.54%** |
| **Experiment C-DB2** | **Domain-Balanced Pooled $\to$ Test CICIoT2023** | **70.32%** | — | 96.45% | **78.89%** |
| **Experiment D** | Zero-Shot Train Edge-IIoTset $\to$ Test CICIoT2023 | **27.66%** | 20.42% | 42.11% | 34.43% |
| **Experiment E** | Zero-Shot Train CICIoT2023 $\to$ Test Edge-IIoTset | **9.34%** (12.70% bal.) | 14.43% | 12.19% | 15.81% |

---

## 4. Key Reports & IEEE Manuscript Artifacts (in `reports/`)

- `reports/FORENSIC_RECONCILIATION_AND_AUDIT_REPORT.md` — Complete Phase 1–3 forensic audit resolving all historical discrepancies.
- `reports/forensic_evidence_registry.csv` — Cryptographic SHA-256 registry of datasets, splits, checkpoints, and experiments (`E1`–`E12`).
- `reports/IEEE_IoT_IDS_Manuscript.pdf` & `reports/IEEE_IoT_IDS_Manuscript.tex` — Compiled 10-page double-column IEEE journal manuscript with 5 vector figures (`artifacts/figures/fig1..fig5.pdf`).
- `reports/SCOPUS_REFERENCE_AUDIT_TABLE.md` — 30 verified Scopus-indexed Q1/Q2 peer-reviewed journal references (zero conference or arXiv citations).

---

## 5. Quick-Start & Execution Commands

```bash
# 1. Prepare Zero-Leakage 70/15/15 Split with All K=33 Non-Constant Protocol Features
python3 scripts/prepare_data.py --mode full --k-features 33

# 2. Train CatBoost, XGBoost, LightGBM, and Calibrated SLSQP Ensemble
python3 scripts/train_models.py

# 3. Run Locked-Test & Cross-Dataset Evaluation Suite
python3 scripts/evaluate_models.py
python3 scripts/run_ablations_and_statistics.py
python3 scripts/update_artifacts.py

# 4. Run Automated Pytest Verification Suite
pytest tests/ -v

# 5. Launch Interactive SOC Streamlit Dashboard
streamlit run app.py
```

---

## 6. Cryptographic Dataset Manifests (SHA-256)

| Dataset File | File Size (Bytes) | SHA-256 Digest |
| :--- | :--- | :--- |
| `Edge-IIoTset dataset.zip` | 1,741,481,351 | `4a4a16542756d72390b60c7cde5c46a8439ac5c6e83502e6fc8c64d2d7669d72` |
| `archive.zip` (CICIoT2023) | 500,559,747 | `793b212d1dc40973068d06cec4d0676eaf7f17436a859dfa522303e199c7ea5a` |
| `ML-EdgeIIoT-dataset.csv` | 82,184,390 | `53101fad091af20bee815860962a5016b802ee45d456a141042f6183094c3c1a` |
| `CICIOT23 test.csv` | 347,958,926 | `15aeb1f47a7bd3e3fc394d18d7b196b3640329abaae65d5220df4def6119bfc6` |
| `CICIOT23 validation.csv` | 347,921,865 | `b13775e87ef64c8dd66c910d4585323f99b1209f2508c75a0a0a8d584caf20d8` |

---

## 7. License
This framework is released under the **MIT License**.
