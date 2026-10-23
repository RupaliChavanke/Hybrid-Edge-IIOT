# Deliverable 4: Claim-to-Evidence Matrix

**Manuscript Title:** *Empirical Feature-Collision Analysis, Leakage-Safe Optimization, and Calibrated Boosting Ensembles for IoT/IIoT Intrusion Detection*  
**Audited Artifacts:** `experiments/verified_experiment_matrix_E1_to_E12.json`, `artifacts/metrics/locked_test_metrics.json`, `reports/forensic_evidence_registry.csv`, `split_audit.csv`.  
**Date:** October 24, 2026  

---

## 1. Traceability Matrix

| Claim ID | Manuscript Location | Reported Value | Evidence File | Reproduction Status | Discrepancy | Required Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C1** | Sec. III-A, IV-B, Tab. II | Edge-IIoTset 70/15/15: Train=110,460, Val=23,670, Test=23,670 (Total=157,800) | `data/processed/edge_iiot_processed.npz`, `split_audit.csv` | **VERIFIED (Exact)** | None | Maintain locked split numbers. |
| **C2** | Sec. IV-B, Tab. III | Raw dataset duplicates: 814 (0.52%) in Edge-IIoTset; 36 (0.04%) in CIC-IoT sample | `reports/audit/duplicate_candidates.csv` | **VERIFIED (Exact)** | None | Document in data quality table. |
| **C3** | Sec. III-B, Tab. III | Retained non-collinear protocol features: $K=33$ ($\rho \le 0.95$) | `artifacts/selected_features.json` | **VERIFIED (Exact)** | None | Confirm zero timestamp leakage. |
| **C4** | Abstract, Sec. III-B, Tab. III | $K=25$ Collision Bound: 532 colliding records, 196 errors, ceiling $=99.1719\%$ | `exp_part4_optimized_system_and_stats.py` | **VERIFIED (Exact)** | None | Clarify finite-sample empirical bound. |
| **C5** | Abstract, Sec. V-C, Tab. IV | $K=25$ MITM Recall Collapse: $36.26\%$ (66 / 182 detected) | `artifacts/metrics/locked_test_metrics.json` | **VERIFIED (Exact)** | None | Document Layer-2 MI truncation mechanism. |
| **C6** | Abstract, Sec. III-B, Tab. III | $K=33$ Collision Bound: 348 colliding records, 136 errors, ceiling $=99.4254\%$ | `exp_part4_optimized_system_and_stats.py` | **VERIFIED (Exact)** | None | Highlight Layer-2 ARP resolution. |
| **C7** | Abstract, Sec. V-C, Tab. IV | $K=33$ MITM Recall: $100.00\%$ (182 / 182 detected, $F_1 = 83.87\%$) | `artifacts/metrics/locked_test_metrics.json` | **VERIFIED (Exact)** | None | Highlight $+30.64\text{ pp}$ $F_1$ recovery. |
| **C8** | Abstract, Sec. V-A, Tab. IV | Locked Test Macro-$F_1$ Gain: $95.53\% \to 97.73\%$ ($+2.20\text{ pp}$) | `locked_test_metrics.json`, `verified_matrix` | **VERIFIED (Exact)** | None | Reconcile with bootstrap CI. |
| **C9** | Sec. V-E, Tab. V | Paired McNemar Test ($K=33$ vs. $K=25$): $\chi^2 = 20.34, p = 6.49 \times 10^{-6}$ | `statistical_comparisons.csv` | **VERIFIED (Exact)** | None | Include Edwards continuity correction. |
| **C10** | Sec. V-E, Tab. V | Ensemble vs. LightGBM Equivalence: $p = 1.0000$, $\Delta\text{Macro-}F_1 = +0.004\text{ pp}$ | `statistical_comparisons.csv` | **VERIFIED (Exact)** | None | Disclose that ensemble equals LightGBM point acc. |
| **C11** | Abstract, Sec. V-D, Fig. 4 | Probability Calibration: Brier $=0.01232$ (Isotonic), ECE $=0.073\%$ (Temperature) | `artifacts/calibration.json` | **VERIFIED (Exact)** | None | Clarify metric attribution across methods. |
| **C12** | Abstract, Sec. V-D, Tab. IV | False-Alarm Rates: OvR Macro $\text{FPR}=0.06\%$, Binary $\text{FPR}=0.33\%$, $\text{FNR}=0.00\%$ | `locked_test_metrics.json`, `split_audit.csv` | **VERIFIED (Exact)** | Previous script conflated Backdoor with Normal. | Reconciled via true class index $7=\text{Normal}$. |
| **C13** | Abstract, Sec. VI-B, Tab. VI | Cross-Dataset Zero-Shot: Edge$\to$CIC $=27.66\%$, CIC$\to$Edge $=12.70\%$ | `cross_dataset_results.csv` | **VERIFIED (Exact)** | Previous unshuffled slice gave 16.5% / 0.0%. | Document and fix sequential slice bug. |
| **C14** | Abstract, Sec. VI-C, Tab. VI | Multi-Source Domain-Balanced Transfer: CIC-IoT-2023 Macro-$F_1 = 70.32\%$ | `cross_dataset_results.csv` | **VERIFIED (Exact)** | None | Highlight $+4.16\text{ pp}$ gain over in-domain. |
| **C15** | Sec. VIII-A, Tab. VII | Standalone Model Latency: CatBoost P50 $=0.073\text{ ms}$, LightGBM P50 $=0.187\text{ ms}$ | `runtime_latency_benchmark.json` | **VERIFIED (Exact)** | None | Decouple isolated from streaming. |
| **C16** | Abstract, Sec. VIII-B, Tab. VII| End-to-End Streaming Latency: $11.842\text{ ms}$ P50, $79.6\%$ DataFrame overhead | `runtime_latency_benchmark.json` | **VERIFIED (Exact)** | None | Profile component-level bottlenecks. |
| **C17** | Abstract, Sec. VIII-C, Tab. VII| Micro-Batched Streaming: $B=500$, $0.048\text{ ms/event}$, $20,640\text{ events/s}$, $0\%$ drop | `runtime_latency_benchmark.json` | **VERIFIED (Exact)** | None | Distinguish benchmark from live cluster. |

---

## 2. Integrity and Traceability Summary
- **Total Empirical Claims Audited:** 17
- **Directly Verified from Source Code and Predictions:** 17 (100.0%)
- **Unverified or Fabricated Claims:** 0
- **Retracted Narrative Claims:** 1 (Retracted unverified MITM support table from obsolete `reports/final/model_comparison.md`, replaced with verified confusion matrix).
