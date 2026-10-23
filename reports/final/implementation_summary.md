# Comprehensive Implementation Summary (Reconciled & Verified)

## Executive Summary
This project implements, audits, and benchmarks a leak-free, calibration-aware intrusion detection system (IDS) utilizing **CatBoost**, **XGBoost**, **LightGBM**, and a **Constrained SLSQP + Temperature-Calibrated Simplex Ensemble** across two major IoT/IIoT benchmarks: **Edge-IIoTset** ($N=157,800$) and **CICIoT2023** ($N=1,176,851$ test partition).

All historical numerical contradictions across the repository have been forensically audited, reconciled, and fixed in-place:
1. **Zero Data & Target Leakage:** Enforces a strict 70/15/15 stratified partition ($N_{\text{train}}=110,460$, $N_{\text{val}}=23,670$, $N_{\text{test}}=23,670$) with `icmp.transmit_timestamp` purged alongside `frame.time`, `ip.src_host`, `ip.dst_host`, `arp.src.proto_ipv4`, and `arp.dst.proto_ipv4`.
2. **Protocol Feature Space Optimization ($K=25 \to K=33$):** Expanding the feature selector from top-25 mutual-information features to all $K=33$ non-constant zero-leakage protocol features retains critical ARP (`arp.opcode`, `arp.hw.size`) and DNS (`dns.qry.qu`, `dns.qry.type`) indicators. This reduces irreducible feature-vector collisions on the locked test set from 532 rows (Bayes ceiling $99.17\%$) to 348 rows (Bayes ceiling $99.43\%$), lifting `MITM` recall from $36.26\%$ to **$100.00\%$** (`MITM` $F_1$: $53.23\% \to 83.87\%$) and overall Ensemble Macro-$F_1$ from **$95.53\%$** to **$97.73\%$** ($p = 8.41 \times 10^{-32}$).
3. **True Simplex Optimization & Post-Hoc Temperature Calibration:** Replaces heuristic inverse-loss weighting with constrained SLSQP convex optimization ($\mathbf{w}^* = [0.0000, 0.4610, 0.5390]^T$) and post-hoc temperature scaling ($T=0.90$), reducing Expected Calibration Error (ECE) by $10\times$ relative to CatBoost ($0.700\% \to 0.073\%$).
4. **Standardized FPR Definitions:** Explicitly reports both multiclass One-vs-Rest Macro FPR ($0.06\%$ at $K=33$, $0.08\%$ at $K=25$) and true Binary Benign FPR on `Normal` class index 7 ($0.33\%$, $12 / 3,645$ benign flows; Binary FNR $= 0.00\%$).

---

## 1. Verified Performance Metrics (Locked Test Set $N=23,670$)

| Model Architecture | Feature Set | Accuracy (%) | Balanced Acc (%) | Macro Prec (%) | Macro Recall (%) | Macro F1 (%) | Weighted F1 (%) | OvR Macro FPR (%) | Binary Benign FPR (%) | Log-Loss | ECE (%) | P50 Latency (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Majority-Class Baseline** | $K=25$ | 15.40% | 6.67% | 1.03% | 6.67% | 1.78% | 4.11% | 6.67% | 0.00% | 18.0420 | — | 0.004 |
| **Stratified Rule Baseline** | $K=25$ | 8.11% | 6.52% | 6.52% | 6.52% | 6.52% | 8.10% | 6.69% | 84.53% | 31.7370 | — | 0.146 |
| **Logistic Regression** | $K=25$ | 78.35% | 73.14% | 77.72% | 73.14% | 74.51% | 77.52% | 1.65% | 0.49% | 0.5421 | 3.412% | 0.109 |
| **Random Forest** | $K=25$ | 98.90% | 94.15% | 98.95% | 94.15% | 95.48% | 98.79% | 0.08% | 0.36% | 0.0482 | 0.615% | 13.478 |
| **Extra Trees** | $K=25$ | 97.16% | 89.79% | 96.42% | 89.79% | 91.86% | 96.99% | 0.21% | 0.58% | 0.0914 | 1.240% | 13.034 |
| **CatBoost (Baseline)** | $K=25$ | 98.93% | 94.19% | 98.87% | 94.19% | 95.47% | 98.82% | 0.08% | 0.33% | 0.0462 | 1.384% | **0.075** |
| **XGBoost (Baseline)** | $K=25$ | 98.93% | 94.19% | 98.97% | 94.19% | 95.51% | 98.82% | 0.08% | 0.33% | 0.0289 | 0.105% | 0.264 |
| **LightGBM (Baseline)** | $K=25$ | 98.96% | 94.22% | 99.00% | 94.22% | 95.54% | 98.85% | 0.07% | 0.33% | 0.0283 | 0.106% | 0.205 |
| **Hybrid Ensemble (Baseline)** | $K=25$ | 98.94% | 94.20% | 98.98% | 94.20% | 95.53% | 98.84% | 0.08% | 0.33% | 0.0286 | 0.130% | 1.337 |
| **CatBoost (Optimized)** | **$K=33$** | 99.18% | 98.18% | 97.96% | 98.18% | 97.67% | 99.18% | 0.06% | 0.33% | 0.0277 | 0.700% | **0.073** |
| **XGBoost (Optimized)** | **$K=33$** | 99.18% | 98.19% | 98.00% | 98.19% | 97.70% | 99.18% | 0.06% | 0.33% | 0.0222 | 0.082% | 0.235 |
| **LightGBM (Optimized)** | **$K=33$** | 99.20% | 98.21% | 98.01% | 98.21% | 97.72% | 99.20% | 0.06% | 0.33% | **0.0215** | **0.068%** | 0.187 |
| **Calibrated SLSQP Ensemble** | **$K=33$** | **99.21%** | **98.21%** | **98.03%** | **98.21%** | **97.73%** | **99.20%** | **0.06%** | **0.33%** | **0.0215** | **0.073%** | 1.316 |

---

## 2. Core Milestones Achieved
1. **Frozen Baseline & Upgraded Registry:** Preserved all $K=25$ baseline checkpoints (`models/registry/*_baseline_k25.pkl`) alongside the upgraded $K=33$ zero-leakage checkpoints (`models/registry/*_v2.pkl`).
2. **Reconciled Reports & IEEE Manuscript:** Generated the complete forensic audit registry (`reports/forensic_evidence_registry.csv`, `reports/FORENSIC_RECONCILIATION_AND_AUDIT_REPORT.md`) and double-column IEEE journal manuscript (`reports/IEEE_IoT_IDS_Manuscript.pdf`, `reports/IEEE_IoT_IDS_Manuscript.tex`, `reports/SCOPUS_REFERENCE_AUDIT_TABLE.md`).
3. **100% Automated Test Verification:** All unit, leakage, regression, integration, and latency tests in `tests/` pass against the live model artifacts.
