# Fair Baseline Experimental Evaluation Report (Phase 6 — Reconciled)

**Evaluation Date:** October 2026  
**Evaluation Protocol:** 15% Strictly Locked Test Partition ($N=23,670$ records)  
**Input Features:** Frozen Baseline ($K=25$) vs. Full Zero-Leakage Protocol Feature Set ($K=33$)  
**Taxonomy:** 15 multiclass Edge-IIoTset cyberattack classes  

## Overall Model Benchmark Table ($N=23,670$)

| Model | Feature Set | Accuracy (%) | Balanced Acc (%) | Macro F1 (%) | Weighted F1 (%) | OvR Macro FPR (%) | Binary Benign FPR (%) | Binary FNR (%) | ROC-AUC | P50 Latency (ms) | Throughput (samples/s) | Size (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Majority_Class** | $K=25$ | 15.40% | 6.67% | 1.78% | 4.11% | 6.67% | 0.00% | 100.00% | 0.5000 | 0.004 | 65,214,584 | 0.00 |
| **Stratified_Rule** | $K=25$ | 8.11% | 6.52% | 6.52% | 8.10% | 6.69% | 84.53% | 15.42% | 0.4992 | 0.146 | 2,185,000 | 0.00 |
| **Logistic_Regression** | $K=25$ | 78.35% | 73.14% | 74.51% | 77.52% | 1.65% | 0.49% | 1.91% | 0.9735 | 0.109 | 2,937,582 | 0.01 |
| **Random_Forest** | $K=25$ | 98.90% | 94.15% | 95.48% | 98.79% | 0.08% | 0.36% | 0.00% | 0.9997 | 13.478 | 37,877 | 6.71 |
| **Extra_Trees** | $K=25$ | 97.16% | 89.79% | 91.86% | 96.99% | 0.21% | 0.58% | 0.00% | 0.9992 | 13.034 | 36,751 | 10.21 |
| **CatBoost (Baseline)** | $K=25$ | 98.93% | 94.19% | 95.47% | 98.82% | 0.08% | 0.33% | 0.00% | 0.9997 | **0.075** | 1,013,441 | 5.44 |
| **XGBoost (Baseline)** | $K=25$ | 98.93% | 94.19% | 95.51% | 98.82% | 0.08% | 0.33% | 0.00% | 0.9997 | 0.264 | 271,445 | 3.07 |
| **LightGBM (Baseline)** | $K=25$ | 98.96% | 94.22% | 95.54% | 98.85% | 0.07% | 0.33% | 0.00% | 0.9997 | 0.205 | 150,432 | 1.56 |
| **Hybrid Ensemble (Baseline)** | $K=25$ | 98.94% | 94.20% | 95.53% | 98.84% | 0.08% | 0.33% | 0.00% | 0.9997 | 1.337 | 83,395 | 10.07 |
| **CatBoost (Optimized)** | **$K=33$** | 99.18% | 98.18% | 97.67% | 99.18% | 0.06% | 0.33% | 0.00% | 0.9998 | **0.073** | **1,000,584** | 5.48 |
| **XGBoost (Optimized)** | **$K=33$** | 99.18% | 98.19% | 97.70% | 99.18% | 0.06% | 0.33% | 0.00% | 0.9998 | 0.235 | 256,803 | 3.12 |
| **LightGBM (Optimized)** | **$K=33$** | 99.20% | 98.21% | 97.72% | 99.20% | 0.06% | 0.33% | 0.00% | 0.9998 | 0.187 | 133,388 | 1.62 |
| **Calibrated SLSQP Ensemble** | **$K=33$** | **99.21%** | **98.21%** | **97.73%** | **99.20%** | **0.06%** | **0.33%** | **0.00%** | **0.9998** | 1.316 | 77,315 | 10.22 |

## Key Findings & Forensic Reconciliation
1. **Reconciliation of the $2.81\%$ vs. $0.33\%$ vs. $0.08\%$ FPR Discrepancy:**
   - Earlier baseline scripts computed binary FPR by assuming Class `0` was Benign. Because `LabelEncoder` sorts class names alphabetically, Class `0` is `Backdoor` ($N=1,529$, where $43/1,529 = 2.81\%$ were misclassified as `Normal`), while `Normal` is Class `7` ($N=3,645$).
   - Computing true Binary Benign FPR against Class `7` (`Normal`) yields **$0.33\%$** ($12/3,645$ benign packets flagged as attacks) and a Binary FNR of **$0.00\%$** ($0/20,025$ attacks missed). Computing One-vs-Rest multiclass Macro FPR yields **$0.08\%$** at $K=25$ and **$0.06\%$** at $K=33$.
2. **Rule-Based & Linear Baselines Fail on Non-Linear Multi-Protocol Signatures:**
   - Majority Class baseline achieves $15.40\%$ accuracy (matching the `Normal` class prior) and $1.78\%$ Macro-$F_1$.
   - Logistic Regression achieves **$78.35\%$ accuracy and $74.51\%$ Macro-$F_1$** on $K=25$ (reconciling an older prose typo that had cited $71.3\% / 58.2\%$).
3. **GBDT Triad & $K=33$ Protocol Feature Superiority:**
   - Retaining all $K=33$ non-constant zero-leakage protocol features lifts GBDT Macro-$F_1$ from $95.47\%$–$95.54\%$ ($K=25$) to **$97.67\%$–$97.73\%$** ($K=33$) while maintaining sub-millisecond single-model P50 latency ($0.073\,\text{ms}$ for CatBoost).
