# Hybrid Ensemble Optimization & Architecture Analysis (Phase 8 — Reconciled)

**Evaluation Date:** October 2026  
**Evaluation Partition:** 15% Strictly Locked Test Set ($N=23,670$ samples)  
**Tuning Partition:** 15% Validation Split ($N=23,670$; Constrained SLSQP Simplex & Post-Hoc Calibration)  

## 1. Comprehensive Ensemble Comparison Table ($K=25$ Frozen Baseline vs. $K=33$ Zero-Leakage Protocol Features)

| Architecture Variant | Feature Set | Accuracy (%) | Balanced Acc (%) | Macro F1 (%) | Weighted F1 (%) | OvR Macro FPR (%) | Binary Benign FPR (%) | Log-Loss | Brier Score | ECE (%) | P50 Latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Variant A0: CatBoost Alone (Baseline)** | $K=25$ | 98.93% | 94.19% | 95.47% | 98.82% | 0.08% | 0.33% | 0.04622 | 0.01696 | 1.384% | 0.075 ms |
| **Variant B0: XGBoost Alone (Baseline)** | $K=25$ | 98.93% | 94.19% | 95.51% | 98.82% | 0.08% | 0.33% | 0.02892 | 0.01609 | 0.105% | 0.264 ms |
| **Variant C0: LightGBM Alone (Baseline)** | $K=25$ | 98.96% | 94.22% | 95.54% | 98.85% | 0.07% | 0.33% | 0.02833 | 0.01564 | 0.106% | 0.205 ms |
| **Variant D0: Hybrid Ensemble (Baseline)** | $K=25$ | 98.94% | 94.20% | 95.53% | 98.84% | 0.08% | 0.33% | 0.02863 | 0.01583 | 0.130% | 1.337 ms |
| **Variant A: CatBoost Alone (Optimized)** | **$K=33$** | 99.18% | 98.18% | 97.67% | 99.18% | 0.06% | 0.33% | 0.02771 | 0.01310 | 0.700% | **0.073 ms** |
| **Variant B: XGBoost Alone (Optimized)** | **$K=33$** | 99.18% | 98.19% | 97.70% | 99.18% | 0.06% | 0.33% | 0.02217 | 0.01274 | 0.082% | 0.235 ms |
| **Variant C: LightGBM Alone (Optimized)** | **$K=33$** | 99.20% | 98.21% | 97.72% | 99.20% | 0.06% | 0.33% | **0.02148** | 0.01237 | **0.068%** | 0.187 ms |
| **Variant D: Uniform Averaging** | **$K=33$** | 99.20% | 98.20% | 97.71% | 99.20% | 0.06% | 0.33% | 0.02186 | 0.01250 | 0.204% | 0.495 ms |
| **Variant E: Validation Weighted (Simplex SLSQP)** | **$K=33$** | 99.21% | 98.21% | 97.73% | 99.20% | 0.06% | 0.33% | 0.02153 | 0.01246 | 0.083% | 0.495 ms |
| **Variant F: SLSQP + Temperature Scaling ($T=0.90$)** | **$K=33$** | **99.21%** | **98.21%** | **97.73%** | **99.20%** | **0.06%** | **0.33%** | 0.02151 | 0.01245 | **0.073%** | 1.316 ms |
| **Variant G: SLSQP + Isotonic Calibration** | **$K=33$** | 99.21% | 98.21% | 97.72% | 99.20% | 0.06% | 0.33% | 0.02418 | **0.01232** | 0.072% | 1.320 ms |
| **Variant H: SLSQP + Platt Sigmoid Calibration** | **$K=33$** | 99.20% | 98.20% | 97.70% | 99.19% | 0.06% | 0.33% | 0.02685 | 0.01241 | 0.151% | 1.318 ms |

## 2. Simplex SLSQP Optimization Weights ($K=33$ Validation Log-Loss Minimization)
- **CatBoost Weight ($w_1$):** `0.0000`
- **XGBoost Weight ($w_2$):** `0.4610`
- **LightGBM Weight ($w_3$):** `0.5390`
- **Post-Hoc Temperature Parameter ($T^*$):** `0.90`

*(Under validation Macro-$F_1$ objective via Powell search, the optimal simplex weights are $\mathbf{w}_{\text{F1}}^* = [0.2194, 0.3712, 0.4094]^T$.)*

## 3. Deep Scientific Analysis
1. **Why Constrained SLSQP Assigns $w_1 = 0$ to CatBoost Under Log-Loss:**
   - Symmetric oblivious trees (CatBoost) exhibit higher multiclass negative log-likelihood ($\text{LogLoss}=0.02771$, $\text{ECE}=0.700\%$) than leaf-wise LightGBM ($\text{LogLoss}=0.02148$, $\text{ECE}=0.068\%$) and depth-wise XGBoost ($\text{LogLoss}=0.02217$, $\text{ECE}=0.082\%$). Constrained convex optimization on $\Delta^2$ projects onto the XGBoost–LightGBM edge ($w_2=0.4610, w_3=0.5390$).
2. **Why Platt Sigmoid Scaling Degrades Log-Loss on Already-Calibrated GBDT Ensembles:**
   - Fitting 15 independent One-vs-Rest logistic sigmoids and normalizing row sums distorts rare-class tail probabilities, raising Log-Loss from $0.02153$ to $0.02685$. Single-parameter Temperature Scaling ($T=0.90$) preserves the simplex ranking while improving both Log-Loss ($0.02151$) and ECE ($0.073\%$).
3. **Standardized FPR Reconciliation ($2.81\%$ vs. $0.33\%$ vs. $0.06\%$):**
   - Earlier scripts (`scripts/train_baselines.py` and `scripts/train_ensemble.py`) mistakenly indexed Class `0` (`Backdoor`, alphabetical index 0) as the benign class when computing binary FPR ($43/1529 = 2.81\%$). Explicitly indexing `Normal` (Class `7`, $N=3,645$) yields a true Binary Benign FPR of **$0.33\%$** ($12/3,645$) and a Binary FNR of **$0.00\%$** ($0/20,025$ attacks missed), alongside a One-vs-Rest multiclass Macro FPR of **$0.06\%$** ($K=33$) and **$0.08\%$** ($K=25$).
