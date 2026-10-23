# Deliverable 7: Figure and Table Package

**Manuscript Title:** *Empirical Feature-Collision Analysis, Leakage-Safe Optimization, and Calibrated Boosting Ensembles for IoT/IIoT Intrusion Detection*  
**Audited Artifacts:** Vector graphics in `artifacts/figures/`, LaTeX source `IEEE_IoT_IDS_Manuscript.tex`, and PDF compilation.  
**Date:** October 24, 2026  

---

## 1. Publication Figure Registry

All figures are compiled as publication-quality vector PDFs and high-resolution visuals adhering strictly to IEEE two-column specifications:

### Figure 1: Pipeline Architecture and Dataflow
- **Source File:** Embedded in `IEEE_IoT_IDS_Manuscript.tex` (two-column boxed structure `\parbox{0.92\textwidth}`).
- **Caption:** *Figure 1: End-to-end architecture of the leakage-safe, calibrated gradient-boosted intrusion detection and streaming inference pipeline.*
- **Dataflow Steps:**
  1. *Telemetry Ingestion:* Dual-source ingestion (Edge-IIoTset $D=63$; CIC-IoT-2023 $D=47$).
  2. *Strict Partitioning Contract:* Stratified 70/15/15 split ($N_{\text{train}}=110,460$, $N_{\text{val}}=23,670$, locked $N_{\text{test}}=23,670$).
  3. *Training-Fitted Sanitization:* Purge IP/MAC identifiers, timestamps, and 22 zero-variance columns; extract 5 payload lengths and medians strictly on $\mathcal{D}_{\text{train}}$.
  4. *Protocol Feature Selection:* Collinearity filter $\tau_{\rho}=0.95$ yields $K=33$ protocol features, resolving Layer-2 ARP/DNS collisions.
  5. *Heterogeneous Boosting Triad:* Train CatBoost, XGBoost, and LightGBM with early stopping on validation deviance.
  6. *Simplex Blending & Calibration:* Solve $\mathbf{w}^* = [0.0000, 0.4610, 0.5390]^T$ via SLSQP; fit isotonic regression and temperature scaling ($T=0.90$).
  7. *Streaming Inference Engine:* Micro-batch vectorization ($B=500$) achieving $20,640\text{ events/s}$ with zero message loss.

---

### Figure 2: Per-Class $F_1$-Score Comparison ($K=25$ vs. $K=33$)
- **Vector PDF:** `artifacts/figures/fig2_per_class_f1_comparison.pdf`
- **Caption:** *Figure 2: Per-class $F_1$-score comparison across all 15 Edge-IIoTset classes between baseline ($K=25$) and optimized protocol ensemble ($K=33$).*
- **Visual Evidence:** Directly illustrates the $+30.64\text{ pp}$ $F_1$ recovery for `MITM` ($53.23\% \to 83.87\%$) and $+3.27\text{ pp}$ for `Port_Scanning` ($95.54\% \to 98.81\%$), while 7 attack classes maintain $100.00\%$ $F_1$.

---

### Figure 3: Cross-Dataset Generalization Dynamics
- **Vector PDF:** `artifacts/figures/fig3_cross_dataset_transfer.pdf`
- **Caption:** *Figure 3: Cross-dataset generalization and domain adaptation performance across 10-class Macro-$F_1$, Balanced Accuracy, and Binary Intrusion Macro-$F_1$ on held-out Edge-IIoTset ($N=3,750$) and CIC-IoT-2023 ($N=3,750$) test partitions.*
- **Visual Evidence:** Depicts severe zero-shot degradation ($27.66\%$ Edge$\to$CIC; $9.34\%$ CIC$\to$Edge) contrasted against domain-balanced multi-source recovery ($70.32\%$ on CIC-IoT-2023, $+4.16\text{ pp}$ above in-domain training).

---

### Figure 4: Probability Calibration Metrics
- **Vector PDF:** `artifacts/figures/fig4_calibration_metrics.pdf`
- **Caption:** *Figure 4: Probabilistic calibration metrics across base learners and ensembles at $K=33$: (a) Log-Loss, (b) Brier Score, and (c) ECE.*
- **Visual Evidence:** Shows overconfidence of uncalibrated CatBoost (ECE $=0.700\%$) and demonstrates how temperature scaling reduces ECE to $0.073\%$ while isotonic regression minimizes Brier score ($0.01232$).

---

### Figure 5: Streaming Latency Scaling and Throughput
- **Vector PDF:** `artifacts/figures/fig5_streaming_latency_scaling.pdf`
- **Caption:** *Figure 5: Streaming latency scaling: (a) Model-only latency percentiles (P50, P95, P99), and (b) End-to-end pipeline throughput and amortized latency across micro-batch sizes.*
- **Visual Evidence:** Compares standalone model inference (CatBoost P50: $0.073\text{ ms}$; LightGBM P50: $0.187\text{ ms}$) with unbatched streaming latency ($11.842\text{ ms}$ P50, $79.6\%$ DataFrame overhead) and amortized micro-batch scaling ($0.048\text{ ms/event}$, $20,640\text{ events/s}$ at $B=500$).

---

## 2. Publication Table Registry

| Table ID | Manuscript Title | Columns | Scope & Verification |
| :--- | :--- | :--- | :--- |
| **Table I** | Methodological Comparison Against Prior Literature | 11 columns | Benchmarks proposed system against 9 peer-reviewed studies across 10 criteria. |
| **Table II** | Dataset Partition Statistics and Canonical Mapping | 9 columns | Edge-IIoTset 70/15/15 partitions ($110,460 / 23,670 / 23,670$) and 10 shared canonical classes. |
| **Table III** | Cross-Dataset Feature Harmonization and Collision Audit | 6 columns | 12 common flow features, units, mapping fidelity, and duplicate/collision counts. |
| **Table IV (Alg. 1)** | Leakage-Safe Training and Streaming Pipeline | 1 column block | Step-by-step pseudo-code covering offline training, SLSQP optimization, and streaming inference. |
| **Table V** | Overall Classification Metrics on Locked Test Set | 10 columns | Evaluates 8 models on $N_{\text{test}}=23,670$ (Acc, Bal Acc, Macro-$F_1$, W-$F_1$, OvR FPR, Bin FPR, LogLoss, ECE, Size). |
| **Table VI** | Verified Per-Class Detection Metrics ($K=25$ vs. $K=33$) | 11 columns | 15 classes, exact test support, Precision, Recall, $F_1$, FPR, and $\Delta F_1$ (highlighting MITM $+30.64\text{ pp}$). |
| **Table VII** | Paired Statistical Significance Tests | 9 columns | Edwards-corrected McNemar $\chi^2$, $p$-values, 500-replicate bootstrap 95% CIs, Holm-Bonferroni correction. |
| **Table VIII** | Cross-Dataset Generalization by Target Domain | 9 columns | Evaluates 12 adaptation strategies on $N_{\text{test}}=3,750$ per domain across 10-class and binary metrics. |
| **Table IX** | Feature Cutoff Sensitivity, Stability, and Ablation | 6 columns | Evaluates $K \in \{15, 20, 25, 30, 33, 43\}$ (Jaccard stability, Accuracy, Macro-$F_1$, MITM recall). |
| **Table X** | Inference Latency, Pipeline Profiling, and Micro-Batching | 11 columns | Training time, P50/P95/P99 latency, throughput at $B=1,100,500$, memory RSS, and message drop rate. |

---

## 3. Formatting and Layout Compliance
- All tables utilize clean horizontal lines (`\toprule`, `\midrule`, `\bottomrule`) via `booktabs` with zero vertical rules, adhering to IEEE Transactions layout standards.
- Wide comparative tables (`Table I`, `Table II`, `Table III`, `Table V`, `Table VI`, `Table VII`, `Table VIII`, `Table X`) utilize `table*` spanning both columns with `\resizebox{\textwidth}{!}{...}`, preventing margin clipping.
- Caption placement strictly follows IEEE style: table titles above tables, figure captions below figures.
