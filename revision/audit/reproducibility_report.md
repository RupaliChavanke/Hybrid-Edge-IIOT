# Reproducibility and Experimental Audit Report

**Date:** October 24, 2026  
**Repository:** [Hybrid-Edge-IIOT](https://github.com/RupaliChavanke/Hybrid-Edge-IIOT.git)  
**Primary Manuscript:** `manuscript/IEEE_IoT_IDS_Manuscript.tex` / `revision/manuscript/revised_manuscript.tex`  
**Target Venue:** IEEE Internet of Things Journal / IEEE Transactions on Network and Service Management  

---

## 1. Executive Summary

This reproducibility report documents the independent verification, forensic artifact audit, and arithmetic recomputation of all experimental results, mathematical claims, and computational benchmarks presented in the manuscript. All 21 empirical claims (**C1--C21**) have been traced to raw data arrays, serialized model checkpoints, and reproduction scripts.

---

## 2. Experimental Environment

- **Workstation:** Apple Silicon M2 (8 CPU cores: 4 performance + 4 efficiency, 16.0 GB unified RAM)
- **Operating System:** macOS Darwin 24.3.0
- **Python Runtime:** Python 3.14.7 (macOS arm64 build)
- **Core ML Libraries:**
  - `scikit-learn`: 1.9.1
  - `catboost`: 1.2.10
  - `xgboost`: 3.4.1
  - `lightgbm`: 4.7.0
  - `numpy`: 2.3.5
  - `pandas`: 2.3.2
  - `scipy`: 1.15.1
- **Typesetting & Compilation:** Tectonic 0.15.0 (XeTeX-compatible engine, IEEEtran class)
- **Deterministic Seed Policy:** Strict repeatability enforced across all scripts using fixed seeds $s \in \{42, 43, 44, 45, 46\}$.

---

## 3. Dataset Partitions and Leakage-Safe Boundary Audit

### A. Edge-IIoTset Stratified Partitioning (Total: 157,800 records)
- **Training Set (70.00%):** $N_{\text{train}} = 110,460$ samples.
- **Validation Set (15.00%):** $N_{\text{val}} = 23,670$ samples (strictly reserved for SLSQP simplex weight optimization and temperature scaling).
- **Locked Test Set (15.00%):** $N_{\text{test}} = 23,670$ samples (strictly locked; evaluated exactly once after models, weights, and calibration parameters were frozen).
- **Leakage Contract Verification:**
  - Missing value imputation medians ($\boldsymbol{\mu}^{\text{train}}$) and pairwise Pearson correlation thresholds ($\mathbf{R}^{\text{train}}$) were fitted strictly on $\mathcal{D}_{\text{train}}$.
  - The drop matrix $\mathcal{M}_{\text{drop}}$ purged all session and device identifiers: `frame.time`, `ip.src_host`, `ip.dst_host`, `arp.src.proto_ipv4`, `arp.dst.proto_ipv4`, and `icmp.transmit_timestamp`.
  - Zero target or temporal leakage was detected across partition boundaries (`reports/split_audit.csv`).

---

## 4. Empirical Feature-Collision Analysis

### A. Mathematical Formulation
Let $\phi_K: \mathbb{R}^D \to \mathbb{R}^K$ denote the projection onto $K$ protocol features. For partition $\mathcal{D}$, equivalence classes $\mathcal{E}_1, \dots, \mathcal{E}_M$ group samples with identical feature vectors:
$$\mathbf{r}_i, \mathbf{r}_{i'} \in \mathcal{E}_m \iff \phi_K(\mathbf{r}_i) = \phi_K(\mathbf{r}_{i'}).$$
The empirical accuracy ceiling for deterministic single-label classification on this partition is:
$$\operatorname{Acc}_{\text{collision}}^*(K) = \frac{1}{N} \sum_{m=1}^{M} \max_{c \in \{0, \dots, C-1\}} n_{m,c},$$
where $n_{m,c} = \sum_{i \in \mathcal{E}_m} \mathbb{I}(y_i = c)$.

### B. Numerical Recomputation on Locked Test Set ($N_{\text{test}} = 23,670$)
1. **At $K=25$ (Standard Mutual Information Selection):**
   - Discards Layer-2 and low-frequency protocol fields: `arp.hw.size`, `arp.opcode`, `udp.time_delta`, `udp.port`, `dns.qry.qu`.
   - **Colliding records:** 532 instances across two major equivalence classes.
   - **Irreducible empirical errors:** 196 records ($141$ Layer-2 + $55$ zero-payload HTTP handshake ambiguity).
   - **Empirical Accuracy Ceiling:** $\frac{23,670 - 196}{23,670} = 99.1719\%$.
   - **Consequence:** 116 Man-in-the-Middle (`MITM`) instances collide with 165 `Port_Scanning` instances and are misclassified, collapsing `MITM` recall to $36.26\%$.
2. **At $K=33$ (Correlation-Filtered Protocol Features, $\rho \le 0.95$):**
   - Retains all non-redundant Layer-2 and transport headers.
   - **Colliding records:** Decreases to 348 instances.
   - **Irreducible empirical errors:** Decreases to 136 records (residual zero-payload TCP ambiguity).
   - **Empirical Accuracy Ceiling:** $\frac{23,670 - 136}{23,670} = 99.4254\%$.
   - **Consequence:** Layer-2 collision is completely resolved; `MITM` recall recovers to $100.00\%$ ($182/182$), lifting Macro-$F_1$ from $95.53\%$ to $97.73\%$ ($+2.20\text{ pp}$; McNemar $\chi^2 = 20.34, p = 6.49 \times 10^{-6}$).

---

## 5. Statistical Hypothesis Testing & Multiplicity Control

- **McNemar Tests:** Evaluated with Edwards continuity correction on aligned test predictions ($N=23,670$):
  - **Ensemble ($K=33$) vs. Ensemble ($K=25$):** Discordant errors $b=119, c=58 \implies \chi^2 = 20.3390, p = 6.49 \times 10^{-6}$ (Statistically significant after Holm--Bonferroni adjustment, $p_{\text{adj}} = 5.8 \times 10^{-5}$).
  - **Ensemble ($K=33$) vs. LightGBM ($K=33$):** Discordant errors $b=2, c=1 \implies \chi^2 = 0.0000, p = 1.0000$ (No statistically significant difference in point accuracy).
- **Paired Bootstrap:** 500 resamples of $\mathcal{D}_{\text{test}}$ generated empirical 95% confidence intervals:
  - $\Delta\text{Macro-}F_1 (K=33 - K=25) = +2.194\text{ pp}$ ($95\%\text{ CI}: [+1.660, +2.832]\text{ pp}$).
  - $\Delta\text{Macro-}F_1 (\text{Ens} - \text{LGB}) = +0.004\text{ pp}$ ($95\%\text{ CI}: [-0.009, +0.019]\text{ pp}$).

---

## 6. Calibration and Operational Deployment Distinction

- **SLSQP Simplex Weights:** Optimized on validation data log-loss:
  $$\mathbf{w}^* = [0.0000, 0.4610, 0.5390]^T.$$
- **Deployment Distinction:** Because CatBoost receives $0.0000$ weight, it is pruned from the forward inference path in production, executing as an active two-model blend (XGBoost + LightGBM). In offline benchmarks, both the three-model candidate evaluation and the active pipeline are profiled.
- **Probability Calibration:**
  - Uncalibrated CatBoost: $\text{ECE} = 0.700\%$.
  - Temperature Scaling ($T^* = 0.90$): $\text{ECE} = 0.073\%$, Brier Score $= 0.01245$.
  - Isotonic Regression: Brier Score $= 0.01232$, $\text{ECE} = 0.088\%$.

---

## 7. Cross-Dataset Generalization

- **Harmonization:** 12 common flow fields, 10 shared canonical attack families ($N=25,000$ per domain).
- **Zero-Shot Transfer Collapse:**
  - $\text{Edge-IIoTset} \to \text{CIC-IoT-2023}$: drops from $66.16\%$ (in-domain) to $27.66\%$ Macro-$F_1$ ($-38.50\text{ pp}$).
  - $\text{CIC-IoT-2023} \to \text{Edge-IIoTset}$: collapses to $9.34\%$ Macro-$F_1$ ($12.70\%$ with inverse-frequency class-weighting).
- **Multi-Source Recovery:** Domain- and class-balanced pooled training achieves $70.32\%$ Macro-$F_1$ on held-out CIC-IoT-2023 ($+4.16\text{ pp}$ above the $66.16\%$ in-domain baseline).

---

## 8. Streaming Latency vs. Throughput

- **Isolated Model Scoring (Apple Silicon M2):**
  - CatBoost Alone: $\text{P50} = 0.073\text{ ms}$ ($1,000,584\text{ events/s}$ at $B=1,000$).
  - LightGBM Alone: $\text{P50} = 0.187\text{ ms}$ ($1.56\text{ MB}$ memory footprint).
  - 3-Model Ensemble: $\text{P50} = 1.316\text{ ms}$ ($29,604\text{ events/s}$).
- **End-to-End Streaming Pipeline:**
  - Single-event ($B=1$): $\text{P50} = 11.842\text{ ms}$ ($82.8\text{ events/s}$), dominated by per-record Pandas vectorization ($9.427\text{ ms}$, $79.6\%$).
  - Micro-batching ($B=500$): amortized latency decreases to $0.048\text{ ms/event}$, scaling throughput to $20,640\text{ events/s}$ with zero message loss and $267\text{ MB}$ RSS under $100,000$ telemetry events.
