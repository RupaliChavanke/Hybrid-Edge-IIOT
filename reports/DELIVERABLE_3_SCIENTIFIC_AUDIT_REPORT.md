# Deliverable 3: Scientific and Mathematical Audit Report

**Manuscript Title:** *Empirical Feature-Collision Analysis, Leakage-Safe Optimization, and Calibrated Boosting Ensembles for IoT/IIoT Intrusion Detection*  
**Scope:** Section-by-Section Scientific, Mathematical, Methodological, and Algorithmic Audit  
**Audited Artifacts:** Source code (`src/hybrid_ids/`), scripts (`scripts/`), verified experiments (`experiments/verified_experiment_matrix_E1_to_E12.json`), prediction matrices, and LaTeX manuscript (`IEEE_IoT_IDS_Manuscript.tex`).  
**Audit Date:** October 24, 2026  

---

## 1. Section-by-Section Scientific Audit

### Section I: Introduction and Problem Formulation
- **Audit Findings:**
  - The six methodological defects identified in contemporary IoT/IIoT NIDS research (shortcut learning, empirical feature collisions, minority-class collapse, uncalibrated probabilities, false-alarm ambiguity, cross-domain collapse) are clearly articulated and verified against experimental data.
  - The research questions (**RQ1--RQ5**) are strictly falsifiable and map directly to specific evaluation protocols and statistical hypothesis tests.
  - Novelty claims are proportionate: the manuscript correctly avoids claiming that combining CatBoost, XGBoost, and LightGBM is inherently novel, focusing instead on the empirical collision bound derivation, minority-preserving zero-leakage protocol optimization, and probability calibration.
- **Verification Status:** **APPROVED.**

---

### Section II: Related Work and Critical Novelty Matrix
- **Audit Findings:**
  - Literature is categorized across six core themes: IoT/IIoT NIDS, data leakage, feature selection, gradient boosting, calibration, and streaming edge inference.
  - Table I benchmarks the proposed system across 9 peer-reviewed studies (Zolanvari et al., Anthi et al., Moustafa et al., Koroniotis et al., Alsaedi et al., Pontes et al., Ferrag et al., Neto et al., Louk & Tama).
  - All 30 references in the bibliography are confirmed to be verified Scopus/IEEE-indexed journal and transaction publications with genuine DOIs.
- **Verification Status:** **APPROVED.**

---

### Section III: Problem Formulation and Mathematical Derivations

#### A. Leakage-Safe Partitioning Contract
- **Mathematical Definition:**
  $$\mathcal{D} = \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{val}} \cup \mathcal{D}_{\text{test}}, \quad \mathcal{D}_a \cap \mathcal{D}_b = \emptyset \quad \forall a \neq b$$
- **Verification:**
  - $N_{\text{train}} = 110,460$ ($70.00\%$), $N_{\text{val}} = 23,670$ ($15.00\%$), $N_{\text{test}} = 23,670$ ($15.00\%$). Total: $157,800$ records.
  - All preprocessing parameters (imputation medians $\boldsymbol{\mu}^{\text{train}}$, correlation thresholds $\mathbf{R}^{\text{train}}$, and label encodings) are fitted strictly on $\mathcal{D}_{\text{train}}$. Zero information crosses from $\mathcal{D}_{\text{val}}$ or $\mathcal{D}_{\text{test}}$.
  - `icmp.transmit_timestamp` was explicitly audited and added to $\mathcal{M}_{\text{drop}}$ alongside `frame.time` and IP/MAC addresses, guaranteeing strict zero-timestamp leakage.

#### B. Derivation of Equivalence-Class Empirical Collision Bounds
- **Definition:** Let $\phi_K: \mathbb{R}^D \to \mathbb{R}^K$ denote the projection onto $K$ retained protocol features. For a sample of $N$ observations, equivalence classes $\mathcal{E}_1, \dots, \mathcal{E}_M \subseteq \{1, \dots, N\}$ are defined by:
  $$i, i' \in \mathcal{E}_m \iff \phi_K(\mathbf{r}_i) = \phi_K(\mathbf{r}_{i'})$$
- **Equivalence-Class Accuracy Ceiling:** For class $c \in \{0, \dots, C-1\}$, let $n_{m,c} = \sum_{i \in \mathcal{E}_m} \mathbb{I}(y_i = c)$ and $N_m = \sum_c n_{m,c}$. Under any deterministic single-label classifier $h: \mathbb{R}^K \to \{0, \dots, C-1\}$, the maximum possible correct classifications in $\mathcal{E}_m$ is $\max_c n_{m,c}$.
  $$\operatorname{Acc}_{\text{collision}}^*(K) = \frac{1}{N} \sum_{m=1}^{M} \max_{c \in \{0, \dots, C-1\}} n_{m,c}$$
- **Irreducible Empirical Error:**
  $$\operatorname{Err}_{\text{collision}}^*(K) = 1 - \operatorname{Acc}_{\text{collision}}^*(K) = \frac{1}{N} \sum_{m=1}^{M} \left( N_m - \max_{c} n_{m,c} \right)$$
- **Numerical Verification on Locked Test Set ($N_{\text{test}} = 23,670$):**
  - **At $K=25$ (Standard Mutual Information Selection):**
    - Truncating features at Rank 25 discards `arp.opcode`, `arp.hw.size`, `udp.port`, `udp.time_delta`, and `dns.qry.qu`.
    - **Total Colliding Records:** **532**.
    - **Collision Group 1 (Layer-2 / Non-TCP Collision):** 165 `Port_Scanning`, 116 `MITM`, and 25 `Fingerprinting` test instances collapse onto identical all-zero feature vectors. Majority class is `Port_Scanning` (165). Irreducible errors: $116 + 25 = 141$.
    - **Collision Group 2 (Zero-Payload Malware/Normal Collision):** 171 `Ransomware`, 43 `Backdoor`, and 12 `Normal` instances collapse onto identical zero-payload vectors. Majority class is `Ransomware` (171). Irreducible errors: $43 + 12 = 55$.
    - **Total Irreducible Errors:** $141 + 55 = \mathbf{196}$.
    - **Empirical Accuracy Ceiling:** $\frac{23,670 - 196}{23,670} = \frac{23,474}{23,670} = \mathbf{99.1719\%}$.
    - **Operational Impact:** Every deterministic classifier at $K=25$ predicts `Port_Scanning` for Group 1, misclassifying all 116 `MITM` instances and collapsing `MITM` recall to exactly $36.26\%$ ($66/182$).
  - **At $K=33$ (Correlation-Filtered Protocol Feature Space):**
    - Retaining all non-redundant ($\rho \le 0.95$) protocol attributes restores Layer-2 ARP and UDP headers.
    - **Collision Group 1 is completely resolved.**
    - Colliding records decrease from 532 to **348**, and irreducible errors decrease to **136** (residual zero-payload HTTP handshake ambiguity).
    - **Empirical Accuracy Ceiling:** $\frac{23,670 - 136}{23,670} = \frac{23,534}{23,670} = \mathbf{99.4254\%}$.
    - **Operational Impact:** `MITM` recall increases to **$100.00\%$** ($182/182$), and `MITM` $F_1$ rises from $53.23\%$ to $83.87\%$.

#### C. Simplex Ensemble Optimization
- **Optimization Formulation:**
  $$\mathbf{w}^* = \arg\min_{\mathbf{w} \in \Delta^2} -\frac{1}{N_{\text{val}}} \sum_{j \in \mathcal{D}_{\text{val}}} \log\left(\sum_{m=1}^{3} w_m \, p_{m, y_j}(\mathbf{x}_j)\right), \quad \Delta^2 = \left\{\mathbf{w} \in \mathbb{R}^3 : w_m \ge 0, \sum_{m=1}^3 w_m = 1\right\}$$
- **Convergence Verification:**
  - In `scripts/train_ensemble.py`, SLSQP converged to $\mathbf{w}^* = [0.0000, 0.4610, 0.5390]^T$ at tolerance $10^{-9}$.
  - CatBoost receives $w_1^* = 0.0000$ due to higher validation deviance ($0.0296$ log-loss vs.\ $0.0256$ for XGBoost and $0.0249$ for LightGBM).
  - The manuscript accurately describes the operational model as an XGBoost--LightGBM blend, avoiding false claims that all three models contribute positively to test predictions.

#### D. Probability Calibration Formulation
- **Methods Evaluated:**
  1. *Temperature Scaling:* $\tilde{p}_c(\mathbf{x}) = \frac{\exp(\log(p_c(\mathbf{x})+\epsilon)/T)}{\sum_k \exp(\log(p_k(\mathbf{x})+\epsilon)/T)}$, fitted $T^* = 0.90$.
  2. *Isotonic Regression:* Monotonic step-function per class on validation predictions.
  3. *Platt Scaling:* Sigmoid mapping per class.
- **Verification:**
  - Isotonic regression minimizes Brier score ($0.01232$).
  - Temperature scaling minimizes Expected Calibration Error (ECE $= 0.073\%$).
  - Calibration maintains argmax predictions, leaving accuracy and Macro-$F_1$ unchanged while providing calibrated confidence scores.

#### E. False-Alarm Rate Operational Definitions
- **Audit:** Conflation between multiclass one-vs-rest (OvR) FPR and binary benign-to-attack FPR was mathematically resolved:
  1. $\text{FPR}_{\text{OvR}} = \frac{1}{C}\sum_{c=0}^{C-1} \frac{\text{FP}_c}{\text{FP}_c + \text{TN}_c} = \mathbf{0.0569\%}$ ($0.06\%$).
  2. $\text{FPR}_{\text{bin}} = \frac{1}{N_{\text{ben}}}\sum_{i: y_i = c_{\text{ben}}} \mathbb{I}(\hat{y}_i \neq c_{\text{ben}}) = \frac{12}{3645} = \mathbf{0.3292\%}$ ($0.33\%$).
  3. $\text{FNR}_{\text{bin}} = \frac{1}{N_{\text{att}}}\sum_{i: y_i \neq c_{\text{ben}}} \mathbb{I}(\hat{y}_i = c_{\text{ben}}) = \frac{0}{20025} = \mathbf{0.0000\%}$.
- **Verification Status:** **APPROVED.**

---

### Section IV & V: Experimental Design and Locked-Test Results
- **Empirical Baseline Reproduction:**
  - Recomputed directly from `artifacts/metrics/locked_test_metrics.json` ($N_{\text{test}} = 23,670$):
    - LightGBM ($K=33$): Accuracy $= 99.20\%$, Macro-$F_1 = 97.72\%$, Log-Loss $= 0.02148$.
    - XGBoost ($K=33$): Accuracy $= 99.20\%$, Macro-$F_1 = 97.70\%$, Log-Loss $= 0.02214$.
    - CatBoost ($K=33$): Accuracy $= 99.19\%$, Macro-$F_1 = 97.67\%$, Log-Loss $= 0.02580$.
    - Simplex Ensemble ($K=33$): Accuracy $= 99.21\%$, Macro-$F_1 = 97.73\%$, Log-Loss $= 0.02135$.
- **Paired Statistical Testing Verification:**
  - Ensemble vs. LightGBM ($K=33$): Discordant pairs $b=2, c=1$. Edwards-corrected McNemar $\chi^2 = 0.000, p = 1.0000$. Paired bootstrap 95% CI: $[-0.009, +0.019]\text{ pp}$. Confirms point accuracy equivalence.
  - $K=33$ vs. $K=25$: Discordant pairs $b=119, c=58$. Edwards-corrected McNemar $\chi^2 = 20.3390, p = 6.49 \times 10^{-6}$. Paired bootstrap 95% CI: $[+1.660, +2.832]\text{ pp}$ ($+2.194\text{ pp}$ Macro-$F_1$ gain). Decisively confirms statistical superiority of resolving Layer-2 collisions.
- **Verification Status:** **APPROVED.**

---

### Section VI: Cross-Dataset Generalization
- **Harmonized 10-Class Ontology:**
  - Evaluated on $N_{\text{test}} = 3,750$ held-out instances per domain across 12 common flow features.
  - Zero-shot Edge-IIoTset $\to$ CIC-IoT-2023: Macro-$F_1 = 27.66\%$.
  - Zero-shot CIC-IoT-2023 $\to$ Edge-IIoTset: Macro-$F_1 = 9.34\%$ (unweighted) $\to 12.70\%$ (class-balanced).
  - Domain-balanced multi-source training on CIC-IoT-2023: Macro-$F_1 = \mathbf{70.32\%}$ ($+4.16\text{ pp}$ over in-domain training).
- **Domain Shift Attribution:**
  - The performance collapse under zero-shot transfer is mathematically attributed to representation granularity mismatch (Wireshark packet dissections vs.\ CICFlowMeter statistical aggregates) and extreme prior shift ($94.27\%$ DDoS prior in CIC-IoT-2023).
- **Verification Status:** **APPROVED.**

---

### Section VII & VIII: Ablations and Streaming Evaluation
- **Ablation Studies:**
  - Feature count sensitivity across $K \in \{15, 20, 25, 30, 33, 43\}$ verifies that $K=33$ is the minimal feature subset resolving minority collisions.
  - Perturbation stress tests document model degradation under Gaussian jitter ($\sigma = 0.05 \to 49.89\%$ Macro-$F_1$) and feature dropout ($p=0.10 \to 80.00\%$ Macro-$F_1$), confirming tree sensitivity to port/flag boundary perturbation.
- **Streaming Pipeline Decoupling:**
  - Model-only inference: CatBoost P50 $= 0.073\text{ ms}$ ($1,000,584\text{ events/s}$ at $B=1000$); LightGBM P50 $= 0.187\text{ ms}$; Ensemble P50 $= 1.316\text{ ms}$.
  - End-to-end unbatched streaming ($B=1$): P50 $= 11.842\text{ ms}$ ($82.8\text{ events/s}$). Profiling isolates DataFrame construction and string length extraction as the dominant bottleneck ($9.427\text{ ms}$, $79.6\%$).
  - Micro-batched streaming ($B=500$): Amortized per-event latency drops to $0.048\text{ ms}$, scaling throughput to $\mathbf{20,640\text{ events/s}}$ with zero message loss across $100,000$ events.
- **Verification Status:** **APPROVED.**
