# Deliverable 5: Discrepancy Register and Numerical Reconciliation Log

**Manuscript Title:** *Empirical Feature-Collision Analysis, Leakage-Safe Optimization, and Calibrated Boosting Ensembles for IoT/IIoT Intrusion Detection*  
**Audited Artifacts:** Repository code, logs, previous markdown reports, CSV/JSON outputs, and LaTeX manuscript.  
**Audit Date:** October 24, 2026  

---

## 1. Discrepancy Register

| Discrepancy ID | Location in Artifacts | Original / Conflicting Value | Independently Recomputed Value | Cause / Mechanism | Downstream Impact & Resolution | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DISC-01** | `baseline_results.csv`, `ensemble_comparison.csv` | FPR reported as $2.81\%$ (XGB/LGB) and $2.94\%$ (CatBoost) | Binary $\text{FPR} = \mathbf{0.33\%}$ ($12/3645$); OvR Macro $\text{FPR} = \mathbf{0.06\%}$ | `train_baselines.py` coded `benign_mask = (y_true == 0)`, but index `0` is `"Backdoor"` while `"Normal"` is index `7`. Script calculated the false-negative rate of Backdoor. | Corrected formulas in manuscript. Explicitly documented OvR Macro FPR ($0.06\%$) vs. true binary benign-to-attack FPR ($0.33\%$). | **RESOLVED & RECONCILED** |
| **DISC-02** | `cross_dataset_results.csv`, `cross_dataset_evaluation.md` | In-domain Macro-$F_1 = 16.50\%$; Transfer Macro-$F_1 = 0.00\%$ | In-domain Macro-$F_1 = \mathbf{59.05\%}$; Transfer Macro-$F_1 = \mathbf{27.66\%}$ | `cross_dataset_evaluation.py` used `pd.read_csv(nrows=40000)` on class-sorted CSV without shuffling. Training split received classes 0--4; test split received classes 5--9. | Re-ran evaluation with stratified sampling across full dataset. Restored genuine transfer benchmarks ($27.66\%$ Edge$\to$CIC, $70.32\%$ domain-balanced pooled). | **RESOLVED & RECONCILED** |
| **DISC-03** | `reports/final/model_comparison.md` (Sec. 3 table) | MITM test support $= 1,050$, MITM $F_1 = 99.28\%$ | Exact test support $= \mathbf{182}$, verified $F_1 = \mathbf{53.23\%}$ ($K=25$) / $\mathbf{83.87\%}$ ($K=33$) | Unverified manual markdown table containing fictitious supports and inflated metrics. | Formally retracted unverified table. Manuscript uses strictly audited confusion matrix values ($N_{\text{test}}=182$). | **RESOLVED & RETRACTED** |
| **DISC-04** | Repository documentation & legacy reports | Test set size reported alternately as $N = 23,548$ and $N = 23,670$ | Exact primary test set: $N_{\text{test}} = \mathbf{23,670}$ | $N = 23,548$ was $15\%$ of $156,986$ rows in the superseded legacy v1.0 neural pipeline after dropping 814 duplicate rows. Primary v2.0/v2.1 uses $15\%$ of all $157,800$ rows. | Clearly segregated legacy v1.0 reports. Standardized primary manuscript strictly on $N_{\text{test}} = 23,670$. | **RESOLVED & STANDARDIZED** |
| **DISC-05** | `src/hybrid_ids/models/ensemble.py` vs. `scripts/train_ensemble.py` | `strategy="uniform"` vs. SLSQP weights $\mathbf{w}^* = [0.0000, 0.4610, 0.5390]^T$ | SLSQP optimal weights: $\mathbf{w}^* = [0.0000, 0.4610, 0.5390]^T$ | Nelder-Mead on validation Macro-F1 fell back to uniform averaging; SLSQP on validation cross-entropy found interior convex optimum with zero weight on CatBoost. | Reported SLSQP convex weights alongside uniform baseline. Clarified that operational predictor is an XGBoost--LightGBM blend. | **RESOLVED & DISCLOSED** |
| **DISC-06** | `reports/ablations/ablation_report.md` | Narrative claimed "$<2.5\%$ degradation under noise", while table showed Macro-$F_1$ drop to $49.89\%$ | Table verified: $\sigma=0.05 \to \mathbf{49.89\%}$ Macro-$F_1$ ($-45.64\text{ pp}$ drop) | Narrative was boilerplate text copied from unrelated synthetic tests that contradicted the actual perturbation table. | Overwrote misleading narrative. Explained that continuous jitter crosses discrete TCP flag thresholds, causing severe degradation. | **RESOLVED & CORRECTED** |
| **DISC-07** | Manuscript draft line 17 | Footnote contained: `Target Venue: IEEE IoT Journal / IEEE TNSM` | Footnote removed from submission copy | Editorial note left in author thanks block during drafting. | Removed editorial note to ensure strict compliance with IEEE submission instructions. | **RESOLVED & CLEANED** |
| **DISC-08** | Section III-B Equation 2 | Termed "universal Bayes optimal accuracy ceiling" | "Finite-sample empirical accuracy ceiling under equivalence-class majority voting" | Semantic imprecision: empirical bound on sample representation was described as a universal population law. | Rephrased text and derivations to establish precisely what the finite-sample bound proves. | **RESOLVED & REPHRASED** |

---

## 2. Integrity Certification
All identified discrepancies across code, documentation, and draft manuscripts have been documented, traced to their computational root causes, and resolved. Zero unverified numbers remain in the manuscript.
