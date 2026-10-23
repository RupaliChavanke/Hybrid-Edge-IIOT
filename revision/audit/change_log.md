# Manuscript Revision Change Log

**Manuscript Title:** *Empirical Feature-Collision Analysis, Leakage-Safe Optimization, and Calibrated Boosting Ensembles for IoT/IIoT Intrusion Detection*  
**Date:** October 24, 2026  
**Revision:** IEEE Transactions Refinement Cycle  

---

## 1. Structural and Section-by-Section Modifications

### Abstract
- **Previous State:** 318 words (exceeded standard IEEE Transactions limit of 250 words).
- **Revised State:** 221 words.
- **Modifications:** Condensed phrasing while strictly preserving all six required IEEE elements: problem statement, methodology, evaluation datasets and locked split size, principal quantitative finding ($K=25 \to K=33$ collision resolution and $+2.20\text{ pp}$ Macro-$F_1$ gain), secondary calibration/cross-dataset findings ($0.073\%$ ECE, $70.32\%$ multi-source Macro-$F_1$), and streaming throughput scaling ($20,640\text{ ev/s}$ at $B=500$).
- **Reason:** Full compliance with IEEE author guidelines requiring abstract $\le 250$ words.

### Section I: Introduction
- **Modifications:**
  - Renamed **RQ2** from "Feature Collision and Bayes Ceiling" to "Feature Collision and Accuracy Ceiling".
  - Refined **Contribution 1** from "Equivalence-Class Collision Proof and Bayes Bound Formulation" to "Equivalence-Class Collision Formulation and Accuracy Ceiling".
- **Reason:** Adheres strictly to Non-Negotiable Rule 6 (never describe an empirical collision ceiling on a fixed sample as a universal theoretical Bayes-optimal bound).

### Section II: Related Work
- **Table I Header Modification:** Replaced "Formal Bayes Bound" column entry for the proposed framework with "Collision Bound".
- **Reason:** Prevents theoretical overclaiming in literature benchmark table.

### Section III: Problem Formulation and Proposed Method
- **Modifications:**
  - Qualified Equation (15) as characterizing the maximum empirical accuracy achievable by deterministic classification on the evaluated sample and feature representation.
  - Formally differentiated the evaluation of three candidate base learners from the operational deployment of the active two-model blend resulting from SLSQP weight convergence ($\mathbf{w}^* = [0.0000, 0.4610, 0.5390]^T$).

### Section IV: Experimental Protocol
- **Section IV-D Expansion:** Added comprehensive *Reproducibility Protocol and Repository Validation Mapping* connecting data partitions, manifests, models, calibrations, ablations, and streaming scripts directly to public repository paths.
- **LaTeX Formatting:** Switched file paths to `\nolinkurl{...}` to enable hyphen-less wrapping at slashes and underscores in two-column format, eliminating all overfull `\hbox` warnings.

### Section V: Within-Dataset Results and Feature-Collision Analysis
- **Table IV (Baselines Benchmark):** Added repository validation footnote (`artifacts/metrics/baseline_results.csv`, `models/registry/`).
- **Table V (Optimized Models):** Added repository validation footnote (`artifacts/metrics/locked_test_metrics.json`, `experiments/verified_experiment_matrix_E1_to_E12.json`).
- **Table VI (Per-Class Performance):** Added repository validation footnote (`artifacts/per_class_metrics.csv`, `artifacts/metrics/per_class_baselines.csv`). Verified all delta arithmetic.
- **Table VII (Statistical Hypothesis Tests):** Added repository validation footnote (`artifacts/metrics/statistical_comparisons.csv`, `scripts/run_ablations_and_statistics.py`). Rephrased comparison between Ensemble and LightGBM to state that no statistically significant difference in point accuracy was detected ($p=1.00$).

### Section VI: Cross-Dataset Generalization
- **Table VIII (Cross-Dataset Results):** Added repository validation footnote (`artifacts/metrics/cross_dataset_results.csv`, `scripts/cross_dataset_evaluation.py`). Clarified target evaluation partitions and metrics.

### Section VII: Ablation and Sensitivity Analysis
- **Section VII-A (Feature-Cutoff Ablation):** Corrected prose to reflect that truncating at or below $K=25$ degrades MITM recall to $36.26\%$, whereas $K=30$ also recovers MITM recall ($100.00\%$). Clarified that $K=33$ is chosen due to complete 5-seed Jaccard stability ($1.000$ vs $0.905$ at $K=30$) and peak Macro-$F_1$ ($97.73\%$).
- **Table IX (Feature Cutoff Table):** Added repository validation footnote (`artifacts/metrics/ablation_results.csv`, `experiments/ablation_results.json`).
- **Table X (Perturbation Sensitivity):**
  - Corrected displayed $\Delta F_1$ values from legacy numbers ($-27.80, -45.64, -7.97, -15.53$) that were mistakenly subtracted from $K=25$ ($95.53\%$) to exact values relative to the clean $K=33$ baseline ($97.73\%$): $-30.00, -47.84, -10.17, -17.73\text{ pp}$.
  - Added explicit explanatory table note.
  - Added repository validation footnote (`artifacts/metrics/ablation_results.csv`, `reports/ablations/ablation_report.md`).

### Section VIII: Streaming and Resource Evaluation
- **Table XI (Streaming Benchmark):** Added repository validation footnote (`artifacts/metrics/runtime_latency_benchmark.json`, `scripts/benchmark_runtime.py`). Placed micro-batch scaling note cleanly before `\bottomrule`.

### Section XI: Conclusion and Future Work
- **Terminology Refinement:** Replaced "empirical Bayes accuracy ceiling" with "empirical feature-collision accuracy ceiling". Rephrased ensemble vs LightGBM comparison to "no statistically significant difference in point accuracy was detected".
