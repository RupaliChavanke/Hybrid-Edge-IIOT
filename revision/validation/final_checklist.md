# Phase 12: Automated and Manual Quality Gates Checklist

**Manuscript Title:** *Empirical Feature-Collision Analysis, Leakage-Safe Optimization, and Calibrated Boosting Ensembles for IoT/IIoT Intrusion Detection*  
**Date:** October 24, 2026  
**Compiled Artifact:** `revision/manuscript/revised_manuscript.pdf` (13 pages, IEEEtran twocolumn)  

---

### 1. Scientific Integrity
- [x] **Every major claim is traceable to evidence:** All 21 claims (**C1--C21**) verified in `claim_evidence_matrix.csv`.
- [x] **Every critical numerical discrepancy is resolved:** Documented in `discrepancy_register.csv` (7 resolved issues, 0 unresolved).
- [x] **No fabricated data, citations, metrics, or experiments exist:** All values originate from logged executions and genuine Scopus/IEEE DOIs.
- [x] **No locked-test leakage is introduced:** Imputation medians, correlation matrices, and ensembling weights fitted strictly on training/validation partitions; test partition evaluated strictly once.
- [x] **Empirical bounds are described with correct assumptions:** Defined as empirical feature-collision accuracy ceilings under finite-sample representation, not universal theoretical population Bayes bounds.
- [x] **Statistical conclusions match reproduced tests:** McNemar paired tests and 500-replicate bootstrap CIs verified; non-significant differences stated as "no statistically significant difference was detected".

### 2. Methodological Clarity
- [x] **Dataset splits and sample counts are consistent:** Stratified 70/15/15 split ($N=110,460 / 23,670 / 23,670$; total $157,800$) verified across all sections and tables.
- [x] **Feature selection and preprocessing fit boundaries are clear:** Protocol features ($K=33$) selected via Pearson correlation filtering ($\rho \le 0.95$) strictly on $\mathcal{D}_{\text{train}}$.
- [x] **Calibration and threshold-selection protocols are documented:** Temperature scaling ($T^*=0.90$) and isotonic regression fitted on validation blend $\mathbf{P}_{\text{ens}}(\mathbf{w}^*)$.
- [x] **Cross-dataset ontology and transfer directions are explicit:** 12 harmonized features, 10 shared attack classes; zero-shot transfer evaluated in both directions ($\text{Edge}\to\text{CIC}$ and $\text{CIC}\to\text{Edge}$).
- [x] **Latency and throughput measurement boundaries are clear:** Standalone model scoring ($0.073\text{ ms}$ P50) explicitly decoupled from end-to-end streaming ingestion ($11.842\text{ ms}$ P50 at $B=1$; $0.048\text{ ms}$ at $B=500$).

### 3. Structural Quality
- [x] **Research questions map to results and conclusions:** **RQ1--RQ5** answered directly across Sections V, VI, VII, VIII, and IX.
- [x] **The proposed method is reproducible from the description:** Algorithm 1 and mathematical equations define exact steps.
- [x] **Results and interpretation are distinguishable:** Empirical metric tables presented before narrative interpretation.
- [x] **Discussion does not repeat the Results section:** Section IX synthesizes trade-offs, operational implications, and compares with related literature.
- [x] **Limitations are candid and specific:** Section X covers single-record evaluation limitations, synthetic testbed traffic differences, and physical distributed cluster networking latency boundaries.
- [x] **Conclusion contains no unsupported claims:** Summarizes only verified findings with exact numerical values.

### 4. Tables and Figures
- [x] **All metrics and deltas are recalculated:** Table X perturbation deltas corrected to $-30.00, -47.84, -10.17, -17.73\text{ pp}$ relative to clean $K=33$ baseline ($97.73\%$).
- [x] **Captions, labels, and text agree:** Table footnotes provide exact repository validation paths.
- [x] **All figures are legible in IEEE two-column layout:** Figures 1--5 verified for clear font scaling and vector rendering.
- [x] **No clipped, overlapping, or missing elements remain:** Tectonic compilation completed with zero overfull `\hbox` warnings.
- [x] **All cross-references resolve correctly:** Sections, figures, tables, and equations numbered sequentially without `??` broken links.

### 5. IEEE Production
- [x] **Correct template and class for intended venue:** `\documentclass[10pt,journal,twocolumn]{IEEEtran}`.
- [x] **Correct title, abstract, index terms, and bibliography style:**
  - Abstract length: **221 words** (strictly within $\le 250$ word limit).
  - IEEE keywords: 9 standard terms included.
  - References: 30 verified Scopus/IEEE journal and transaction citations with valid DOIs.
- [x] **No compilation errors or missing references remain:** Tectonic exit code 0.
- [x] **Page layout and typography inspected in compiled PDF:** Exactly 13 double-column pages matching target length (10--14 pages).

---

### Certification
**Status:** **APPROVED FOR SUBMISSION / EDITORIAL REVIEW.**  
All quality gates have been satisfied without unresolved discrepancies.
