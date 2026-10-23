# Deliverable 10: Submission Readiness Checklist and Scope Assessment

**Manuscript Title:** *Empirical Feature-Collision Analysis, Leakage-Safe Optimization, and Calibrated Boosting Ensembles for IoT/IIoT Intrusion Detection*  
**Audited Target Journals:**  
1. **Primary Target:** *IEEE Internet of Things Journal* (IEEE IoT-J)  
2. **Alternative Target:** *IEEE Transactions on Network and Service Management* (IEEE TNSM)  
**Audit Date:** October 24, 2026  

---

## 1. Quality-Gate Verification Checklist (Phase 15 Audit)

| Check Item | Requirement Description | Verification Evidence | Audit Status |
| :---: | :--- | :--- | :---: |
| **QG-01** | Every abstract claim matches final verified results. | Verified against `locked_test_metrics.json` and Table V--X. | **PASSED** |
| **QG-02** | Dataset counts and partitions reconcile exactly. | Edge-IIoTset: 110,460 / 23,670 / 23,670 (Total: 157,800). Verified in `split_audit.csv`. | **PASSED** |
| **QG-03** | No preprocessing or model selection uses locked-test labels. | Contract verified: `imputer`, `selector`, `encoder` fitted strictly on `df_train`. | **PASSED** |
| **QG-04** | Feature-collision derivation and reported counts are reproducible. | $K=25$: 532 rows, 196 errors ($99.1719\%$). $K=33$: 348 rows, 136 errors ($99.4254\%$). | **PASSED** |
| **QG-05** | Every performance metric is recomputed from raw vectors. | Confusion matrix, Macro-$F_1$, precision, recall, OvR FPR, binary FPR verified. | **PASSED** |
| **QG-06** | Statistical tests use correct paired predictions. | Edwards-corrected McNemar ($\chi^2 = 20.34, p = 6.49\times 10^{-6}$); $p=1.00$ for Ens vs LGB. | **PASSED** |
| **QG-07** | Bootstrap confidence intervals are reproducible. | 500-replicate paired bootstrap: $\Delta\text{Macro-}F_1 = +2.19\text{ pp}$ ($95\%$ CI $[+1.66, +2.83]\text{ pp}$). | **PASSED** |
| **QG-08** | Calibration methods correctly associated with metrics. | Isotonic minimizes Brier ($0.01232$); Temperature ($T=0.90$) minimizes ECE ($0.073\%$). | **PASSED** |
| **QG-09** | Cross-dataset label mappings are transparent. | 10 shared canonical attack classes mapped explicitly in Table II and Table III. | **PASSED** |
| **QG-10** | Streaming latency and throughput define timing boundaries. | Standalone model inference decoupled from end-to-end DataFrame overhead ($79.6\%$). | **PASSED** |
| **QG-11** | All tables and figures agree with underlying data. | Verified across 5 figures and 10 tables against JSON/CSV evidence files. | **PASSED** |
| **QG-12** | All references and DOIs checked and verified. | 30 / 30 references verified in Scopus/IEEE with genuine, resolveable DOIs. | **PASSED** |
| **QG-13** | Limitations and negative findings reported honestly. | Inactive CatBoost weight, zero-shot transfer collapse, and noise sensitivity documented. | **PASSED** |
| **QG-14** | Novelty claims are proportionate to actual evidence. | Emphasizes collision bounds, leakage-safe protocol selection, and calibration. | **PASSED** |
| **QG-15** | Manuscript follows current IEEE author instructions. | Two-column IEEEtran, proper section hierarchy, no internal drafting notes. | **PASSED** |
| **QG-16** | LaTeX source compiles without errors. | Compiled via Tectonic to `IEEE_IoT_IDS_Manuscript.pdf` with 0 errors (13 pages). | **PASSED** |
| **QG-17** | No unresolved contradictions hidden or silently altered. | All 8 contradictions documented in Discrepancy Register (`DELIVERABLE_5`). | **PASSED** |

---

## 2. Journal Scope Fit Assessment

### Primary Target: *IEEE Internet of Things Journal* (IEEE IoT-J)
- **Scope Alignment:** Exceptional. The manuscript addresses cyber-physical and network security across Industrial IoT (IIoT) protocols (Modbus, MQTT, BACnet) and smart-home IoT devices.
- **Key Strengths for IEEE IoT-J:**
  - Evaluates both the flagship industrial dataset (*Edge-IIoTset*) and smart-home dataset (*CIC-IoT-2023*).
  - Explicitly addresses edge gateway resource constraints ($1.56\text{ MB}$ LightGBM model footprint, sub-$100\,\mu\text{s}$ inference).
  - Profiles Redpanda-compatible streaming telemetry ingestion for edge-to-cloud architectures.
- **Recommendation:** **Submit as Primary Target.**

### Alternative Target: *IEEE Transactions on Network and Service Management* (IEEE TNSM)
- **Scope Alignment:** High. IEEE TNSM focuses on network operations, monitoring, management architectures, and cross-domain operational telemetry.
- **Key Strengths for IEEE TNSM:**
  - Mathematical characterization of feature collisions in operational network monitoring.
  - End-to-end streaming telemetry pipeline benchmarking and micro-batch throughput optimization ($20,640\text{ events/s}$).
  - Operational false-alarm analysis ($\text{FPR}_{\text{bin}} = 0.33\%$, $\text{FNR}_{\text{bin}} = 0.00\%$) for Security Operations Centers (SOC).
- **Recommendation:** Strong alternative if editor prioritizes network operations over IoT device telemetry.

---

## 3. Disclosures and Ethical Compliance
- **Funding / Competing Interests:** No commercial conflicts of interest or third-party sponsorship influencing reported experimental outcomes.
- **Data Availability:** All processed partitions, trained models, prediction matrices, and validation scripts are archived under the GitHub repository suite (`https://github.com/RupaliChavanke/Hybrid-Edge-IIOT.git`).
- **Submission Readiness Status:** **CERTIFIED READY FOR SUBMISSION.**
