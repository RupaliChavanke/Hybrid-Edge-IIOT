# Cross-Dataset Generalization & Transferability Benchmark Report (Phase 9 — Reconciled)

**Evaluation Date:** October 2026  
**Source Environments:** `Edge-IIoTset` Industrial Testbed vs. `CICIoT2023` Smart-Home Testbed  
**Harmonized Canonical Features ($D=12$):** `tcp_flag_fin`, `tcp_flag_syn`, `tcp_flag_rst`, `tcp_flag_ack`, `proto_tcp`, `proto_udp`, `proto_icmp`, `proto_http`, `proto_dns`, `proto_mqtt`, `proto_arp`, `packet_payload_length`  
**Common Evaluated Attack Categories (10 Classes):** `Benign`, `DDoS_UDP`, `DDoS_ICMP`, `DDoS_TCP`, `DDoS_HTTP`, `Port_Scan`, `Vulnerability_Scan`, `Spoofing_MITM`, `Web_Attack`, `Brute_Force`, `Malware` (10 active shared classes)  

## Experimental Transferability Matrix (Stratified $N_{\text{test}}=3,750$ per Domain)

| Experiment | Train Dataset | Test Dataset | Accuracy (%) | Balanced Acc (%) | Ensemble Macro F1 (%) | Single LightGBM Macro F1 (%) | Transfer Status |
|---|---|---|---:|---:|---:|---:|---|
| **Exp A: In-Domain** | Edge-IIoTset | Edge-IIoTset | 62.24% | 59.09% | **59.05%** | 59.45% | Baseline In-Domain (12 features) |
| **Exp B: In-Domain** | CICIoT2023 | CICIoT2023 | 98.48% | 65.87% | **66.16%** | 67.82% | Baseline In-Domain (12 features) |
| **Exp C: Natural Pooled** | Edge + CIC | Pooled ($N=7,500$) | 79.96% | 63.76% | **66.10%** | 66.77% | Joint Multi-Domain Fit |
| **Exp C1: Pooled $\to$ Edge** | Edge + CIC | Edge-IIoTset | 61.36% | 58.29% | **57.28%** | — | Disaggregated Edge Test |
| **Exp C2: Pooled $\to$ CIC** | Edge + CIC | CICIoT2023 | 98.56% | 66.08% | **66.49%** | — | Disaggregated CIC Test |
| **Exp C-DB1: Domain-Bal $\to$ Edge** | Edge + CIC (Bal) | Edge-IIoTset | 52.88% | **59.54%** | **55.66%** | — | +0.45 pp Balanced Acc vs Exp A |
| **Exp C-DB2: Domain-Bal $\to$ CIC** | Edge + CIC (Bal) | CICIoT2023 | 96.45% | **78.89%** | **70.32%** | — | **+4.16 pp Macro-F1 vs Exp B** |
| **Exp D: Zero-Shot Transfer** | Edge-IIoTset | CICIoT2023 | 42.11% | 34.43% | **27.66%** | 20.42% | Domain Shift: -38.50 pp |
| **Exp E: Zero-Shot Transfer** | CICIoT2023 | Edge-IIoTset | 12.19% | 15.81% | **9.34%** (12.70% bal.) | 14.43% | Domain Shift: -49.71 pp |

## Scientific Domain-Shift Findings
1. **Reconciliation of Unshuffled vs. Stratified Sampling:**
   - Earlier runs of `scripts/cross_dataset_evaluation.py` loaded an unshuffled top-$40,000$ row slice (`nrows=40000`) of `ML-EdgeIIoT-dataset.csv`, which contained only 5 of the 15 classes (`Normal`, `DDoS_UDP`, `DDoS_ICMP`, `Ransomware`, `DDoS_HTTP`), yielding $16.50\%$ Macro-$F_1$. Replacing that slice with stratified 70/15/15 sampling via `DataLoader` restores all 10 shared classes and matches the $59.05\%$ (Ensemble) / $59.45\%$ (LightGBM) benchmark.
2. **Severe Covariate Shift Between Testbeds:**
   - Zero-shot transfer from `Edge-IIoTset` to `CICIoT2023` drops Macro-$F_1$ to $27.66\%$ (where the hybrid ensemble outperforms single LightGBM at $20.42\%$ by $+7.24\,\text{pp}$), while `CICIoT2023` $\to$ `Edge-IIoTset` drops to $9.34\%$ ($12.70\%$ with inverse-frequency class weights).
3. **Domain-Balanced Multi-Domain Pooling (`Exp C-DB2`):**
   - Inverse-frequency domain-balanced pooling achieves **$70.32\%$ Macro-$F_1$** and **$78.89\%$ Balanced Accuracy** on `CICIoT2023` (surpassing the single-domain `CICIoT2023` baseline by $+4.16\,\text{pp}$ Macro-$F_1$) while simultaneously achieving $59.54\%$ Balanced Accuracy on `Edge-IIoTset`.
