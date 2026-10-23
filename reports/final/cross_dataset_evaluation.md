# Cross-Dataset Generalization & Transferability Study (Reconciled & Verified)

## 1. Study Objective & Reconciled Experimental Protocol
Evaluate the generalization capacity of intrusion detection models trained on one IoT network topology and deployed zero-shot or via multi-domain pooling onto another network topology.

- **Source Environment 1:** `Edge-IIoTset` (Industrial IoT Testbed: Modbus, MQTT, BACnet; 15 native classes mapped to 10 shared classes)
- **Source Environment 2:** `CICIoT2023` (Smart-Home IoT Testbed: 34 native classes mapped to 10 shared classes)
- **Harmonized Feature Space ($D=12$):** `tcp_flag_fin`, `tcp_flag_syn`, `tcp_flag_rst`, `tcp_flag_ack`, `proto_tcp`, `proto_udp`, `proto_icmp`, `proto_http`, `proto_dns`, `proto_mqtt`, `proto_arp`, `packet_payload_length`
- **Stratified Evaluation Protocol:** Stratified 70/15/15 split of $25,000$ samples per dataset ($N_{\text{train}}=17,500$, $N_{\text{val}}=3,750$, $N_{\text{test}}=3,750$ per domain; $7,500$ pooled test rows).
- **Forensic Reconciliation Note:** An earlier auxiliary script (`scripts/cross_dataset_evaluation.py`) used an unshuffled sequential slice (`nrows=40000`) of `ML-EdgeIIoT-dataset.csv`, which captured only 5 of the 15 classes because the raw CSV is ordered by attack type, artificially depressing Exp A to $16.50\%$ Macro-$F_1$. Fixing `scripts/cross_dataset_evaluation.py` to use stratified sampling restores all 10 shared classes and reconciles all cross-dataset benchmarks.

---

## 2. Verified Cross-Dataset Transferability Matrix (10 Shared Classes)

| Experiment | Training Domain | Evaluation Domain | Single LightGBM Macro-$F_1$ (%) | Hybrid Ensemble Accuracy (%) | Hybrid Ensemble Balanced Acc (%) | Hybrid Ensemble Macro-$F_1$ (%) | Hybrid Ensemble Weighted-$F_1$ (%) |
|---|---|---|---:|---:|---:|---:|---:|
| **Exp A (In-Domain Edge)** | Edge-IIoTset | Edge-IIoTset ($N=3,750$) | 59.45% | 62.24% | 59.09% | **59.05%** | 58.65% |
| **Exp B (In-Domain CIC)** | CICIoT2023 | CICIoT2023 ($N=3,750$) | 67.82% | 98.48% | 65.87% | **66.16%** | 98.42% |
| **Exp C (Natural Pooled)** | Edge + CIC | Pooled Test ($N=7,500$) | 66.77% | 79.96% | 63.76% | **66.10%** | 77.78% |
| **Exp C1 (Natural Pooled $\to$ Edge)** | Edge + CIC | Edge-IIoTset ($N=3,750$) | — | 61.36% | 58.29% | **57.28%** | 57.11% |
| **Exp C2 (Natural Pooled $\to$ CIC)** | Edge + CIC | CICIoT2023 ($N=3,750$) | — | 98.56% | 66.08% | **66.49%** | 98.51% |
| **Exp C-DB1 (Domain-Balanced $\to$ Edge)** | Edge + CIC (Balanced) | Edge-IIoTset ($N=3,750$) | — | 52.88% | **59.54%** | **55.66%** | 55.20% |
| **Exp C-DB2 (Domain-Balanced $\to$ CIC)** | Edge + CIC (Balanced) | CICIoT2023 ($N=3,750$) | — | 96.45% | **78.89%** | **70.32%** | 97.14% |
| **Exp D (Zero-Shot Edge $\to$ CIC)** | Edge-IIoTset | CICIoT2023 ($N=3,750$) | 20.42% | 42.11% | 34.43% | **27.66%** | 45.24% |
| **Exp E (Zero-Shot CIC $\to$ Edge)** | CICIoT2023 | Edge-IIoTset ($N=3,750$) | 14.43% | 12.19% | 15.81% | **9.34%** (12.70% bal.) | 8.21% |

---

## 3. Scientific Findings & Domain Shift Analysis

1. **Quantifying the Representation Compression Cost ($97.73\% \to 59.05\%$):**
   - Projecting `Edge-IIoTset` from its full $K=33$ native protocol feature set (15 classes) down to the 12-feature cross-domain intersection (10 shared classes) strips fine-grained TCP sequence, acknowledgment, checksum, port, and URI length attributes. Consequently, within-domain `Edge-IIoTset` Macro-$F_1$ drops from $97.73\%$ (or $95.53\%$ at $K=25$) to $59.05\%$ ($59.45\%$ single LightGBM).
2. **Severe Zero-Shot Covariate Shift (Experiments D and E):**
   - Zero-shot transfer from `Edge-IIoTset` to `CICIoT2023` drops Macro-$F_1$ to $27.66\%$ (Ensemble) / $20.42\%$ (LightGBM), while `CICIoT2023` $\to$ `Edge-IIoTset` drops to $9.34\%$ ($12.70\%$ with class-balanced weighting; $14.43\%$ single LightGBM).
   - Root cause: `Edge-IIoTset` represents packet/row-level industrial telemetry, whereas `CICIoT2023` represents flow-aggregated consumer smart-home summaries with starkly different class priors (`DDoS_UDP`/`DDoS_ICMP`/`DDoS_TCP` comprise $>69\%$ of `CICIoT2023`).
3. **Positive Multi-Domain Transfer via Domain-Balanced Pooling (+4.16 pp on CICIoT2023):**
   - Domain-balanced multi-domain training (`Exp C-DB2`) elevates `CICIoT2023` Macro-$F_1$ from $66.16\%$ to **$70.32\%$** ($+4.16\,\text{pp}$) and Balanced Accuracy from $65.87\%$ to **$78.89\%$** ($+13.02\,\text{pp}$) by enriching minority web and spoofing classes (`Web_Attack`, `Spoofing_MITM`, `Vulnerability_Scan`) with industrial attack examples.
