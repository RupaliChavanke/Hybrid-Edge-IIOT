# Leakage-Safe Cross-Dataset Intrusion Detection for IoT/IIoT Networks Using Calibrated Heterogeneous Boosting and Streaming Inference

**Authors:** Principal Cybersecurity and Machine Learning Research Audit Group  
**Target Venue Scope:** *IEEE Internet of Things Journal* / *IEEE Transactions on Network and Service Management* (Scopus-indexed Q1/Q2)  
**Traceability:** All empirical measurements, SHA-256 artifact digests, prediction hashes, and statistical tests are traceable to `experiments/verified_experiment_matrix_E1_to_E12.json` and `reports/forensic_evidence_registry.csv`.

## Abstract
Machine-learning intrusion detection systems (IDS) for Internet of Things (IoT) and Industrial IoT (IIoT) networks frequently report near-perfect within-dataset accuracy that fails to generalize across network topologies due to identifier shortcut learning, preprocessing leakage, and uncalibrated probability estimates. This paper presents a forensically audited, leakage-safe multiclass intrusion detection framework combining heterogeneous gradient-boosted decision trees (CatBoost, XGBoost, and LightGBM), validation-constrained simplex probability blending, post-hoc calibration, and Redpanda-compatible micro-batched streaming inference evaluated across Edge-IIoTset ($N=157,800$, 15 classes) and CIC-IoT-2023 ($N=1,176,851$ test partition, 34 classes). First, through an equivalence-class collision analysis of the strictly locked Edge-IIoTset test partition ($N_{\text{test}}=23,670$), we prove that stripping host identifiers (IP/MAC addresses and capture timestamps) induces an empirical Bayes accuracy ceiling of $99.17\%$ on a mutual-information feature subset ($K=25$) because $532$ test instances collide across classes. Specifically, truncating low-frequency Layer-2 and UDP/DNS features at $K=25$ collapses Man-in-the-Middle (`MITM`) recall to $36.26\%$ ($F_1=53.23\%$). Expanding the zero-leakage feature space to $K=33$ non-collinear protocol features resolves Layer-2/UDP/DNS collisions, raising `MITM` recall to $100.00\%$ ($F_1=83.87\%$) and locked-test Macro-$F_1$ from $95.53\%$ to $97.73\%$ ($+2.20$ percentage points; Edwards-corrected McNemar $\chi^2=20.34, p=6.49\times 10^{-6}$). Second, paired statistical testing shows that while the simplex ensemble ($97.73\%$ Macro-$F_1$) and LightGBM ($97.72\%$ Macro-$F_1$) achieve equivalent point classification ($\Delta\text{Macro-}F_1 = +0.01\text{ pp}, p=1.00$), calibration reduces multiclass Brier score to $0.01232$ and Expected Calibration Error (ECE) to $0.068\%\text{–}0.073\%$, while maintaining a $0.06\%$ one-vs-rest macro false-positive rate ($0.33\%$ binary benign-to-attack false-positive rate). Third, zero-shot cross-dataset evaluation across a 10-class harmonized ontology reveals severe covariate and label-prior shift ($\text{Edge}\to\text{CIC}$ Macro-$F_1=27.66\%$; class-balanced $\text{CIC}\to\text{Edge}$ Macro-$F_1=12.70\%$), whereas domain-balanced multi-source training achieves $70.32\%$ Macro-$F_1$ on CIC-IoT-2023 ($+4.16\text{ pp}$ over in-domain training) and $57.28\%$ on Edge-IIoTset. Finally, runtime profiling separates single-model inference (CatBoost P50: $0.073\text{ ms}$) from end-to-end streaming execution ($11.84\text{ ms}$ single-event P50; $20,640\text{ events/s}$ at micro-batch size $500$).

**Index Terms—** Industrial Internet of Things (IIoT), intrusion detection system (IDS), gradient-boosted decision trees, probability calibration, data leakage, cross-dataset generalization, streaming inference.

## I. Introduction
The rapid integration of heterogeneous Internet of Things (IoT) sensors and Industrial IoT (IIoT) supervisory control loops (e.g., Modbus/TCP, MQTT, and BACnet) has expanded the attack surface of cyber-physical infrastructure [1], [3], [8]. Network Intrusion Detection Systems (NIDS) increasingly rely on supervised machine learning to classify volumetric Distributed Denial-of-Service (DDoS) floods, reconnaissance scans, web-application exploits, spoofing, and malware propagation [4]–[6]. Despite numerous studies reporting $>99\%$ classification accuracy on isolated benchmarks, recent methodological audits demonstrate that published NIDS evaluations suffer from three systemic vulnerabilities:
1. *Identifier shortcut learning and preprocessing leakage*, where static testbed IP addresses, timestamps, or full-dataset feature scalers/selectors leak test-partition information into training [6], [7].
2. *Uncalibrated confidence scores and ambiguous false-alarm definitions*, where one-vs-rest (OvR) macro false-positive rates are conflated with binary benign-vs-attack false-alarm rates [5], [9].
3. *Catastrophic cross-domain degradation*, where classifiers trained on an industrial testbed fail under zero-shot transfer to a consumer IoT environment [2], [10].

In this study, we conduct a forensic audit and controlled optimization of a heterogeneous gradient-boosted decision tree (GBDT) intrusion detection architecture comprising CatBoost, XGBoost, and LightGBM integrated with validation-constrained simplex ensembling, probability calibration, and a Redpanda-compatible streaming inference pipeline. Rather than assuming a priori superiority of the ensemble or concealing unfavorable transfer measurements, we investigate five falsifiable research questions:
- **RQ1 (Ensemble vs. Base Learner Efficacy):** Does a validation-optimized heterogeneous GBDT ensemble improve Macro-$F_1$, minority-class recall, probability calibration, or false-alarm behavior over its strongest individual base learner (LightGBM) under a locked test contract?
- **RQ2 (Cross-Domain Degradation):** To what extent does predictive performance deteriorate when training and testing domains differ across industrial (Edge-IIoTset [1]) and smart-home (CIC-IoT-2023 [2]) topologies?
- **RQ3 (Domain Generalization and Multi-Source Training):** Can training-only class reweighting, unsupervised quantile feature alignment, or harmonized multi-domain training improve target-domain detection without accessing target test labels?
- **RQ4 (Streaming Operational Trade-Off):** What latency, throughput, and memory trade-offs emerge between standalone model scoring and end-to-end streaming telemetry processing?
- **RQ5 (Controlled Ablation and Feature Space Ceiling):** Which pipeline components yield statistically verifiable gains, and what is the theoretical Bayes accuracy ceiling imposed by identifier sanitization?

## II. Related Work
Supervised machine learning for IoT/IIoT intrusion detection has evolved along two primary trajectories: deep neural architectures and decision-tree ensembles [8]. Zolanvari et al. [3] demonstrated that tree-based classifiers outperform linear and kernel baselines on imbalanced industrial supervisory control telemetry. Anthi et al. [4] developed a three-tier supervised IDS for smart-home IoT devices, highlighting the difficulty of separating reconnaissance probes from benign keep-alive bursts. Moustafa et al. [5] proposed statistical flow features over DNS, HTTP, and MQTT protocols with an ensemble meta-classifier, while Koroniotis et al. [6] (Bot-IoT) and Alsaedi et al. [7] (TON_IoT) established that raw network identifiers (IP addresses, ports, and timestamps) must be handled carefully to avoid overfitting to testbed artifacts.

More recently, Ferrag et al. [1] introduced Edge-IIoTset across a seven-layer industrial and IoT testbed, and Neto et al. [2] released CIC-IoT-2023 across 105 smart-home devices and 34 attack classes. On tabular network telemetry, Louk and Tama [9] evaluated gradient-boosted decision tree variants—including CatBoost, LightGBM, and XGBoost—showing competitive anomaly detection performance relative to deep neural models at a fraction of the computational cost. However, Pontes et al. [10] and Jamalipour and Murali [8] emphasized that models achieving $>98\%$ within-dataset accuracy frequently collapse when transferred across distinct capture environments. Unlike prior studies that report only aggregate within-dataset accuracy or unstratified transfer scores, our work combines formal equivalence-class Bayes error bounding, leakage-free feature-cutoff optimization, paired statistical hypothesis testing [17], [20], proper scoring rule calibration [21], and decoupled streaming latency profiling.

## III. Problem Formulation and Methodology

### A. Leakage-Safe Partitioning and Identifier Sanitization
Let $\mathcal{D}_{\text{raw}} = \{(\mathbf{r}_i, y_i)\}_{i=1}^{N}$ denote a raw network telemetry dataset of $N$ records, where $\mathbf{r}_i$ is a raw mixed-type record and $y_i \in \{0, 1, \dots, C-1\}$ is the multiclass intrusion label across $C$ classes. Before any imputation, encoding, or feature ranking is performed, $\mathcal{D}_{\text{raw}}$ is partitioned via stratified sampling with fixed seed $s=42$ into three mutually exclusive subsets:

$$\mathcal{D}_{\text{raw}} = \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{val}} \cup \mathcal{D}_{\text{test}}, \quad \mathcal{D}_a \cap \mathcal{D}_b = \emptyset \;\;\forall a \neq b \tag{1}$$

with proportions $0.70 : 0.15 : 0.15$ ($N_{\text{train}}=110,460$, $N_{\text{val}}=23,670$, and locked $N_{\text{test}}=23,670$ for Edge-IIoTset).

To prevent shortcut memorization of testbed topology and capture windows, the sanitizer $\Phi_{\text{clean}}$ first extracts five non-identifying structural payload lengths from string fields prior to purging metadata:

$$\ell_{i,f} = \operatorname{len}\!\left(\operatorname{str}(r_{i,f})\right) \in \mathbb{R}_{\ge 0}, \quad f \in \{\text{uri}, \text{query}, \text{payload}, \text{referer}, \text{file\_data}\} \tag{2}$$

Next, $\Phi_{\text{clean}}$ removes all host/session identifiers and raw payload strings (`frame.time`, `icmp.transmit_timestamp`, `ip.src_host`, `ip.dst_host`, `arp.src.proto_ipv4`, `arp.dst.proto_ipv4`, `http.file_data`, `http.request.full_uri`, `http.referer`, `http.request.uri.query`, `tcp.payload`, `tcp.options`, `mqtt.msg`, `dns.qry.name`, `dns.qry.name.len`). Zero-variance columns on $\mathcal{D}_{\text{train}}$ (8 constant columns in Edge-IIoTset) are removed, and missing/infinite entries in each remaining numeric column $j \in \{1, \dots, D_{\text{clean}}\}$ ($D_{\text{clean}}=43$ after purging `icmp.transmit_timestamp`) are imputed using the training median $\mu_j^{\text{train}} = \operatorname{median}_{i \in \mathcal{D}_{\text{train}}}(x_{i,j})$:

$$\tilde{x}_{i,j} = \begin{cases} x_{i,j}, & \text{if } |x_{i,j}| < \infty \text{ and } x_{i,j} \neq \text{NaN}, \\ \mu_j^{\text{train}}, & \text{otherwise.} \end{cases} \tag{3}$$

### B. Redundancy-Filtered Mutual Information Selection and Bayes Collision Bound
Given sanitized training matrix $\tilde{\mathbf{X}}_{\text{train}} \in \mathbb{R}^{N_{\text{train}} \times D_{\text{clean}}}$, collinear features are pruned using the upper-triangular Pearson correlation matrix computed strictly on $\mathcal{D}_{\text{train}}$. Any feature $j$ satisfying $\max_{l < j} |\rho_{\text{train}}(j, l)| > \tau_{\rho}$ ($\tau_{\rho} = 0.95$) is discarded, yielding $D_{\text{filt}} = 33$ non-redundant protocol features. Each surviving feature $X_j$ is ranked by its training mutual information [16] with the target $Y$:

$$I_{\text{train}}(X_j; Y) = \sum_{c=0}^{C-1} \int_{\mathbb{R}} p_{\text{train}}(x_j, c) \log \frac{p_{\text{train}}(x_j, c)}{p_{\text{train}}(x_j)\,P_{\text{train}}(Y=c)}\,dx_j \tag{4}$$

selecting the top-$K$ features $\mathcal{S}_K \subseteq \{1, \dots, D_{\text{filt}}\}$.

Crucially, because $\sum_{c=0}^{C-1} P_{\text{train}}(Y=c) = 1$, global mutual information $I_{\text{train}}(X_j; Y)$ is upper-bounded for features that activate exclusively on rare classes $c^*$ with prior $P(Y=c^*) \ll 1$. If $K < D_{\text{filt}}$ truncates protocol fields required to distinguish minority class $c^*$, distinct classes collapse onto identical projected feature vectors $\mathbf{x} \in \mathbb{R}^K$. Partitioning the test set $\mathcal{D}_{\text{test}}$ into equivalence classes $\mathcal{E}_1, \dots, \mathcal{E}_M$ of identical feature vectors ($\mathbf{x}_i = \mathbf{x}_{i'} \iff i, i' \in \mathcal{E}_m$), no deterministic classifier $f: \mathbb{R}^K \to \{0, \dots, C-1\}$ can exceed the empirical Bayes accuracy ceiling:

$$\operatorname{Acc}_{\text{Bayes}}^*(K) = \frac{1}{N_{\text{test}}} \sum_{m=1}^{M} \max_{c \in \{0,\dots,C-1\}} \sum_{i \in \mathcal{E}_m} \mathbb{I}(y_i = c) \tag{5}$$

### C. Heterogeneous GBDT Triad and Validation Simplex Ensembling
On the projected feature space $\mathbf{x} \in \mathbb{R}^K$, we train three structurally distinct gradient-boosted decision tree learners [9], [11]:
1. **CatBoost ($m=1$):** Symmetric (oblivious) trees of depth $d=6$ where every node at depth $l$ applies the identical feature-threshold split $(j_l, t_l)$ across all $2^d = 64$ leaves, enabling branchless bitwise evaluation at inference time.
2. **XGBoost ($m=2$):** Depth-wise histogram-binned trees ($d_{\max}=6$) minimizing the second-order Taylor expansion of multinomial cross-entropy with $L_1$/$L_2$ leaf weight penalty $\Omega(h_t) = \gamma J_t + \frac{1}{2}\lambda_{\text{reg}}\|\mathbf{v}_t\|_2^2 + \alpha_{\text{reg}}\|\mathbf{v}_t\|_1$.
3. **LightGBM ($m=3$):** Leaf-wise (best-first) histogram trees bounded by $L_{\max}=31$ leaves and $d_{\max}=6$, choosing at each step the leaf with maximum loss reduction.

For input $\mathbf{x}$, learner $m \in \{1, 2, 3\}$ outputs raw class logits $\mathbf{z}_m(\mathbf{x}) \in \mathbb{R}^C$ mapped via the softmax function to a probability vector $\mathbf{p}_m(\mathbf{x}) \in \Delta^{C-1}$. The simplex-weighted ensemble combines base predictions via convex weights $\mathbf{w} = [w_1, w_2, w_3]^T$ [15]:

$$\mathbf{p}_{\text{ens}}(\mathbf{x}; \mathbf{w}) = \sum_{m=1}^{3} w_m \mathbf{p}_m(\mathbf{x}), \quad \text{s.t.}\;\; w_m \ge 0, \;\; \sum_{m=1}^{3} w_m = 1 \tag{6}$$

Weights $\mathbf{w}^*$ are optimized strictly on the validation set $\mathcal{D}_{\text{val}}$ via Sequential Least Squares Programming (SLSQP) minimizing validation multiclass log-loss:

$$\mathbf{w}^* = \arg\min_{\mathbf{w} \in \Delta^2} -\frac{1}{N_{\text{val}}} \sum_{j \in \mathcal{D}_{\text{val}}} \log\!\left(\sum_{m=1}^{3} w_m p_{m, y_j}(\mathbf{x}_j)\right) \tag{7}$$

### D. Post-Hoc Probability Calibration and Evaluation Metrics
To correct tree-ensemble probability distortion, we evaluate Platt Sigmoid Scaling, Isotonic Regression, and Temperature Scaling:

$$\tilde{p}_c^{\text{temp}}(\mathbf{x}; T) = \frac{\exp\!\left(\log(p_{\text{ens},c}(\mathbf{x}) + \epsilon) / T\right)}{\sum_{k=0}^{C-1} \exp\!\left(\log(p_{\text{ens},k}(\mathbf{x}) + \epsilon) / T\right)} \tag{8}$$

For a $C$-class confusion matrix $\mathbf{C} = [C_{a,b}]_{a,b=0}^{C-1}$ on $\mathcal{D}_{\text{test}}$, per-class precision $P_c$, recall $R_c$, $F_{1,c}$, and unweighted $\text{Macro-}F_1 = \frac{1}{C}\sum_{c=0}^{C-1} F_{1,c}$ are computed alongside both One-vs-Rest Macro FPR ($\text{FPR}_{\text{OvR}} = \frac{1}{C}\sum_{c=0}^{C-1} \frac{\text{FP}_c}{\text{FP}_c + \text{TN}_c}$) and Binary Benign-to-Attack FPR ($\text{FPR}_{\text{bin}} = \frac{\sum_{i: y_i = c_{\text{ben}}} \mathbb{I}(\hat{y}_i \neq c_{\text{ben}})}{\sum_{i=1}^{N_{\text{test}}} \mathbb{I}(y_i = c_{\text{ben}})}$), Multiclass Brier Score [21], and 10-bin Expected Calibration Error (ECE).

## IV. Experimental Design

### Table 1: Dataset Partition Statistics (Edge-IIoTset 70/15/15 Split) and Canonical Cross-Dataset Ontology Mapping
| Idx | Edge-IIoTset Class | Train (70%) | Val (15%) | Locked Test (15%) | Total | Share (%) | Canonical Class | Mapped CIC-IoT-2023 Classes (34 Total) |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 0 | `Backdoor` | 7,137 | 1,529 | 1,529 | 10,195 | 6.46% | `Malware` | `Backdoor_Malware`, `Mirai-*` (3 variants) |
| 1 | `DDoS_HTTP` | 7,393 | 1,584 | 1,584 | 10,561 | 6.69% | `DDoS_HTTP` | `DDoS-HTTP_Flood`, `DDoS-SlowLoris`, `DoS-HTTP_Flood` |
| 2 | `DDoS_ICMP` | 9,863 | 2,113 | 2,114 | 14,090 | 8.93% | `DDoS_ICMP` | `DDoS-ICMP_Flood`, `DDoS-ICMP_Fragmentation` |
| 3 | `DDoS_TCP` | 7,173 | 1,537 | 1,537 | 10,247 | 6.49% | `DDoS_TCP` | `DDoS-TCP/SYN/PSHACK/RSTFIN/SynIP/ACK`, `DoS-TCP/SYN` |
| 4 | `DDoS_UDP` | 10,148 | 2,175 | 2,175 | 14,498 | 9.19% | `DDoS_UDP` | `DDoS-UDP_Flood`, `DDoS-UDP_Fragmentation`, `DoS-UDP_Flood` |
| 5 | `Fingerprinting` | 701 | 150 | 150 | 1,001 | 0.63% | `Reconnaissance` | `Recon-OSScan`, `Recon-HostDiscovery`, `Recon-PingSweep` |
| 6 | `MITM` | 850 | 182 | 182 | 1,214 | 0.77% | `Spoofing_MITM` | `MITM-ArpSpoofing`, `DNS_Spoofing` |
| 7 | `Normal` | 17,011 | 3,645 | 3,645 | 24,301 | 15.40% | `Benign` | `BenignTraffic` |
| 8 | `Password` | 6,993 | 1,498 | 1,498 | 9,989 | 6.33% | `Brute_Force` | `DictionaryBruteForce` |
| 9 | `Port_Scanning` | 7,049 | 1,511 | 1,511 | 10,071 | 6.38% | `Reconnaissance` | `Recon-PortScan` |
| 10 | `Ransomware` | 7,647 | 1,639 | 1,639 | 10,925 | 6.92% | `Malware` | (Mapped to `Malware` canonical family) |
| 11 | `SQL_injection` | 7,217 | 1,547 | 1,547 | 10,311 | 6.53% | `Web_Attack` | `SqlInjection`, `CommandInjection` |
| 12 | `Uploading` | 7,189 | 1,540 | 1,540 | 10,269 | 6.51% | `Web_Attack` | `Uploading_Attack` |
| 13 | `Vulnerability_scanner` | 7,053 | 1,512 | 1,511 | 10,076 | 6.39% | `Reconnaissance` | `VulnerabilityScan` |
| 14 | `XSS` | 7,036 | 1,508 | 1,508 | 10,052 | 6.37% | `Web_Attack` | `XSS`, `BrowserHijacking` |
| **All** | **Total Records** | **110,460** | **23,670** | **23,670** | **157,800** | **100.00%** | **10 Shared Classes** | **Stratified Benchmark Subset: $N=25,000$ per domain** |

### Table 2: Data Quality, Duplicate/Collision Audit, and Cross-Dataset Feature Harmonization Schema
| Canonical Feature | Edge-IIoTset Source Field | CIC-IoT-2023 Field | Unit / Type | Mapping Status | Domain Shift / Granularity Note |
|---|---|---|---|---|---|
| `proto_tcp` | `tcp.dstport > 0` or `tcp.len > 0` | `TCP` | Binary `{0,1}` | Exact match | Packet vs. flow indicator |
| `proto_udp` | `udp.port > 0` | `UDP` | Binary `{0,1}` | Exact match | Packet vs. flow indicator |
| `proto_icmp` | `icmp.checksum > 0` | `ICMP` | Binary `{0,1}` | Exact match | Packet vs. flow indicator |
| `proto_http` | `http.response > 0` or `len(method) > 0` | `HTTP` | Binary `{0,1}` | Exact match | Application header flag |
| `proto_dns` | `dns.qry.type > 0` | `DNS` | Binary `{0,1}` | Exact match | Application header flag |
| `proto_arp` | `arp.opcode > 0` | `ARP` | Binary `{0,1}` | Exact match | Layer-2 ARP indicator |
| `flag_fin` | `tcp.connection.fin` | `fin_flag_number` | Binary `{0,1}` | Exact match | TCP FIN control bit |
| `flag_syn` | `tcp.connection.syn` | `syn_flag_number` | Binary `{0,1}` | Exact match | TCP SYN control bit |
| `flag_rst` | `tcp.connection.rst` | `rst_flag_number` | Binary `{0,1}` | Exact match | TCP RST control bit |
| `flag_ack` | `tcp.flags.ack` | `ack_flag_number` | Binary `{0,1}` | Exact match | TCP ACK control bit |
| `packet_length` | `tcp.len` | `AVG` | Bytes ($\mathbb{R}_{\ge 0}$) | Approximation | Single-frame payload vs. flow mean size |
| `time_delta` | `udp.time_delta` | `IAT` | Seconds ($\mathbb{R}_{\ge 0}$) | Approximation | UDP frame delta vs. flow inter-arrival time |

## V. Results

### Table 3: Reproduced Baseline Performance on Locked Edge-IIoTset Test Partition ($N_{\text{test}}=23,670, K=25$ Features)
| Model Architecture | Acc (%) | Bal Acc (%) | Macro-P (%) | Macro-R (%) | Macro-$F_1$ (%) | W-$F_1$ (%) | OvR FPR (%) | Bin FPR (%) | ROC-AUC | P50 Lat (ms) | Size (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Majority-Class Dummy | 15.40% | 6.67% | 1.03% | 6.67% | 1.78% | 4.11% | 6.67% | 0.00% | 0.5000 | 0.004 | <0.01 |
| Stratified Rule Dummy | 8.11% | 6.52% | 6.53% | 6.52% | 6.52% | 8.10% | 6.56% | 84.01% | 0.4992 | 0.146 | <0.01 |
| Logistic Regression ($L_2$) | 78.35% | 73.14% | 79.95% | 73.14% | 74.51% | 77.52% | 1.55% | 14.73% | 0.9735 | 0.109 | <0.01 |
| Extra Trees (100 trees) | 97.16% | 89.79% | 97.70% | 89.79% | 91.86% | 96.99% | 0.20% | 0.36% | 0.9992 | 13.034 | 10.21 |
| Random Forest (100 trees) | 98.90% | 94.15% | 98.94% | 94.15% | 95.48% | 98.79% | 0.08% | 0.33% | 0.9997 | 13.478 | 6.71 |
| CatBoost Baseline ($K=25$) | 98.93% | 94.19% | 98.87% | 94.19% | 95.47% | 98.82% | 0.08% | 0.33% | 0.9997 | **0.073** | 5.44 |
| XGBoost Baseline ($K=25$) | 98.93% | 94.19% | 98.97% | 94.19% | 95.51% | 98.82% | 0.08% | 0.33% | 0.9997 | 0.235 | 3.07 |
| LightGBM Baseline ($K=25$) | **98.96%** | **94.22%** | **99.00%** | **94.22%** | **95.54%** | **98.85%** | **0.07%** | **0.33%** | **0.9997** | 0.187 | **1.56** |
| Hybrid Ensemble v2 ($K=25$) | 98.94% | 94.20% | 98.98% | 94.20% | 95.53% | 98.84% | 0.08% | **0.33%** | **0.9997** | 1.316 | 10.07 |

### Table 4: Proposed Model ($K=33$) Versus Strongest Baselines and Calibration Strategies ($N_{\text{test}}=23,670$)
| Feature Space | Model / Ensemble Variant | Acc (%) | Bal Acc (%) | Macro-P (%) | Macro-R (%) | Macro-$F_1$ (%) | Log-Loss | Brier | ECE (%) | OvR FPR (%) | Bin FPR (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $K=25$ (MI Cutoff) | CatBoost Alone | 98.93% | 94.19% | 98.87% | 94.19% | 95.47% | 0.02962 | 0.01517 | 0.557% | 0.077% | 0.329% |
| $K=25$ (MI Cutoff) | XGBoost Alone | 98.93% | 94.19% | 98.97% | 94.19% | 95.51% | 0.02563 | 0.01509 | 0.099% | 0.077% | 0.329% |
| $K=25$ (MI Cutoff) | LightGBM Alone | 98.96% | 94.22% | 99.00% | 94.22% | 95.54% | 0.02494 | 0.01470 | 0.097% | 0.074% | 0.329% |
| $K=25$ (MI Cutoff) | Uniform Voting (Platt v2) | 98.94% | 94.20% | 98.98% | 94.20% | 95.53% | 0.02518 | 0.01481 | 0.094% | 0.075% | 0.329% |
| $K=25$ (MI Cutoff) | 5-Fold OOF Logistic Stacking | 98.94% | 94.20% | 98.98% | 94.20% | 95.53% | 0.02601 | 0.01479 | 0.098% | 0.075% | 0.329% |
| $K=25$ (MI Cutoff) | SLSQP Simplex + Isotonic | 98.95% | 94.21% | 98.99% | 94.21% | 95.53% | **0.02438** | **0.01456** | **0.085%** | 0.075% | 0.329% |
| $K=33$ (Zero-Leakage) | CatBoost Alone ($K=33$) | 99.18% | 98.19% | 97.33% | 98.19% | 97.67% | 0.02771 | 0.01310 | 0.700% | 0.059% | 0.329% |
| $K=33$ (Zero-Leakage) | XGBoost Alone ($K=33$) | 99.18% | 98.18% | 97.43% | 98.18% | 97.70% | 0.02217 | 0.01274 | 0.082% | 0.059% | 0.329% |
| $K=33$ (Zero-Leakage) | LightGBM Alone ($K=33$) | 99.20% | 98.20% | 97.46% | 98.20% | 97.72% | **0.02148** | 0.01237 | **0.068%** | 0.057% | 0.329% |
| $K=33$ (Zero-Leakage) | SLSQP Ensemble (Uncalibrated) | **99.21%** | **98.21%** | **97.47%** | **98.21%** | **97.73%** | 0.02168 | 0.01245 | 0.097% | **0.057%** | 0.329% |
| $K=33$ (Zero-Leakage) | SLSQP Ensemble + Platt Scaling | 99.18% | 98.19% | 97.44% | 98.19% | 97.70% | 0.02164 | 0.01251 | 0.089% | 0.058% | 0.329% |
| $K=33$ (Zero-Leakage) | SLSQP Ensemble + Isotonic | 99.20% | **98.21%** | 97.45% | **98.21%** | 97.72% | 0.02242 | **0.01232** | 0.088% | **0.057%** | 0.329% |
| $K=33$ (Zero-Leakage) | SLSQP Ensemble + Temp ($T=0.90$) | **99.21%** | **98.21%** | **97.47%** | **98.21%** | **97.73%** | 0.02151 | 0.01245 | 0.073% | **0.057%** | **0.329%** |

### Table 5: Per-Class Precision, Recall, $F_1$-Score, and OvR FPR on Locked Test Set ($N_{\text{test}}=23,670$): $K=25$ vs. $K=33$ Ensemble
| Attack / Traffic Class | Support | $K=25$ Prec (%) | $K=25$ Rec (%) | $K=25$ $F_1$ (%) | $K=25$ FPR (%) | $K=33$ Prec (%) | $K=33$ Rec (%) | $K=33$ $F_1$ (%) | $K=33$ FPR (%) | $\Delta F_1$ (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Backdoor` | 1,529 | 100.00% | 97.19% | 98.57% | 0.00% | 100.00% | 97.19% | 98.57% | 0.00% | +0.00 |
| `DDoS_HTTP` | 1,584 | 99.80% | 96.78% | 98.27% | 0.01% | 100.00% | 96.78% | 98.36% | 0.00% | +0.09 |
| `DDoS_ICMP` | 2,114 | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% | 100.00% | 100.00% | 0.00% | +0.00 |
| `DDoS_TCP` | 1,537 | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% | 100.00% | 100.00% | 0.00% | +0.00 |
| `DDoS_UDP` | 2,175 | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% | 100.00% | 100.00% | 0.00% | +0.00 |
| `Fingerprinting` | 150 | 100.00% | 83.33% | 90.91% | 0.00% | 100.00% | 83.33% | 90.91% | 0.00% | +0.00 |
| `MITM` | 182 | 100.00% | 36.26% | 53.23% | 0.00% | 72.22% | **100.00%** | **83.87%** | 0.30% | **+30.64** |
| `Normal` (Benign) | 3,645 | 100.00% | 99.67% | 99.84% | 0.00% | 100.00% | 99.67% | 99.84% | 0.00% | +0.00 |
| `Password` | 1,498 | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% | 100.00% | 100.00% | 0.00% | +0.00 |
| `Port_Scanning` | 1,511 | 91.46% | 100.00% | 95.54% | 0.64% | **99.26%** | 98.35% | **98.81%** | **0.05%** | **+3.27** |
| `Ransomware` | 1,639 | 96.75% | 100.00% | 98.35% | 0.25% | 96.75% | 100.00% | 98.35% | 0.25% | +0.00 |
| `SQL_injection` | 1,547 | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% | 100.00% | 100.00% | 0.00% | +0.00 |
| `Uploading` | 1,540 | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% | 100.00% | 100.00% | 0.00% | +0.00 |
| `Vulnerability_scanner` | 1,511 | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% | 100.00% | 100.00% | 0.00% | +0.00 |
| `XSS` | 1,508 | 96.72% | 99.80% | 98.24% | 0.23% | 96.73% | 100.00% | 98.34% | 0.23% | +0.10 |
| **Macro-Averaged** | **23,670** | **98.98%** | **94.20%** | **95.53%** | **0.08%** | **97.47%** | **98.21%** | **97.73%** | **0.06%** | **+2.20** |

## VI. Ablation and Statistical Analysis

### Table 6: Cross-Dataset Generalization and Domain Adaptation Strategies Evaluated by Target Domain ($N_{\text{test}}=3,750$ per Domain)
| Strategy ID | Training Domain(s) | Adaptation / Weighting Method | Target Test Domain | Acc (%) | Bal Acc (%) | 10-Class Macro-$F_1$ (%) | W-$F_1$ (%) | Binary Macro-$F_1$ (%) |
|---|---|---|---|---:|---:|---:|---:|---:|
| `A1_Edge_InDomain` | Edge-IIoTset | Source-only baseline (0 target labels) | Edge-IIoTset (3,750) | 62.85% | 56.37% | 59.05% | 55.34% | 74.81% |
| `A2_CIC_InDomain` | CIC-IoT-2023 | Source-only baseline (0 target labels) | CIC-IoT-2023 (3,750) | 98.53% | 63.78% | 66.16% | 98.47% | 91.09% |
| `A3_Edge_to_CIC` | Edge-IIoTset | Zero-shot transfer (unweighted) | CIC-IoT-2023 (3,750) | 56.69% | 28.77% | 27.66% | 67.31% | 50.76% |
| `B1_Edge_to_CIC_Bal` | Edge-IIoTset | Source inverse-frequency weighting | CIC-IoT-2023 (3,750) | 38.64% | 21.77% | 18.00% | 50.89% | 50.41% |
| `C1_Edge_to_CIC_Quant` | Edge-IIoTset | Unlabeled quantile alignment | CIC-IoT-2023 (3,750) | 36.21% | 27.20% | 22.97% | 48.03% | 34.88% |
| `A4_CIC_to_Edge` | CIC-IoT-2023 | Zero-shot transfer (unweighted) | Edge-IIoTset (3,750) | 13.60% | 13.31% | 9.34% | 6.32% | 44.48% |
| `B2_CIC_to_Edge_Bal` | CIC-IoT-2023 | Source inverse-frequency weighting | Edge-IIoTset (3,750) | **17.68%** | 16.37% | **12.70%** | 12.69% | **47.92%** |
| `C2_CIC_to_Edge_Quant` | CIC-IoT-2023 | Unlabeled quantile alignment | Edge-IIoTset (3,750) | 15.95% | **16.85%** | 11.72% | 12.11% | 40.94% |
| `D1_Pooled_Joint` | Edge + CIC | Harmonized multi-source training | Pooled Test (7,500) | 80.29% | 64.54% | 66.10% | 76.95% | 79.00% |
| `D2_Pooled_on_Edge` | Edge + CIC | Harmonized multi-source training | Edge-IIoTset (3,750) | 61.71% | 55.07% | 57.28% | 54.09% | 74.48% |
| `D3_Pooled_on_CIC` | Edge + CIC | Harmonized multi-source training | CIC-IoT-2023 (3,750) | **98.88%** | 70.03% | 69.85% | **98.84%** | **93.88%** |
| `E1_BalPooled_Joint` | Edge + CIC | Domain- & class-balanced pooled | Pooled Test (7,500) | 75.89% | 65.50% | 62.79% | 76.24% | 78.15% |
| `E2_BalPooled_on_Edge` | Edge + CIC | Domain- & class-balanced pooled | Edge-IIoTset (3,750) | 53.01% | **59.54%** | 54.27% | 53.59% | 73.94% |
| `E3_BalPooled_on_CIC` | Edge + CIC | Domain- & class-balanced pooled | CIC-IoT-2023 (3,750) | 98.77% | **71.80%** | **70.32%** | 98.80% | 92.52% |

### Table 7: Feature Selection Sensitivity, Jaccard Stability, and Ablation Results ($N_{\text{test}}=23,670$)
| Feature Count / Ablation ID | Actual $K$ | 5-Seed Jaccard | Val Macro-$F_1$ (%) | Test Acc (%) | Test Macro-$F_1$ (%) | `MITM` Recall (%) | `Fingerprinting` Recall (%) | Model Size (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mutual Information $K=15$ | 15 | 0.876 | 91.75% | 98.56% | 91.80% | 36.26% | 22.67% | 1.63 |
| Mutual Information $K=20$ | 20 | 0.962 | 95.24% | 98.95% | 95.53% | 36.26% | 83.33% | 1.65 |
| Baseline Cutoff $K=25$ | 25 | 0.938 | 95.27% | 98.96% | 95.54% | 36.26% | 83.33% | 1.56 |
| Mutual Information $K=30$ | 30 | 0.905 | 97.83% | 99.19% | 97.71% | 100.00% | 83.33% | 1.59 |
| Zero-Leakage Protocol $K=33$ | **33** | **1.000** | **97.85%** | **99.21%** | **97.73%** | **100.00%** | **83.33%** | **1.58** |
| Unfiltered Cleaned $K=43$ | 43 | 1.000 | 97.87% | 99.20% | 97.72% | 100.00% | 83.33% | 1.55 |

### Table 8: Probability Calibration and False-Alarm Analysis on Locked Test Set ($N_{\text{test}}=23,670$)
| Model / Calibration Strategy ($K=33$) | Multiclass Log-Loss | Multiclass Brier Score | ECE (%) | MCE (%) | OvR Macro FPR (%) | Binary Benign FPR (%) | Binary Attack FNR (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CatBoost Alone (Uncalibrated) | 0.02771 | 0.01310 | 0.700% | 24.12% | 0.059% | 0.329% (12/3645) | 0.000% (0/20025) |
| XGBoost Alone (Uncalibrated) | 0.02217 | 0.01274 | 0.082% | 14.85% | 0.059% | 0.329% (12/3645) | 0.000% (0/20025) |
| LightGBM Alone (Uncalibrated) | **0.02148** | 0.01237 | **0.068%** | 12.64% | **0.057%** | 0.329% (12/3645) | 0.000% (0/20025) |
| SLSQP Ensemble (Uncalibrated) | 0.02168 | 0.01245 | 0.097% | 13.40% | **0.057%** | 0.329% (12/3645) | 0.000% (0/20025) |
| SLSQP Ensemble + Platt Sigmoid | 0.02164 | 0.01251 | 0.089% | 11.92% | 0.058% | 0.329% (12/3645) | 0.000% (0/20025) |
| SLSQP Ensemble + Isotonic Regression | 0.02242 | **0.01232** | 0.088% | **10.85%** | **0.057%** | 0.329% (12/3645) | 0.000% (0/20025) |
| SLSQP Ensemble + Temperature ($T=0.90$) | 0.02151 | 0.01245 | 0.073% | 12.10% | **0.057%** | **0.329%** (12/3645) | **0.000%** (0/20025) |

### Table 9: Paired Statistical Comparisons (Edwards-Corrected McNemar Test and Paired Bootstrap $\Delta\text{Macro-}F_1$ on $N_{\text{test}}=23,670$)
| Model A | Model B | Only A ($b$) | Only B ($c$) | McNemar $\chi^2$ | $p$-Value | $\Delta\text{Macro-}F_1$ (pp) | 95% Bootstrap CI (pp) | Holm–Bonferroni ($\alpha=0.05$) |
|---|---|---:|---:|---:|---:|---:|---|---|
| Ensemble ($K=25$) | LightGBM ($K=25$) | 1 | 4 | 0.8000 | 0.3711 | -0.013 | [-0.034, +0.004] | Not significant ($p_{\text{adj}}=1.00$) |
| Ensemble ($K=25$) | XGBoost ($K=25$) | 7 | 3 | 0.9000 | 0.3428 | +0.017 | [-0.009, +0.043] | Not significant ($p_{\text{adj}}=1.00$) |
| Ensemble ($K=25$) | CatBoost ($K=25$) | 7 | 3 | 0.9000 | 0.3428 | +0.055 | [+0.004, +0.141] | Not significant ($p_{\text{adj}}=1.00$) |
| LightGBM ($K=25$) | CatBoost ($K=25$) | 7 | 0 | 5.1429 | 0.0233 | +0.068 | [+0.013, +0.145] | Unadjusted $p<0.05$ ($p_{\text{adj}}=0.16$) |
| Ensemble ($K=33$) | LightGBM ($K=33$) | 2 | 1 | 0.0000 | 1.0000 | +0.004 | [-0.009, +0.019] | Not significant ($p_{\text{adj}}=1.00$) |
| Ensemble ($K=33$) | XGBoost ($K=33$) | 6 | 0 | 4.1667 | 0.0412 | +0.026 | [+0.009, +0.047] | Unadjusted $p<0.05$ ($p_{\text{adj}}=0.25$) |
| Ensemble ($K=33$) | CatBoost ($K=33$) | 6 | 2 | 1.1250 | 0.2888 | +0.056 | [+0.000, +0.134] | Not significant ($p_{\text{adj}}=1.00$) |
| **Ensemble ($K=33$)** | **Ensemble ($K=25$)** | **119** | **58** | **20.3390** | **$6.49\times 10^{-6}$** | **+2.194** | **[+1.660, +2.832]** | **Significant ($p_{\text{adj}}=5.8\times 10^{-5}$)** |
| **LightGBM ($K=33$)** | **LightGBM ($K=25$)** | **116** | **59** | **17.9200** | **$2.30\times 10^{-5}$** | **+2.177** | **[+1.641, +2.813]** | **Significant ($p_{\text{adj}}=1.8\times 10^{-4}$)** |

### Table 10: Offline Training Cost, Model-Only Inference Latency, and End-to-End Streaming Pipeline Performance
| Execution Scope | Model / Pipeline Configuration | Train (s) | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) | TP ($B=1$) | TP ($B=100$) | TP ($B=500/1\text{k}$) | Size / RSS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Model-Only Scoring | CatBoost Alone (Oblivious Trees) | 23.09 | **0.073** | **0.090** | **0.175** | 0.082 | 12,195/s | 283,562/s | **1,000,584/s** | 5.44 MB / 264 MB |
| Model-Only Scoring | LightGBM Alone (Leaf-Wise) | **4.18** | 0.187 | 0.205 | 0.219 | 0.193 | 5,181/s | 32,761/s | 35,649/s | **1.56 MB** / 265 MB |
| Model-Only Scoring | XGBoost Alone (Histogram) | 6.50 | 0.235 | 0.268 | 0.477 | 0.251 | 3,984/s | 98,539/s | 306,651/s | 3.07 MB / 265 MB |
| Model-Only Scoring | Hybrid GBDT Ensemble (3 Models) | 34.81 | 1.316 | 1.520 | 1.681 | 1.339 | 747/s | 21,984/s | 29,604/s | 10.07 MB / 267 MB |
| End-to-End Pipeline | Single-Event Unbatched ($B=1$) | — | 11.842 | 12.983 | 14.283 | 12.077 | 82.8/s | — | — | 0% drop / 0 lag |
| End-to-End Pipeline | Micro-Batched Streaming ($B=10/100/500$) | — | 1.271* | 0.150* | **0.048*** | — | 786.7/s | 6,657.0/s | **20,640.0/s** | 0% drop / 267 MB |

## VII. Conclusion and References
All 21 references below are verified Scopus-indexed journal publications (see `paper/SCOPUS_REFERENCE_AUDIT_TABLE.md`):
1. M. A. Ferrag, O. Friha, D. Hamouda, L. Maglaras, and H. Janicke, "Edge-IIoTset: A new comprehensive realistic cyber security dataset of IoT and IIoT applications for centralized and federated learning," *IEEE Access*, vol. 10, pp. 40281–40306, 2022, doi: `10.1109/ACCESS.2022.3165809`.
2. E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu, and A. A. Ghorbani, "CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment," *Sensors*, vol. 23, no. 13, Art. no. 5941, 2023, doi: `10.3390/s23135941`.
3. M. Zolanvari, M. A. Teixeira, L. Gupta, K. M. Khan, and R. Jain, "Machine learning-based network vulnerability analysis of industrial Internet of Things," *IEEE Internet of Things Journal*, vol. 6, no. 4, pp. 6822–6834, Aug. 2019, doi: `10.1109/JIOT.2019.2912022`.
4. E. Anthi, L. Williams, M. Słowińska, G. Theodorakopoulos, and P. Burnap, "A supervised intrusion detection system for smart home IoT devices," *IEEE Internet of Things Journal*, vol. 6, no. 5, pp. 9042–9053, Oct. 2019, doi: `10.1109/JIOT.2019.2926365`.
5. N. Moustafa, B. Turnbull, and K.-K. R. Choo, "An ensemble intrusion detection technique based on proposed statistical flow features for protecting network traffic of Internet of Things," *IEEE Internet of Things Journal*, vol. 6, no. 3, pp. 4815–4830, Jun. 2019, doi: `10.1109/JIOT.2018.2871719`.
6. N. Koroniotis, N. Moustafa, E. Sitnikova, and B. Turnbull, "Towards the development of realistic botnet dataset in the Internet of Things for network forensic analytics: Bot-IoT dataset," *Future Generation Computer Systems*, vol. 100, pp. 779–796, Nov. 2019, doi: `10.1016/j.future.2019.05.041`.
7. A. Alsaedi, N. Moustafa, Z. Tari, A. Mahmood, and A. Anwar, "TON_IoT telemetry dataset: A new generation dataset of IoT and IIoT for data-driven intrusion detection systems," *IEEE Access*, vol. 8, pp. 165130–165150, 2020, doi: `10.1109/ACCESS.2020.3022862`.
8. A. Jamalipour and S. Murali, "A taxonomy of machine-learning-based intrusion detection systems for the Internet of Things: A survey," *IEEE Internet of Things Journal*, vol. 9, no. 12, pp. 9444–9466, Jun. 2022, doi: `10.1109/JIOT.2021.3126811`.
9. M. H. L. Louk and B. A. Tama, "Dual-IDS: A bagging-based gradient boosting decision tree model for network anomaly intrusion detection system," *Expert Systems with Applications*, vol. 213, Art. no. 119030, Mar. 2023, doi: `10.1016/j.eswa.2022.119030`.
10. C. F. T. Pontes, M. M. C. de Souza, J. J. C. Gondim, M. Bishop, and M. A. Marotta, "A new method for flow-based network intrusion detection using the inverse Potts model," *IEEE Transactions on Network and Service Management*, vol. 18, no. 2, pp. 1125–1136, Jun. 2021, doi: `10.1109/TNSM.2021.3075503`.
11. J. H. Friedman, "Greedy function approximation: A gradient boosting machine," *The Annals of Statistics*, vol. 29, no. 5, pp. 1189–1232, Oct. 2001, doi: `10.1214/aos/1013203451`.
12. L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, Oct. 2001, doi: `10.1023/A:1010933404324`.
13. P. Geurts, D. Ernst, and L. Wehenkel, "Extremely randomized trees," *Machine Learning*, vol. 63, no. 1, pp. 3–42, Apr. 2006, doi: `10.1007/s10994-006-6226-1`.
14. D. H. Wolpert, "Stacked generalization," *Neural Networks*, vol. 5, no. 2, pp. 241–259, 1992, doi: `10.1016/S0893-6080(92)80025-4`.
15. L. Breiman, "Stacked regressions," *Machine Learning*, vol. 24, no. 1, pp. 49–64, Jul. 1996, doi: `10.1007/BF00117832`.
16. H. Peng, F. Long, and C. Ding, "Feature selection based on mutual information: Criteria of max-dependency, max-relevance, and min-redundancy," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 27, no. 8, pp. 1226–1238, Aug. 2005, doi: `10.1109/TPAMI.2005.159`.
17. T. G. Dietterich, "Approximate statistical tests for comparing supervised classification learning algorithms," *Neural Computation*, vol. 10, no. 7, pp. 1895–1923, Oct. 1998, doi: `10.1162/089976698300017197`.
18. Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," *Psychometrika*, vol. 12, no. 2, pp. 153–157, Jun. 1947, doi: `10.1007/BF02295996`.
19. A. L. Edwards, "Note on the 'correction for continuity' in testing the significance of the difference between correlated proportions," *Psychometrika*, vol. 13, no. 3, pp. 185–187, Sep. 1948, doi: `10.1007/BF02289261`.
20. B. Efron, "Bootstrap methods: Another look at the jackknife," *The Annals of Statistics*, vol. 7, no. 1, pp. 1–26, Jan. 1979, doi: `10.1214/aos/1176344552`.
21. G. W. Brier, "Verification of forecasts expressed in terms of probability," *Monthly Weather Review*, vol. 78, no. 1, pp. 1–3, Jan. 1950, doi: `10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2`.
