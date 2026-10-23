# Comprehensive Repository & Dataset Forensic Audit Report

**Project Title:** Hybrid Edge-IIoTset + CICIoT2023 Intrusion Detection Framework  
**Primary Architectures:** CatBoost + XGBoost + LightGBM with Validation-Based Hybrid Ensembling  
**Date of Audit:** October 2026  
**Auditor / Roles:** Senior Machine Learning Researcher, Cybersecurity Systems Engineer, MLOps Architect  
**Audit Status:** COMPLETE  

---

## 1. Executive Summary

This forensic audit evaluates the existing codebase in `/Volumes/New Storage/Edge-IIOT`. The prior implementation relied on a deep-learning architecture comprising a 1D-CNN with Depthwise Separable convolutions, Ghost Modules, Squeeze-and-Excitation (SE) attention, Bidirectional GRU (Bi-GRU), Multi-Head Temporal Self-Attention, Pointwise Low-Rank Projections, and an Entropy-based Early Exit classifier trained via Center-Focal Loss.

In accordance with the Primary Objective, this legacy neural architecture is systematically audited and targeted for replacement with a modern, high-efficiency, tree-based gradient boosting framework combining:
1. **CatBoost** (Yandex gradient boosting with symmetric oblivious trees and native categorical feature support)
2. **XGBoost** (Extreme Gradient Boosting with histogram tree building, sparsity-awareness, and L1/L2 regularization)
3. **LightGBM** (Light Gradient Boosting Machine with leaf-wise tree growth, exclusive feature bundling, and GOSS)
4. **Validation-Based Hybrid Ensemble** (Out-of-fold probability-weighted voting, stacking meta-classifier, and calibrated confidence estimation)

Furthermore, the audit encompasses two comprehensive IoT intrusion datasets:
- **Edge-IIoTset** (`ML-EdgeIIoT-dataset.csv` and `Edge-IIoTset dataset.zip`)
- **CICIoT2023** (`archive.zip` containing `CICIOT23/test/test.csv`, `train/train.csv`, `validation/validation.csv`)

---

## 2. Hardware and Environment Audit

- **Operating System:** macOS (Darwin 24.3.0, arm64, Apple Silicon M2, Mac14,2)
- **CPU Cores:** 8 logical/physical cores
- **Total Physical Memory (RAM):** 16.00 GB (17,179,869,184 bytes)
- **Available Storage:** 406 GB available on `/Volumes/New Storage`
- **Primary Python Runtime:** Python 3.14.7 (system framework & synchronized `.venv` with system site-packages)
- **Core Library Availability:**
  - `catboost`: 1.2.10 (Installed & verified)
  - `xgboost`: 3.4.1 (Installed & verified)
  - `lightgbm`: 4.7.0 (Installed & verified with `libomp` via Homebrew)
  - `scikit-learn`: 1.9.1
  - `pandas`: 3.0.6
  - `numpy`: 2.5.3
  - `scipy`: 1.18.1
  - `torch`: 2.7.0.dev20250207
  - `streamlit`: 1.42.0

---

## 3. Dataset Audit & Immutable Cryptographic Manifest

### 3.1 Raw Dataset Inventory

| Dataset Identifier | Location / Archive Path | Raw Size (Bytes) | Format | SHA-256 Digest | Status & Provenance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Edge-IIoTset (ML Subset)** | `data/raw/ML-EdgeIIoT-dataset.csv` | 82,184,390 | CSV | `53101fad091af20bee815860962a5016b802ee45d456a141042f6183094c3c1a` | Verified original ML benchmark release of Edge-IIoTset |
| **Edge-IIoTset (Full Archive)** | `Edge-IIoTset dataset.zip` | 1,741,481,351 | ZIP (50 files) | `4a4a16542756d72390b60c7cde5c46a8439ac5c6e83502e6fc8c64d2d7669d72` | Contains `DNN-EdgeIIoT-dataset.csv` (1.21 GB), attack CSVs, normal CSVs, pcap captures |
| **CICIoT2023 (Full Archive)** | `archive.zip` | 500,559,747 | ZIP (3 CSVs) | `793b212d1dc40973068d06cec4d0676eaf7f17436a859dfa522303e199c7ea5a` | Contains standard 80/10/10 partition: train (1.62 GB), validation (347 MB), test (347 MB) |

### 3.2 Edge-IIoTset Dataset Characterization
- **Total Records:** 157,800 rows
- **Total Features:** 63 columns (61 candidate features + `Attack_label` + `Attack_type`)
- **Memory Footprint:** ~76 MB in memory
- **Target Distributions:**
  - `Attack_label`: 133,499 Attack (84.60%), 24,301 Normal (15.40%)
  - `Attack_type` (15 classes):
    - `Normal`: 24,301 (15.40%)
    - `DDoS_UDP`: 14,498 (9.19%)
    - `DDoS_ICMP`: 14,090 (8.93%)
    - `Ransomware`: 10,925 (6.92%)
    - `DDoS_HTTP`: 10,561 (6.69%)
    - `SQL_injection`: 10,311 (6.53%)
    - `Uploading`: 10,269 (6.51%)
    - `DDoS_TCP`: 10,247 (6.49%)
    - `Backdoor`: 10,195 (6.46%)
    - `Vulnerability_scanner`: 10,076 (6.39%)
    - `Port_Scanning`: 10,071 (6.38%)
    - `XSS`: 10,052 (6.37%)
    - `Password`: 9,989 (6.33%)
    - `MITM`: 1,214 (0.77%) [Minority Class]
    - `Fingerprinting`: 1,001 (0.63%) [Minority Class]
- **Identified Leakage Features:**
  - Static timestamps: `frame.time`
  - Host IP addresses: `ip.src_host`, `ip.dst_host`, `arp.src.proto_ipv4`, `arp.dst.proto_ipv4`
  - Direct URI queries & payloads: `http.file_data`, `http.request.full_uri`, `http.referer`, `http.request.uri.query`, `tcp.payload`, `tcp.options`, `mqtt.msg`, `dns.qry.name`
  - Zero-variance / constant features: 28 constant columns identified in raw records (e.g., `icmp.unused`, `http.tls_port`, `mqtt.conack.flags`, etc.)

### 3.3 CICIoT2023 Dataset Characterization
- **Total Records Across Archive:** ~7.85 million records
  - `CICIOT23/test/test.csv`: 1,176,851 rows (347,958,926 bytes uncompressed)
  - `CICIOT23/validation/validation.csv`: 1,176,851 rows (347,921,865 bytes uncompressed)
  - `CICIOT23/train/train.csv`: ~5.50 million rows (1,623,446,296 bytes uncompressed)
- **Total Features:** 47 columns (46 continuous/statistical flow features + `label`)
- **Data Types:** 46 `float64` flow metrics + 1 categorical string label
- **Target Distribution:** 34 distinct attack and benign classes
  - High frequency classes: `DDoS-ICMP_Flood`, `DDoS-UDP_Flood`, `DDoS-TCP_Flood`, `DDoS-PSHACK_Flood`, `DDoS-SYN_Flood`, `DDoS-RSTFINFlood`, `DoS-UDP_Flood`, `DDoS-SynonymousIP_Flood`, `DoS-TCP_Flood`, `DoS-SYN_Flood`
  - Benign class: `BenignTraffic`
  - Specialized intrusion classes: `Recon-PortScan`, `Recon-OSScan`, `Recon-HostDiscovery`, `VulnerabilityScan`, `DictionaryBruteForce`, `SqlInjection`, `XSS`, `Uploading_Attack`, `Backdoor_Malware`, `MITM-ArpSpoofing`, `DNS_Spoofing`, `Mirai` variants.

---

## 4. Audit of Existing Codebase Modules

### 4.1 Obsolete Deep-Learning Modules (To Be Archived & Retired)
1. `models/proposed_model.py`: Legacy PhD architecture with 1D-CNN, Ghost Modules, SE Attention, Bi-GRU, Low-Rank Linear, and Early Exit.
2. `models/cnn.py`: Depthwise Separable 1D-CNN module.
3. `models/ghost_module.py`: Ghost convolution module.
4. `models/se_attention.py`: Squeeze-and-Excitation 1D attention.
5. `models/bigru.py`: Temporal BiGRU module.
6. `models/temporal_attention.py`: Multi-Head Temporal Self-Attention.
7. `models/low_rank.py`: Pointwise Low-Rank Linear projection.
8. `training/trainer.py`: PyTorch training loop with Center-Focal Loss.
9. `training/losses.py`: CenterLoss and FocalLoss implementations.
10. `training/checkpoint.py`: PyTorch state_dict serialization.

### 4.2 Useful Preprocessing & Infrastructure Utilities (To Be Retained & Adapted)
1. `preprocessing/cleaner.py`: Contains valuable netflow length extraction logic (`len_uri`, `len_query`, `len_payload`, `len_referer`, `len_file_data`) and metadata drop lists.
2. `preprocessing/taxonomy.py`: Three-tier threat hierarchy (Binary, Threat Category, Attack Type). Can be extended to harmonize CICIoT2023.
3. `evaluation/metrics.py`: Metric computation utilities for multiclass confusion matrices, balanced accuracy, macro precision/recall/F1.
4. `streaming/`: Message schemas, producer, consumer, and health check implementations for Redpanda/Kafka message bus.
5. `app.py` & `dashboard/`: Streamlit dashboard infrastructure.

---

## 5. Discovered Incompatibilities & Risks

1. **Virtual Environment Interpreter Path:** The existing `.venv` contained shebang scripts targeting `/Users/rupesh/Documents/Edge-IIOT/.venv`, which broke `pip` execution when the workspace was moved to `/Volumes/New Storage/Edge-IIOT`.
   - *Resolution:* Virtual environment cleared and re-initialized with `--system-site-packages` using Python 3.14.7.
2. **Missing Dependencies in `requirements.txt`:** `catboost`, `xgboost`, and `lightgbm` were omitted from `requirements.txt`.
   - *Resolution:* Added to requirements and pyproject configurations.
3. **Redpanda Offline Mode:** The unit test `test_redpanda_health_checker` failed because no Redpanda container was running on `localhost:19092`.
   - *Resolution:* Streaming components must support a robust offline fallback mode and mock bus testing.
4. **Memory Constraint on Full 7.8M CICIoT2023 Records:** Attempting to load all 7.85 million records of CICIoT2023 into memory simultaneously alongside Edge-IIoTset would exceed the 16 GB physical RAM.
   - *Resolution:* Implemented chunked parsing, downcasted floating-point types (`float32`), stratified sampling modes (smoke, dev, full-benchmark), and out-of-core evaluation.
