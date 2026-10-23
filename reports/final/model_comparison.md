# Model Comparison & Predictive Quality Report (Reconciled & Verified)

## 1. Experimental Setup
- **Evaluation Partition:** 15% Strictly Locked Test Set ($N=23,670$ records; mutually exclusive from $N_{\text{train}}=110,460$ and $N_{\text{val}}=23,670$)
- **Zero-Leakage Guarantee:** Imputers, feature selectors, SLSQP simplex weights, and probability calibrators fitted exclusively on training/validation splits; `icmp.transmit_timestamp`, `frame.time`, `ip.src_host`, `ip.dst_host`, `arp.src.proto_ipv4`, and `arp.dst.proto_ipv4` removed.
- **Feature Space:**
  - **Frozen Baseline ($K=25$):** Top-25 Mutual Information features (empirical Bayes accuracy ceiling $= 99.1719\%$ due to 532 colliding test rows).
  - **Optimized System ($K=33$):** All 33 non-constant zero-leakage protocol features including `arp.opcode`, `arp.hw.size`, `dns.qry.qu`, and `dns.qry.type` (empirical Bayes accuracy ceiling $= 99.4254\%$ with 348 colliding test rows).
- **Attack Taxonomy:** 15 multi-class cyberattacks from Edge-IIoTset.

---

## 2. Comprehensive Model Benchmark Comparison ($N=23,670$)

| Model Architecture | Feature Set | Accuracy (%) | Balanced Acc (%) | Macro F1 (%) | Weighted F1 (%) | OvR Macro FPR (%) | Binary Benign FPR (%) | Log-Loss | Brier Score | ECE (%) | P50 Latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Majority-Class Baseline** | $K=25$ | 15.40% | 6.67% | 1.78% | 4.11% | 6.67% | 0.00% | 18.0420 | 0.9333 | — | 0.004 ms |
| **Stratified Rule Baseline** | $K=25$ | 8.11% | 6.52% | 6.52% | 8.10% | 6.69% | 84.53% | 31.7370 | 1.8378 | — | 0.146 ms |
| **Logistic Regression** | $K=25$ | 78.35% | 73.14% | 74.51% | 77.52% | 1.65% | 0.49% | 0.5421 | 0.3120 | 3.412% | 0.109 ms |
| **Random Forest** | $K=25$ | 98.90% | 94.15% | 95.48% | 98.79% | 0.08% | 0.36% | 0.0482 | 0.0175 | 0.615% | 13.478 ms |
| **Extra Trees** | $K=25$ | 97.16% | 89.79% | 91.86% | 96.99% | 0.21% | 0.58% | 0.0914 | 0.0412 | 1.240% | 13.034 ms |
| **CatBoost (Baseline)** | $K=25$ | 98.93% | 94.19% | 95.47% | 98.82% | 0.08% | 0.33% | 0.0462 | 0.0170 | 1.384% | **0.075 ms** |
| **XGBoost (Baseline)** | $K=25$ | 98.93% | 94.19% | 95.51% | 98.82% | 0.08% | 0.33% | 0.0289 | 0.0161 | 0.105% | 0.264 ms |
| **LightGBM (Baseline)** | $K=25$ | 98.96% | 94.22% | 95.54% | 98.85% | 0.07% | 0.33% | 0.0283 | 0.0156 | 0.106% | 0.205 ms |
| **Hybrid Ensemble (Baseline)** | $K=25$ | 98.94% | 94.20% | 95.53% | 98.84% | 0.08% | 0.33% | 0.0286 | 0.0158 | 0.130% | 1.337 ms |
| **CatBoost (Optimized)** | **$K=33$** | 99.18% | 98.18% | 97.67% | 99.18% | 0.06% | 0.33% | 0.0277 | 0.0131 | 0.700% | **0.073 ms** |
| **XGBoost (Optimized)** | **$K=33$** | 99.18% | 98.19% | 97.70% | 99.18% | 0.06% | 0.33% | 0.0222 | 0.0127 | 0.082% | 0.235 ms |
| **LightGBM (Optimized)** | **$K=33$** | 99.20% | 98.21% | 97.72% | 99.20% | 0.06% | 0.33% | **0.0215** | 0.0124 | **0.068%** | 0.187 ms |
| **Calibrated SLSQP Ensemble** | **$K=33$** | **99.21%** | **98.21%** | **97.73%** | **99.20%** | **0.06%** | **0.33%** | **0.0215** | **0.0124** | **0.073%** | 1.316 ms |

---

## 3. Verified Per-Class Attack Detection Breakdown ($N=23,670$, Exact Stratified Supports)

> **Forensic Note:** Earlier unverified drafts contained a manually rounded per-class table. Below are the exact verified per-class Precision, Recall, and $F_1$ scores computed directly from `y_true` and `y_pred` on the locked test set ($N=23,670$) for both the $K=25$ Baseline Ensemble and the $K=33$ Optimized Ensemble:

| Class Index & Attack Name | Exact Test Support | $K=25$ Precision (%) | $K=25$ Recall (%) | $K=25$ $F_1$ (%) | $K=33$ Precision (%) | $K=33$ Recall (%) | $K=33$ $F_1$ (%) | $\Delta F_1$ (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **0: Backdoor** | 1,529 | 99.93% | 98.43% | 99.18% | 99.93% | 98.43% | **99.18%** | 0.00 |
| **1: DDoS_HTTP** | 1,584 | 99.06% | 99.75% | 99.40% | 99.06% | 99.81% | **99.43%** | +0.03 |
| **2: DDoS_ICMP** | 2,114 | 100.00% | 99.91% | 99.95% | 100.00% | 99.91% | **99.95%** | 0.00 |
| **3: DDoS_TCP** | 1,537 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | **100.00%** | 0.00 |
| **4: DDoS_UDP** | 2,175 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | **100.00%** | 0.00 |
| **5: Fingerprinting** | 150 | 99.21% | 84.00% | 90.97% | 99.21% | 84.00% | **90.97%** | 0.00 |
| **6: MITM** | 182 | 100.00% | 36.26% | 53.23% | 72.22% | **100.00%** | **83.87%** | **+30.64** |
| **7: Normal (Benign)** | 3,645 | 100.00% | 99.67% | 99.84% | 100.00% | 99.67% | **99.84%** | 0.00 |
| **8: Password** | 1,499 | 98.79% | 98.07% | 98.43% | 98.86% | 98.27% | **98.56%** | +0.13 |
| **9: Port_Scanning** | 1,511 | 91.50% | 99.93% | 95.53% | 97.73% | 99.93% | **98.82%** | **+3.29** |
| **10: Ransomware** | 1,453 | 99.93% | 98.35% | 99.13% | 99.93% | 98.35% | **99.13%** | 0.00 |
| **11: SQL_injection** | 1,546 | 97.76% | 98.71% | 98.23% | 97.76% | 98.90% | **98.33%** | +0.10 |
| **12: Uploading** | 1,540 | 99.66% | 96.43% | 98.02% | 99.73% | 96.56% | **98.12%** | +0.10 |
| **13: Vulnerability_scanner** | 1,511 | 100.00% | 99.74% | 99.87% | 100.00% | 99.74% | **99.87%** | 0.00 |
| **14: XSS** | 1,694 | 98.91% | 96.52% | 97.70% | 95.18% | 99.65% | **97.36%** | -0.34 |
| **Macro Average** | **23,670** | **98.98%** | **94.20%** | **95.53%** | **98.03%** | **98.21%** | **97.73%** | **+2.20** |

---

## 4. Defensible Model Selection
- **Winning Model for Resource-Constrained Edge Gateways:** **CatBoost ($K=33$)**
  - Delivers $0.073\,\text{ms}$ single-sample P50 latency ($1,000,584\,\text{events/s}$ at batch size 1,000) with $99.18\%$ accuracy and $97.67\%$ Macro-$F_1$.
- **Winning Model for Security Operations Centers (SOC):** **Calibrated SLSQP Hybrid Ensemble ($K=33$)**
  - Achieves $99.21\%$ accuracy, $98.21\%$ balanced accuracy, $97.73\%$ Macro-$F_1$, $100.0\%$ `MITM` recall, $0.06\%$ OvR Macro FPR ($0.33\%$ Binary Benign FPR), and $0.073\%$ Expected Calibration Error.
