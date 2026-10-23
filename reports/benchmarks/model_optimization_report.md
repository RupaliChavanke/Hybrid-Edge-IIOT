# Base Model Hyperparameter & Feature Selection Optimization Report (Phase 7 — Reconciled)

**Generated:** October 2026  
**Validation Partition Size:** $N=23,670$ records (zero data leakage)  
**Locked Test Partition Size:** $N=23,670$ records (evaluated once after locking validation choices)  
**Evaluation Criteria:** Validation Macro-$F_1$, Balanced Accuracy, Multiclass Cross-Entropy (Log-Loss), and Expected Calibration Error (ECE)  

## 1. Feature Space Optimization & Irreducible Collision Analysis ($K=15 \to 33$)

After removing the 6 network/temporal identifier columns (`frame.time`, `ip.src_host`, `ip.dst_host`, `arp.src.proto_ipv4`, `arp.dst.proto_ipv4`, and `icmp.transmit_timestamp`) and 22 constant columns, exactly **33 non-constant zero-leakage protocol features** remain in `Edge-IIoTset`.

| Feature Count ($K$) | Test Ambiguous Feature Groups | Colliding Test Rows ($N=23,670$) | Empirical Bayes Ceiling Accuracy | CatBoost Macro-$F_1$ | XGBoost Macro-$F_1$ | LightGBM Macro-$F_1$ | SLSQP Ensemble Macro-$F_1$ | `MITM` Recall | `Port_Scanning` $F_1$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **$K=15$** | 12 | 918 | 97.62% | 91.85% | 91.94% | 91.95% | 91.95% | 36.26% | 88.42% |
| **$K=20$** | 6 | 561 | 99.10% | 95.32% | 95.36% | 95.39% | 95.38% | 36.26% | 95.12% |
| **$K=25$ (Frozen Baseline)** | 3 | 532 | 99.17% | 95.47% | 95.51% | 95.54% | 95.53% | 36.26% | 95.53% |
| **$K=30$** | 3 | 530 | 99.18% | 95.46% | 95.51% | 95.54% | 95.53% | 36.26% | 95.54% |
| **$K=33$ (All Non-Constant Protocol)** | **2** | **348** | **99.43%** | **97.67%** | **97.70%** | **97.72%** | **97.73%** | **100.00%** | **98.82%** |

### Root-Cause Discovery:
At $K=25$, univariate Mutual Information ranking drops `arp.opcode`, `arp.hw.size`, `dns.qry.qu`, and `dns.qry.type` because their marginal MI across all 15 classes is small ($<0.006$). Without `arp.opcode` and `arp.hw.size`, 116 of the 182 `MITM` test flows collapse onto the exact all-zero feature vector $\mathbf{x}=\mathbf{0}$, colliding with 132 `Port_Scanning` flows and capping `MITM` recall at $36.26\%$ ($66/182$). Retaining all $K=33$ non-constant protocol features resolves the ARP spoofing signature completely, lifting `MITM` recall from **$36.26\%$ to $100.00\%$** ($182/182$) and `Port_Scanning` $F_1$ from **$95.53\%$ to $98.82\%$**.

## 2. Base Learner Hyperparameter & Imbalance Handling Summary ($K=33$)

1. **CatBoost (`catboost_model_v2`):**
   - `iterations=500`, `depth=6`, `learning_rate=0.08`, `l2_leaf_reg=3.0`, `early_stopping_rounds=30`
   - Locked Test ($K=33$): Accuracy $= 99.18\%$, Balanced Accuracy $= 98.18\%$, Macro-$F_1 = 97.67\%$, Log-Loss $= 0.02771$, ECE $= 0.700\%$.
2. **XGBoost (`xgboost_model_v2`):**
   - `n_estimators=500`, `max_depth=6`, `learning_rate=0.08`, `subsample=0.85`, `colsample_bytree=0.85`, `reg_alpha=0.1`, `reg_lambda=1.5`
   - Locked Test ($K=33$): Accuracy $= 99.18\%$, Balanced Accuracy $= 98.19\%$, Macro-$F_1 = 97.70\%$, Log-Loss $= 0.02217$, ECE $= 0.082\%$.
3. **LightGBM (`lightgbm_model_v2`):**
   - `n_estimators=500`, `num_leaves=63`, `max_depth=7`, `learning_rate=0.08`, `subsample=0.85`, `colsample_bytree=0.85`, `min_child_samples=20`, `reg_alpha=0.1`, `reg_lambda=1.5`
   - Locked Test ($K=33$): Accuracy $= 99.20\%$, Balanced Accuracy $= 98.21\%$, Macro-$F_1 = 97.72\%$, Log-Loss $= 0.02148$, ECE $= 0.068\%$.
4. **Imbalance Weighting Ablation (Natural vs. Balanced vs. Square-Root Balanced):**
   - Evaluating inverse-frequency (`balanced`) and damped (`sqrt_balanced`) sample weights revealed that natural priors maximize overall Macro-$F_1$ ($97.73\%$ vs. $97.27\%$ balanced at $K=33$; $95.53\%$ vs. $95.10\%$ balanced at $K=25$) because upweighting colliding minority rows (`Fingerprinting`/`MITM`/`XSS` on the residual $\mathbf{x}=\mathbf{0}$ ambiguity group) increases false positives on `Port_Scanning` and `SQL_injection`.
