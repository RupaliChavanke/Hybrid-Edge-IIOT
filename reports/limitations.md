# Scientific Limitations, Threats to Validity & Operational Constraints (Reconciled)

**Framework:** Hybrid Edge-IIoTset + CICIoT2023 Intrusion Detection Framework  
**Date:** October 2026  
**Document Status:** COMPLETE, FORENSICALLY RECONCILED & VERIFIED  

---

## 1. Irreducible Feature-Vector Collisions & The Empirical Bayes Ceiling

A forensic audit of the locked test set ($N=23,670$) revealed that tabular network intrusion datasets contain distinct packets that project onto identical feature vectors after stripping host IP addresses, timestamps, and raw payload strings:
- **At $K=25$ (Top-25 Mutual Information Features):** 532 test rows belong to ambiguous feature-vector equivalence classes where identical $\mathbf{x}$ vectors carry conflicting ground-truth labels (`MITM` vs. `Port_Scanning` on $\mathbf{x}=\mathbf{0}$, and `SQL_injection`/`Uploading`/`Password`/`XSS` on HTTP SYN/RST/FIN packets). The exact empirical Bayes accuracy ceiling at $K=25$ is **$99.1719\%$** ($23,474 / 23,670$).
- **At $K=33$ (All Non-Constant Zero-Leakage Protocol Features):** Retaining `arp.opcode`, `arp.hw.size`, `dns.qry.qu`, and `dns.qry.type` resolves the `MITM` ARP spoofing collision ($36.26\% \to 100.00\%$ `MITM` recall), reducing colliding test rows to 348 and raising the empirical Bayes ceiling to **$99.4254\%$** ($23,534 / 23,670$).
- **Implication:** Any model claiming $>99.43\%$ zero-leakage accuracy on this `Edge-IIoTset` partition is necessarily exploiting leaked session identifiers (`ip.src_host`, `frame.time`, or `icmp.transmit_timestamp`) rather than genuine protocol semantics. Furthermore, within the remaining 348 colliding HTTP handshake rows, pure packet-header classifiers cannot separate application-layer `SQL_injection`, `Uploading`, `Password`, and `XSS` on payload-free TCP control packets (`tcp.len = 0`) without stateful bi-directional flow context.

---

## 2. Sensitivity of Axis-Aligned Decision Trees to Continuous Gaussian Noise on Discrete Indicators

Controlled perturbation experiments (`scripts/run_ablations_and_statistics.py`) revealed an important negative finding that was misstated in earlier draft summaries:
- Adding continuous zero-mean Gaussian noise $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, (\sigma \cdot \text{std}_j)^2)$ at $\sigma \in \{0.01, 0.05, 0.10\}$ causes locked-test Macro-$F_1$ to drop immediately from $95.53\%$ to **$54.17\%$–$56.23\%$** at $K=25$ (and from $97.73\%$ to **$53.30\%$** at $K=33$).
- **Root Cause:** Unlike continuous image pixels or acoustic spectra, 26 of the 33 network features are discrete binary flags, port numbers, or protocol indicator counts that take exact `0.0` values whenever a protocol layer is absent (e.g., `arp.opcode = 0`, `udp.port = 0`, `dns.qry.name.len = 0`). Axis-aligned GBDTs place tight split thresholds $\tau \approx 10^{-5}$ above zero to test protocol presence. Adding continuous Gaussian noise shifts zero-valued indicators across $\tau$, causing TCP flows to trigger ARP, DNS, and MQTT tree branches simultaneously.
- **Feature Dropout Sensitivity:** Similarly, random zero-masking of $5\%$, $10\%$, and $20\%$ of active features reduces Macro-$F_1$ to $86.78\%$, $75.46\%$, and $56.37\%$ ($K=25$) because setting non-zero TCP ports or flags to zero mimics empty scanning or spoofing flows.
- **Operational Mitigation:** Production preprocessors must enforce schema-aware integer/boolean quantization and valid protocol-header gating prior to tree inference rather than passing unquantized noisy floating-point perturbations into axis-aligned split nodes.

---

## 3. Cross-Dataset Domain Shift & Representation Granularity Gap

Cross-dataset generalization experiments (`Exp A`–`Exp E`) highlight two distinct challenges when transferring between `Edge-IIoTset` (Industrial IoT packet/row telemetry) and `CICIoT2023` (Consumer IoT aggregated flow summaries):
1. **Representation Compression Loss ($97.73\% \to 59.05\%$):** Projecting `Edge-IIoTset` from 33 packet-level features down to the 12 harmonized cross-dataset indicators removes fine-grained TCP sequence, acknowledgment, checksum, and URI length features, reducing in-domain `Edge-IIoTset` Macro-$F_1$ to $59.05\%$ ($59.45\%$ single LightGBM).
2. **Zero-Shot Covariate & Prior Shift:** Zero-shot transfer yields $27.66\%$ Macro-$F_1$ (`Edge` $\to$ `CIC`) and $9.34\%$ Macro-$F_1$ (`CIC` $\to$ `Edge`, $12.70\%$ class-balanced). While domain-balanced multi-domain pooling (`Exp C-DB2`) achieves **$70.32\%$ Macro-$F_1$** on `CICIoT2023` ($+4.16\,\text{pp}$ over in-domain `CICIoT2023`), true zero-shot transfer without target-domain samples remains an open challenge requiring standardized bidirectional flow exporters (e.g., IPFIX/NetFlow v9) across testbeds.

---

## 4. Streaming Broker Serialization Overhead vs. Micro-Batching

Hardware profiling on Apple Silicon M2 shows that single-model in-memory inference requires only **`0.073 ms`** P50 latency for CatBoost (`1,000,584 events/s` at batch size 1,000) and **`1.316 ms`** for the 3-model ensemble. However, executing the complete end-to-end streaming pipeline (`JSON deserialize -> DataFrame construct -> PipelineCleaner -> FeatureSelector -> Ensemble -> AlertEngine`) one event at a time incurs **`11.842 ms`** P50 latency (`82.8 events/s`) due to per-record Pandas DataFrame construction overhead. Micro-batching ($B \ge 100$) amortizes this overhead by $79\times$, reducing per-event latency to **`0.150 ms/event`** (`6,657.0 events/s` at $B=100$) and **`0.048 ms/event`** (`20,640.0 events/s` at $B=500$).
