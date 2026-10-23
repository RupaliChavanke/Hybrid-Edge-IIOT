# CICIoT2023 Independent Dataset Audit (Phase 4)

**Audit Date:** 2026-10-22 02:08:41  
**Source Path:** `data/raw/cic_iot_2023/test.csv`  
**Test Partition SHA-256:** `15aeb1f47a7bd3e3fc394d18d7b196b3640329abaae65d5220df4def6119bfc6`  
**Total Dataset Storage Size:** 663.64 MB  
**Audit Partition Size:** 100,000 rows $\times$ 47 columns  

## 1. Dimensionality & Schema
- **Audited Sample Records:** 100,000
- **Total Features (Columns):** 47
- **Duplicate Records in Sample:** 36 (0.04%)

## 2. Missing & Infinite Values
- **Missing Values:** 0
- **Infinite Values:** 0

## 3. Attack Distribution in CICIoT2023 Sample
- **Target Label Column:** `label`
- **Class Imbalance Ratio:** 3866.75:1
- **Top Attack Classes:**
  - `DDoS-ICMP_Flood`: 15,467 (15.47%)
  - `DDoS-UDP_Flood`: 11,541 (11.54%)
  - `DDoS-TCP_Flood`: 9,578 (9.58%)
  - `DDoS-SYN_Flood`: 8,795 (8.79%)
  - `DDoS-PSHACK_Flood`: 8,695 (8.70%)
  - `DDoS-RSTFINFlood`: 8,575 (8.58%)
  - `DDoS-SynonymousIP_Flood`: 7,710 (7.71%)
  - `DoS-UDP_Flood`: 7,123 (7.12%)
  - `DoS-TCP_Flood`: 5,753 (5.75%)
  - `DoS-SYN_Flood`: 4,378 (4.38%)
  - `BenignTraffic`: 2,370 (2.37%)
  - `Mirai-greeth_flood`: 2,122 (2.12%)
  - `Mirai-udpplain`: 1,985 (1.98%)
  - `Mirai-greip_flood`: 1,627 (1.63%)
  - `DDoS-ICMP_Fragmentation`: 1,018 (1.02%)

## 4. Structural Feature Characteristics
- **Constant Columns:** 6 (`ece_flag_number, cwr_flag_number, Telnet, SMTP, IRC, DHCP`)
- **Near-Constant Columns (>99% dominant):** 6
- **Cross-Domain Feature Comparison:** Features are aggregated statistical flow summaries (e.g. `flow_duration`, `Header_Length`, `Rate`, `Min`, `Max`, `AVG`, `Std`, `Tot_size`) contrasting with Edge-IIoTset packet-level Wireshark dissections.
