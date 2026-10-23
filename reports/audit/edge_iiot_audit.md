# Edge-IIoTset Independent Dataset Audit (Phase 4)

**Audit Date:** 2026-10-22 02:08:40  
**Source Path:** `data/raw/edge_iiot/ML-EdgeIIoT-dataset.csv`  
**SHA-256 Digest:** `53101fad091af20bee815860962a5016b802ee45d456a141042f6183094c3c1a`  
**File Size:** 78.38 MB  
**Load Time:** 1.02 s  
**In-Memory Footprint:** 123.08 MB  

## 1. Dimensionality & Volume
- **Total Records (Rows):** 157,800
- **Total Features (Columns):** 63
- **Duplicate Records:** 814 (0.52%)

## 2. Missing & Infinite Values
- **Columns with Missing Values:** 0
  - None (0 null values detected across all columns).
- **Columns with Infinite Values (`inf` / `-inf`):** 0
  - None (0 infinite values detected).

## 3. Class Imbalance & Threat Distribution
- **Target Label Field:** `Attack_type`
- **Unique Classes:** 15
- **Max-to-Min Imbalance Ratio:** 24.28:1

| Attack Class | Sample Count | Percentage (%) |
|---|---|---|
| `Normal` | 24,301 | 15.40% |
| `DDoS_UDP` | 14,498 | 9.19% |
| `DDoS_ICMP` | 14,090 | 8.93% |
| `Ransomware` | 10,925 | 6.92% |
| `DDoS_HTTP` | 10,561 | 6.69% |
| `SQL_injection` | 10,311 | 6.53% |
| `Uploading` | 10,269 | 6.51% |
| `DDoS_TCP` | 10,247 | 6.49% |
| `Backdoor` | 10,195 | 6.46% |
| `Vulnerability_scanner` | 10,076 | 6.39% |
| `Port_Scanning` | 10,071 | 6.38% |
| `XSS` | 10,052 | 6.37% |
| `Password` | 9,989 | 6.33% |
| `MITM` | 1,214 | 0.77% |
| `Fingerprinting` | 1,001 | 0.63% |

## 4. Feature Quality & Identifier Leakage Risks
- **Constant Columns (0 Variance):** 8 (`icmp.unused, http.tls_port, dns.qry.type, dns.retransmit_request_in, mqtt.msg_decoded_as, mbtcp.len, mbtcp.trans_id, mbtcp.unit_id`)
- **Near-Constant Columns (>99% dominant):** 13
- **High-Risk Identifier Fields:** 10
  - Examples: `frame.time, ip.src_host, ip.dst_host, arp.dst.proto_ipv4, arp.src.proto_ipv4, icmp.transmit_timestamp, http.request.uri.query, http.request.full_uri`
  - *Recommendation:* Strip raw IPs, MACs, and full URIs to prevent shortcut learning. Retain extracted length and statistical signatures.
