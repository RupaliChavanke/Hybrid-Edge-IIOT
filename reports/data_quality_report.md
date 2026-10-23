# Forensic Data Quality and Schema Validation Report

**Date:** 2026-10-22T01:43:34.736184

## 1. Edge-IIoTset Machine Learning Partition

- **Total Rows:** 157,800
- **Total Columns:** 63
- **Memory Footprint:** 123.08 MB
- **Duplicate Rows:** 814
- **Constant Columns (8):** icmp.unused, http.tls_port, dns.qry.type, dns.retransmit_request_in, mqtt.msg_decoded_as, mbtcp.len, mbtcp.trans_id, mbtcp.unit_id...

### Target Class Distribution (Attack_type):

| Attack Class | Count | Percentage |
| :--- | :--- | :--- |
| Normal | 24,301 | 15.4% |
| DDoS_UDP | 14,498 | 9.188% |
| DDoS_ICMP | 14,090 | 8.929% |
| Ransomware | 10,925 | 6.923% |
| DDoS_HTTP | 10,561 | 6.693% |
| SQL_injection | 10,311 | 6.534% |
| Uploading | 10,269 | 6.508% |
| DDoS_TCP | 10,247 | 6.494% |
| Backdoor | 10,195 | 6.461% |
| Vulnerability_scanner | 10,076 | 6.385% |
| Port_Scanning | 10,071 | 6.382% |
| XSS | 10,052 | 6.37% |
| Password | 9,989 | 6.33% |
| MITM | 1,214 | 0.769% |
| Fingerprinting | 1,001 | 0.634% |

## 2. CICIoT2023 Telemetry Dataset

- **Sampled Rows:** 50,000
- **Total Columns:** 47
- **Memory Footprint:** 9.9 MB
- **Duplicate Rows in Sample:** 639
- **Constant Columns in Sample:** ece_flag_number, cwr_flag_number, Telnet, SMTP, IRC, DHCP

### Top 15 Attack Classes in CICIoT2023 Sample:

| Attack Class | Count | Percentage |
| :--- | :--- | :--- |
| DDoS-ICMP_Flood | 7,811 | 15.622% |
| DDoS-UDP_Flood | 5,811 | 11.622% |
| DDoS-TCP_Flood | 4,742 | 9.484% |
| DDoS-PSHACK_Flood | 4,383 | 8.766% |
| DDoS-SYN_Flood | 4,369 | 8.738% |
| DDoS-RSTFINFlood | 4,329 | 8.658% |
| DDoS-SynonymousIP_Flood | 3,863 | 7.726% |
| DoS-UDP_Flood | 3,525 | 7.05% |
| DoS-TCP_Flood | 2,853 | 5.706% |
| DoS-SYN_Flood | 2,230 | 4.46% |
| BenignTraffic | 1,135 | 2.27% |
| Mirai-greeth_flood | 1,032 | 2.064% |
| Mirai-udpplain | 938 | 1.876% |
| Mirai-greip_flood | 796 | 1.592% |
| DDoS-ICMP_Fragmentation | 480 | 0.96% |
