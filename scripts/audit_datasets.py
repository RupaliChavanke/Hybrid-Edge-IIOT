#!/usr/bin/env python3
"""
Comprehensive Dataset Audit and Harmonization Tool (Phase 4).

Audits Edge-IIoTset and CICIoT2023 independently and generates:
  - reports/audit/edge_iiot_audit.md
  - reports/audit/cic_iot_2023_audit.md
  - reports/audit/feature_compatibility_matrix.csv
  - reports/audit/label_mapping.csv
  - reports/audit/duplicate_record_report.csv
"""

import os
import sys
import time
import json
import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

EDGE_IIOT_PATH = REPO_ROOT / "data" / "raw" / "edge_iiot" / "ML-EdgeIIoT-dataset.csv"
CIC_TEST_PATH = REPO_ROOT / "data" / "raw" / "cic_iot_2023" / "test.csv"
CIC_VAL_PATH = REPO_ROOT / "data" / "raw" / "cic_iot_2023" / "validation.csv"
AUDIT_DIR = REPO_ROOT / "reports" / "audit"


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_edge_iiot() -> Dict[str, Any]:
    print(f"[Audit] Auditing Edge-IIoTset from {EDGE_IIOT_PATH}...")
    t0 = time.time()
    size_bytes = EDGE_IIOT_PATH.stat().st_size
    file_hash = sha256_file(EDGE_IIOT_PATH)

    # Read data
    df = pd.read_csv(EDGE_IIOT_PATH, low_memory=False)
    load_time = time.time() - t0
    mem_mb = df.memory_usage(deep=True).sum() / (1024**2)

    n_rows, n_cols = df.shape
    cols = list(df.columns)

    # Missing & Infs
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0].to_dict()

    # Numerical infinite check
    numeric_df = df.select_dtypes(include=[np.number])
    inf_counts = np.isinf(numeric_df).sum()
    inf_cols = inf_counts[inf_counts > 0].to_dict()

    # Constants & near constants
    constant_cols = [c for c in cols if df[c].nunique(dropna=False) <= 1]
    near_constant_cols = [c for c in cols if df[c].nunique(dropna=False) > 1 and (df[c].value_counts().iloc[0] / n_rows) > 0.99]

    # Identifier fields
    id_fields = [c for c in cols if any(k in c.lower() for k in ["ip", "host", "mac", "uri", "time", "date", "payload"])]

    # Label distribution
    label_col = "Attack_type" if "Attack_type" in df.columns else "label"
    label_counts = df[label_col].value_counts().to_dict()
    imbalance_ratio = max(label_counts.values()) / min(label_counts.values())

    # Exact duplicates
    duplicate_rows = int(df.duplicated().sum())

    res = {
        "dataset_name": "Edge-IIoTset",
        "file_path": str(EDGE_IIOT_PATH.relative_to(REPO_ROOT)),
        "file_size_mb": size_bytes / (1024**2),
        "sha256": file_hash,
        "load_time_sec": load_time,
        "memory_mb": mem_mb,
        "row_count": n_rows,
        "col_count": n_cols,
        "columns": cols,
        "null_cols": null_cols,
        "inf_cols": inf_cols,
        "constant_cols": constant_cols,
        "near_constant_cols": near_constant_cols,
        "identifier_fields": id_fields,
        "label_counts": label_counts,
        "imbalance_ratio": imbalance_ratio,
        "duplicate_rows": duplicate_rows,
    }

    dup_pct = f"{res['duplicate_rows'] / res['row_count'] * 100:.2f}%"
    imb_ratio_str = f"{res['imbalance_ratio']:.2f}:1"
    file_size_str = f"{res['file_size_mb']:.2f} MB"
    load_time_str = f"{res['load_time_sec']:.2f} s"
    mem_str = f"{res['memory_mb']:.2f} MB"

    # Generate edge_iiot_audit.md
    md = f"""# Edge-IIoTset Independent Dataset Audit (Phase 4)

**Audit Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Source Path:** `{res['file_path']}`  
**SHA-256 Digest:** `{res['sha256']}`  
**File Size:** {file_size_str}  
**Load Time:** {load_time_str}  
**In-Memory Footprint:** {mem_str}  

## 1. Dimensionality & Volume
- **Total Records (Rows):** {res['row_count']:,}
- **Total Features (Columns):** {res['col_count']}
- **Duplicate Records:** {res['duplicate_rows']:,} ({dup_pct})

## 2. Missing & Infinite Values
- **Columns with Missing Values:** {len(res['null_cols'])}
"""
    if res['null_cols']:
        for c, cnt in res['null_cols'].items():
            col_pct = f"{cnt / res['row_count'] * 100:.2f}%"
            md += f"  - `{c}`: {cnt:,} missing ({col_pct})\n"
    else:
        md += "  - None (0 null values detected across all columns).\n"

    md += f"- **Columns with Infinite Values (`inf` / `-inf`):** {len(res['inf_cols'])}\n"
    if res['inf_cols']:
        for c, cnt in res['inf_cols'].items():
            md += f"  - `{c}`: {cnt:,} infinite values\n"
    else:
        md += "  - None (0 infinite values detected).\n"

    md += f"""
## 3. Class Imbalance & Threat Distribution
- **Target Label Field:** `Attack_type`
- **Unique Classes:** {len(res['label_counts'])}
- **Max-to-Min Imbalance Ratio:** {imb_ratio_str}

| Attack Class | Sample Count | Percentage (%) |
|---|---|---|
"""
    for l, cnt in res['label_counts'].items():
        md += f"| `{l}` | {cnt:,} | {cnt / res['row_count'] * 100:.2f}% |\n"

    md += f"""
## 4. Feature Quality & Identifier Leakage Risks
- **Constant Columns (0 Variance):** {len(res['constant_cols'])} (`{', '.join(res['constant_cols']) or 'None'}`)
- **Near-Constant Columns (>99% dominant):** {len(res['near_constant_cols'])}
- **High-Risk Identifier Fields:** {len(res['identifier_fields'])}
  - Examples: `{', '.join(res['identifier_fields'][:8])}`
  - *Recommendation:* Strip raw IPs, MACs, and full URIs to prevent shortcut learning. Retain extracted length and statistical signatures.
"""
    (AUDIT_DIR / "edge_iiot_audit.md").write_text(md, encoding="utf-8")
    return res


def audit_cic_iot() -> Dict[str, Any]:
    print(f"[Audit] Auditing CICIoT2023 from {CIC_TEST_PATH} and {CIC_VAL_PATH}...")
    t0 = time.time()
    size_bytes = CIC_TEST_PATH.stat().st_size + (CIC_VAL_PATH.stat().st_size if CIC_VAL_PATH.exists() else 0)
    test_hash = sha256_file(CIC_TEST_PATH)

    # Sample read for high performance (first 100,000 rows)
    df = pd.read_csv(CIC_TEST_PATH, nrows=100000, low_memory=False)
    load_time = time.time() - t0
    mem_mb = df.memory_usage(deep=True).sum() / (1024**2)

    n_rows, n_cols = df.shape
    cols = list(df.columns)

    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0].to_dict()

    numeric_df = df.select_dtypes(include=[np.number])
    inf_counts = np.isinf(numeric_df).sum()
    inf_cols = inf_counts[inf_counts > 0].to_dict()

    constant_cols = [c for c in cols if df[c].nunique(dropna=False) <= 1]
    near_constant_cols = [c for c in cols if df[c].nunique(dropna=False) > 1 and (df[c].value_counts().iloc[0] / n_rows) > 0.99]

    label_col = "label" if "label" in df.columns else "Attack_type"
    label_counts = df[label_col].value_counts().to_dict()
    imbalance_ratio = max(label_counts.values()) / max(1, min(label_counts.values()))
    dup_rows = int(df.duplicated().sum())

    res = {
        "dataset_name": "CICIoT2023",
        "file_path": str(CIC_TEST_PATH.relative_to(REPO_ROOT)),
        "file_size_mb": size_bytes / (1024**2),
        "sha256": test_hash,
        "sample_load_time_sec": load_time,
        "sample_rows": n_rows,
        "col_count": n_cols,
        "columns": cols,
        "null_cols": null_cols,
        "inf_cols": inf_cols,
        "constant_cols": constant_cols,
        "near_constant_cols": near_constant_cols,
        "label_counts": label_counts,
        "imbalance_ratio": imbalance_ratio,
        "duplicate_rows": dup_rows,
    }

    dup_pct_cic = f"{res['duplicate_rows'] / res['sample_rows'] * 100:.2f}%"
    imb_ratio_cic = f"{res['imbalance_ratio']:.2f}:1"
    file_size_cic = f"{res['file_size_mb']:.2f} MB"

    # Generate cic_iot_2023_audit.md
    md = f"""# CICIoT2023 Independent Dataset Audit (Phase 4)

**Audit Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Source Path:** `{res['file_path']}`  
**Test Partition SHA-256:** `{res['sha256']}`  
**Total Dataset Storage Size:** {file_size_cic}  
**Audit Partition Size:** {res['sample_rows']:,} rows $\\times$ {res['col_count']} columns  

## 1. Dimensionality & Schema
- **Audited Sample Records:** {res['sample_rows']:,}
- **Total Features (Columns):** {res['col_count']}
- **Duplicate Records in Sample:** {res['duplicate_rows']:,} ({dup_pct_cic})

## 2. Missing & Infinite Values
- **Missing Values:** {len(res['null_cols'])}
- **Infinite Values:** {len(res['inf_cols'])}

## 3. Attack Distribution in CICIoT2023 Sample
- **Target Label Column:** `{label_col}`
- **Class Imbalance Ratio:** {imb_ratio_cic}
- **Top Attack Classes:**
"""
    for l, cnt in list(res['label_counts'].items())[:15]:
        md += f"  - `{l}`: {cnt:,} ({cnt / res['sample_rows'] * 100:.2f}%)\n"

    md += f"""
## 4. Structural Feature Characteristics
- **Constant Columns:** {len(res['constant_cols'])} (`{', '.join(res['constant_cols']) or 'None'}`)
- **Near-Constant Columns (>99% dominant):** {len(res['near_constant_cols'])}
- **Cross-Domain Feature Comparison:** Features are aggregated statistical flow summaries (e.g. `flow_duration`, `Header_Length`, `Rate`, `Min`, `Max`, `AVG`, `Std`, `Tot_size`) contrasting with Edge-IIoTset packet-level Wireshark dissections.
"""
    (AUDIT_DIR / "cic_iot_2023_audit.md").write_text(md, encoding="utf-8")
    return res


def generate_feature_compatibility_matrix():
    """Builds canonical feature dictionary and cross-dataset compatibility matrix."""
    matrix = [
        ["canonical_feature", "edge_iiot_column", "cic_iot_column", "definition", "unit", "compatibility_decision", "transformation", "known_limitations"],
        ["flow_duration", "frame.time_delta", "flow_duration", "Duration of network flow or inter-frame delta", "seconds / microseconds", "Compatible with unit scaling", "Standardize to milliseconds", "Edge-IIoT measures delta while CIC aggregates duration"],
        ["header_length", "tcp.hdr_len", "Header_Length", "Total TCP/IP header byte length", "bytes", "Compatible", "Direct numeric cast", "CIC aggregates total headers; Edge captures per-frame headers"],
        ["packet_rate", "rate", "Rate", "Transmission rate of packets or frames", "packets/sec", "Compatible", "Standardize rate ratio", "Sampling window differences between capture topologies"],
        ["total_bytes", "tcp.len", "Tot_size", "Total bytes transferred in flow payload", "bytes", "Compatible", "Direct numeric cast", "Edge contains per-packet payload length; CIC contains flow total"],
        ["protocol_type", "ip.proto", "Protocol_Type", "IP layer transport protocol identifier", "categorical / int", "Compatible", "Integer protocol mapping (6=TCP, 17=UDP, 1=ICMP)", "Consistent standardized IANA protocol codes"],
        ["tcp_flags", "tcp.flags", "FIN_Flag_Cnt / SYN_Flag_Cnt", "TCP control flags", "bitmask vs count", "Partially Compatible", "Flag extraction (SYN, ACK, FIN)", "Edge provides raw 16-bit hex flags; CIC provides individual counts"],
        ["payload_mean", "http.content_length", "AVG", "Mean packet / content byte size", "bytes", "Compatible with median imputation", "Robust numeric scaling", "Missing in non-HTTP traffic for Edge-IIoTset"],
    ]

    with open(AUDIT_DIR / "feature_compatibility_matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(matrix)


def generate_label_mapping_matrix():
    """Builds unified attack taxonomy mapping table."""
    taxonomy = [
        ["source_dataset", "source_label", "canonical_category", "canonical_fine_grained", "shared_evaluation", "domain_shift_notes"],
        ["Edge-IIoTset", "Normal", "Benign", "Benign", "True", "Operational baseline traffic in IoT testbed"],
        ["Edge-IIoTset", "DDoS_UDP", "DDoS", "DDoS_UDP", "True", "High-volume UDP flood targeted at IoT broker"],
        ["Edge-IIoTset", "DDoS_ICMP", "DDoS", "DDoS_ICMP", "True", "ICMP ping flood exhaustion attack"],
        ["Edge-IIoTset", "DDoS_HTTP", "DDoS", "DDoS_HTTP", "True", "Layer-7 HTTP request flood"],
        ["Edge-IIoTset", "DDoS_TCP", "DDoS", "DDoS_TCP", "True", "TCP SYN / ACK flood attacks"],
        ["Edge-IIoTset", "Port_Scanning", "Reconnaissance", "Reconnaissance", "True", "Nmap network topology discovery"],
        ["Edge-IIoTset", "Fingerprinting", "Reconnaissance", "Reconnaissance", "True", "OS and service discovery probing"],
        ["Edge-IIoTset", "Vulnerability_scanner", "Reconnaissance", "Reconnaissance", "True", "Nessus / Nikto automated scanner"],
        ["Edge-IIoTset", "Password", "Brute_Force", "Brute_Force", "True", "Hydra dictionary password attacks"],
        ["Edge-IIoTset", "MITM", "Spoofing", "Spoofing_MITM", "True", "ARP poisoning man-in-the-middle"],
        ["Edge-IIoTset", "SQL_injection", "Web_Attack", "Web_Attack", "True", "SQL payload injections via web forms"],
        ["Edge-IIoTset", "XSS", "Web_Attack", "Web_Attack", "True", "Cross-site scripting injections"],
        ["Edge-IIoTset", "Uploading", "Web_Attack", "Web_Attack", "True", "Malicious file upload to web server"],
        ["Edge-IIoTset", "Backdoor", "Malware", "Malware", "True", "Persistent backdoor agent execution"],
        ["Edge-IIoTset", "Ransomware", "Malware", "Malware", "True", "WannaCry / IoT ransomware execution"],
        ["CICIoT2023", "BenignTraffic", "Benign", "Benign", "True", "Legitimate smart home device network traffic"],
        ["CICIoT2023", "DDoS-UDP_Flood", "DDoS", "DDoS_UDP", "True", "Direct equivalent to Edge-IIoTset DDoS_UDP"],
        ["CICIoT2023", "DDoS-ICMP_Flood", "DDoS", "DDoS_ICMP", "True", "Direct equivalent to Edge-IIoTset DDoS_ICMP"],
        ["CICIoT2023", "DDoS-HTTP_Flood", "DDoS", "DDoS_HTTP", "True", "Direct equivalent to Edge-IIoTset DDoS_HTTP"],
        ["CICIoT2023", "DDoS-TCP_Flood", "DDoS", "DDoS_TCP", "True", "Direct equivalent to Edge-IIoTset DDoS_TCP"],
        ["CICIoT2023", "Recon-PortScan", "Reconnaissance", "Reconnaissance", "True", "Port scanning activity equivalent"],
        ["CICIoT2023", "Recon-OSScan", "Reconnaissance", "Reconnaissance", "True", "OS fingerprinting scan equivalent"],
        ["CICIoT2023", "DictionaryBruteForce", "Brute_Force", "Brute_Force", "True", "Password brute forcing equivalent"],
        ["CICIoT2023", "MITM-ArpSpoofing", "Spoofing", "Spoofing_MITM", "True", "ARP spoofing MITM equivalent"],
        ["CICIoT2023", "SqlInjection", "Web_Attack", "Web_Attack", "True", "SQL injection web attack equivalent"],
        ["CICIoT2023", "BrowserHijacking", "Web_Attack", "Web_Attack", "True", "Client-side web attack equivalent"],
        ["CICIoT2023", "Mirai-greeth_flood", "Malware", "Malware", "True", "Mirai IoT botnet malware campaign"],
    ]

    with open(AUDIT_DIR / "label_mapping.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(taxonomy)


def generate_duplicate_record_report(edge_info: Dict[str, Any], cic_info: Dict[str, Any]):
    """Generates duplicate record analysis report."""
    rows = [
        ["dataset", "total_records", "exact_duplicate_records", "duplicate_rate_pct", "handling_strategy", "scientific_rationale"],
        ["Edge-IIoTset", edge_info["row_count"], edge_info["duplicate_rows"], f"{edge_info['duplicate_rows'] / edge_info['row_count'] * 100:.2f}%", "Preserve in distinct temporal partitions", "High-frequency packet floods naturally emit identical TCP ACK/RST packets; deduplication would artificially distort attack density."],
        ["CICIoT2023", cic_info["sample_rows"], cic_info["duplicate_rows"], f"{cic_info['duplicate_rows'] / cic_info['sample_rows'] * 100:.2f}%", "Preserve in distinct flow partitions", "Flow statistical summaries repeat across uniform probing bursts; dropping records destroys flow volume signatures."],
    ]

    with open(AUDIT_DIR / "duplicate_record_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def main():
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    edge_info = audit_edge_iiot()
    cic_info = audit_cic_iot()

    print("[Audit] Generating feature compatibility matrix...")
    generate_feature_compatibility_matrix()

    print("[Audit] Generating label mapping matrix...")
    generate_label_mapping_matrix()

    print("[Audit] Generating duplicate record report...")
    generate_duplicate_record_report(edge_info, cic_info)

    print("[Audit] Phase 4 dataset audit and harmonization artifacts successfully written to reports/audit/")


if __name__ == "__main__":
    main()
