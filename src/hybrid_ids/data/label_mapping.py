"""
Canonical Cybersecurity Attack Taxonomy & Harmonization Engine.
Maps source-specific labels from Edge-IIoTset (15 classes) and CICIoT2023 (34 classes)
into a mathematically defensible unified ontology with audit provenance.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd


# Canonical High-Level Threat Categories
CANONICAL_CATEGORIES = [
    "Benign",
    "DDoS",
    "DoS",
    "Reconnaissance",
    "Brute_Force",
    "Spoofing",
    "Web_Attack",
    "Malware"
]

# Canonical Fine-Grained Attack Classes
CANONICAL_CLASSES = [
    "Benign",
    "DDoS_UDP",
    "DDoS_ICMP",
    "DDoS_TCP",
    "DDoS_HTTP",
    "DoS_Generic",
    "Reconnaissance",
    "Brute_Force",
    "Spoofing_MITM",
    "Web_Attack",
    "Malware"
]

# Edge-IIoTset Mapping Dictionary
EDGE_IIOT_MAPPING = {
    "Normal": {
        "canonical_class": "Benign",
        "canonical_category": "Benign",
        "rationale": "Standard normal operational baseline traffic",
        "confidence": "high",
        "shared_evaluation": True
    },
    "DDoS_UDP": {
        "canonical_class": "DDoS_UDP",
        "canonical_category": "DDoS",
        "rationale": "Volumetric UDP flood targeting exhaustion of socket buffers",
        "confidence": "high",
        "shared_evaluation": True
    },
    "DDoS_ICMP": {
        "canonical_class": "DDoS_ICMP",
        "canonical_category": "DDoS",
        "rationale": "Volumetric ICMP echo request flood causing bandwidth saturation",
        "confidence": "high",
        "shared_evaluation": True
    },
    "DDoS_TCP": {
        "canonical_class": "DDoS_TCP",
        "canonical_category": "DDoS",
        "rationale": "TCP SYN/ACK flood exhausting kernel connection state tables",
        "confidence": "high",
        "shared_evaluation": True
    },
    "DDoS_HTTP": {
        "canonical_class": "DDoS_HTTP",
        "canonical_category": "DDoS",
        "rationale": "Application-layer HTTP GET/POST flood saturating web server threads",
        "confidence": "high",
        "shared_evaluation": True
    },
    "Port_Scanning": {
        "canonical_class": "Reconnaissance",
        "canonical_category": "Reconnaissance",
        "rationale": "Active network port discovery probes (Nmap/masscan)",
        "confidence": "high",
        "shared_evaluation": True
    },
    "Fingerprinting": {
        "canonical_class": "Reconnaissance",
        "canonical_category": "Reconnaissance",
        "rationale": "Active OS / service banner grabbing probes",
        "confidence": "high",
        "shared_evaluation": True
    },
    "Vulnerability_scanner": {
        "canonical_class": "Reconnaissance",
        "canonical_category": "Reconnaissance",
        "rationale": "Automated security scanner probing for known CVEs",
        "confidence": "high",
        "shared_evaluation": True
    },
    "Password": {
        "canonical_class": "Brute_Force",
        "canonical_category": "Brute_Force",
        "rationale": "Credential dictionary and brute-force authentication attacks",
        "confidence": "high",
        "shared_evaluation": True
    },
    "MITM": {
        "canonical_class": "Spoofing_MITM",
        "canonical_category": "Spoofing",
        "rationale": "ARP poisoning and DNS spoofing man-in-the-middle interception",
        "confidence": "high",
        "shared_evaluation": True
    },
    "SQL_injection": {
        "canonical_class": "Web_Attack",
        "canonical_category": "Web_Attack",
        "rationale": "Web application SQL query tampering and injection",
        "confidence": "high",
        "shared_evaluation": True
    },
    "XSS": {
        "canonical_class": "Web_Attack",
        "canonical_category": "Web_Attack",
        "rationale": "Cross-site scripting payload injection into web application",
        "confidence": "high",
        "shared_evaluation": True
    },
    "Uploading": {
        "canonical_class": "Web_Attack",
        "canonical_category": "Web_Attack",
        "rationale": "Unrestricted malicious web shell or binary upload",
        "confidence": "high",
        "shared_evaluation": True
    },
    "Backdoor": {
        "canonical_class": "Malware",
        "canonical_category": "Malware",
        "rationale": "Persistent remote access trojan / reverse shell execution",
        "confidence": "high",
        "shared_evaluation": True
    },
    "Ransomware": {
        "canonical_class": "Malware",
        "canonical_category": "Malware",
        "rationale": "Cryptographic extortion malware encrypting local filesystem",
        "confidence": "high",
        "shared_evaluation": True
    }
}

# CICIoT2023 Mapping Dictionary
CICIOT2023_MAPPING = {
    "BenignTraffic": {
        "canonical_class": "Benign",
        "canonical_category": "Benign",
        "rationale": "Legitimate baseline IoT network traffic",
        "confidence": "high",
        "shared_evaluation": True
    },
    "DDoS-UDP_Flood": {
        "canonical_class": "DDoS_UDP",
        "canonical_category": "DDoS",
        "rationale": "Distributed UDP packet flood",
        "confidence": "high",
        "shared_evaluation": True
    },
    "DDoS-UDP_Fragmentation": {
        "canonical_class": "DDoS_UDP",
        "canonical_category": "DDoS",
        "rationale": "Fragmented UDP volumetric flood",
        "confidence": "high",
        "shared_evaluation": True
    },
    "DDoS-ICMP_Flood": {
        "canonical_class": "DDoS_ICMP",
        "canonical_category": "DDoS",
        "rationale": "Distributed ICMP ping flood",
        "confidence": "high",
        "shared_evaluation": True
    },
    "DDoS-ICMP_Fragmentation": {
        "canonical_class": "DDoS_ICMP",
        "canonical_category": "DDoS",
        "rationale": "Fragmented ICMP ping flood",
        "confidence": "high",
        "shared_evaluation": True
    },
    "DDoS-TCP_Flood": {
        "canonical_class": "DDoS_TCP",
        "canonical_category": "DDoS",
        "rationale": "Distributed TCP flood",
        "confidence": "high",
        "shared_evaluation": True
    },
    "DDoS-SYN_Flood": {
        "canonical_class": "DDoS_TCP",
        "canonical_category": "DDoS",
        "rationale": "Distributed SYN flood exhausting TCP handshakes",
        "confidence": "high",
        "shared_evaluation": True
    },
    "DDoS-PSHACK_Flood": {
        "canonical_class": "DDoS_TCP",
        "canonical_category": "DDoS",
        "rationale": "TCP PSH-ACK flag volumetric flood",
        "confidence": "high",
        "shared_evaluation": True
    },
    "DDoS-RSTFINFlood": {
        "canonical_class": "DDoS_TCP",
        "canonical_category": "DDoS",
        "rationale": "TCP RST-FIN anomaly flood",
        "confidence": "high",
        "shared_evaluation": True
    },
    "DDoS-SynonymousIP_Flood": {
        "canonical_class": "DDoS_TCP",
        "canonical_category": "DDoS",
        "rationale": "Spoofed identical IP TCP flood",
        "confidence": "high",
        "shared_evaluation": True
    },
    "DDoS-ACK_Fragmentation": {
        "canonical_class": "DDoS_TCP",
        "canonical_category": "DDoS",
        "rationale": "Fragmented TCP ACK flood",
        "confidence": "high",
        "shared_evaluation": True
    },
    "DDoS-SlowLoris": {
        "canonical_class": "DDoS_HTTP",
        "canonical_category": "DDoS",
        "rationale": "Low-and-slow HTTP header exhaustion attack",
        "confidence": "high",
        "shared_evaluation": True
    },
    "DDoS-HTTP_Flood": {
        "canonical_class": "DDoS_HTTP",
        "canonical_category": "DDoS",
        "rationale": "Distributed HTTP GET/POST flood",
        "confidence": "high",
        "shared_evaluation": True
    },
    "DoS-HTTP_Flood": {
        "canonical_class": "DDoS_HTTP",
        "canonical_category": "DoS",
        "rationale": "Single-source HTTP GET/POST flood",
        "confidence": "high",
        "shared_evaluation": True
    },
    "DoS-UDP_Flood": {
        "canonical_class": "DDoS_UDP",
        "canonical_category": "DoS",
        "rationale": "Single-source UDP packet flood",
        "confidence": "high",
        "shared_evaluation": True
    },
    "DoS-TCP_Flood": {
        "canonical_class": "DDoS_TCP",
        "canonical_category": "DoS",
        "rationale": "Single-source TCP flood",
        "confidence": "high",
        "shared_evaluation": True
    },
    "DoS-SYN_Flood": {
        "canonical_class": "DDoS_TCP",
        "canonical_category": "DoS",
        "rationale": "Single-source SYN handshake exhaustion flood",
        "confidence": "high",
        "shared_evaluation": True
    },
    "Recon-PortScan": {
        "canonical_class": "Reconnaissance",
        "canonical_category": "Reconnaissance",
        "rationale": "Network port scan reconnaissance",
        "confidence": "high",
        "shared_evaluation": True
    },
    "Recon-OSScan": {
        "canonical_class": "Reconnaissance",
        "canonical_category": "Reconnaissance",
        "rationale": "Operating system fingerprinting scan",
        "confidence": "high",
        "shared_evaluation": True
    },
    "Recon-HostDiscovery": {
        "canonical_class": "Reconnaissance",
        "canonical_category": "Reconnaissance",
        "rationale": "IP subnet ping sweep and host discovery",
        "confidence": "high",
        "shared_evaluation": True
    },
    "Recon-PingSweep": {
        "canonical_class": "Reconnaissance",
        "canonical_category": "Reconnaissance",
        "rationale": "ICMP ping sweep for live target identification",
        "confidence": "high",
        "shared_evaluation": True
    },
    "VulnerabilityScan": {
        "canonical_class": "Reconnaissance",
        "canonical_category": "Reconnaissance",
        "rationale": "Automated security flaw scanning",
        "confidence": "high",
        "shared_evaluation": True
    },
    "DictionaryBruteForce": {
        "canonical_class": "Brute_Force",
        "canonical_category": "Brute_Force",
        "rationale": "Authentication brute-force / password spraying",
        "confidence": "high",
        "shared_evaluation": True
    },
    "MITM-ArpSpoofing": {
        "canonical_class": "Spoofing_MITM",
        "canonical_category": "Spoofing",
        "rationale": "Layer 2 ARP poisoning interception",
        "confidence": "high",
        "shared_evaluation": True
    },
    "DNS_Spoofing": {
        "canonical_class": "Spoofing_MITM",
        "canonical_category": "Spoofing",
        "rationale": "DNS cache poisoning / redirection",
        "confidence": "high",
        "shared_evaluation": True
    },
    "SqlInjection": {
        "canonical_class": "Web_Attack",
        "canonical_category": "Web_Attack",
        "rationale": "Web SQL injection",
        "confidence": "high",
        "shared_evaluation": True
    },
    "CommandInjection": {
        "canonical_class": "Web_Attack",
        "canonical_category": "Web_Attack",
        "rationale": "Remote shell command injection",
        "confidence": "high",
        "shared_evaluation": True
    },
    "XSS": {
        "canonical_class": "Web_Attack",
        "canonical_category": "Web_Attack",
        "rationale": "Cross-site scripting",
        "confidence": "high",
        "shared_evaluation": True
    },
    "Uploading_Attack": {
        "canonical_class": "Web_Attack",
        "canonical_category": "Web_Attack",
        "rationale": "Web file upload exploit",
        "confidence": "high",
        "shared_evaluation": True
    },
    "BrowserHijacking": {
        "canonical_class": "Web_Attack",
        "canonical_category": "Web_Attack",
        "rationale": "Client-side browser manipulation exploit",
        "confidence": "medium",
        "shared_evaluation": True
    },
    "Backdoor_Malware": {
        "canonical_class": "Malware",
        "canonical_category": "Malware",
        "rationale": "Persistent backdoor installation",
        "confidence": "high",
        "shared_evaluation": True
    },
    "Mirai-greeth_flood": {
        "canonical_class": "Malware",
        "canonical_category": "Malware",
        "rationale": "Mirai botnet GRE Ethernet encapsulation flood",
        "confidence": "high",
        "shared_evaluation": True
    },
    "Mirai-greip_flood": {
        "canonical_class": "Malware",
        "canonical_category": "Malware",
        "rationale": "Mirai botnet GRE IP encapsulation flood",
        "confidence": "high",
        "shared_evaluation": True
    },
    "Mirai-udpplain": {
        "canonical_class": "Malware",
        "canonical_category": "Malware",
        "rationale": "Mirai botnet UDP plain traffic flood",
        "confidence": "high",
        "shared_evaluation": True
    }
}


class LabelHarmonizer:
    """Harmonizes raw labels from both datasets into canonical classes."""

    def __init__(self):
        self.edge_map = EDGE_IIOT_MAPPING
        self.cic_map = CICIOT2023_MAPPING

    def map_label(self, raw_label: str, dataset_origin: str) -> Dict[str, Any]:
        """Maps a raw label to its canonical representation."""
        clean_label = str(raw_label).strip()
        if dataset_origin.lower() in ("edge_iiot", "edge"):
            entry = self.edge_map.get(clean_label)
            if entry is not None:
                return {
                    "original_label": clean_label,
                    "dataset_origin": "edge_iiot",
                    **entry
                }
        elif dataset_origin.lower() in ("cic_iot_2023", "ciciot23", "cic"):
            entry = self.cic_map.get(clean_label)
            if entry is not None:
                return {
                    "original_label": clean_label,
                    "dataset_origin": "cic_iot_2023",
                    **entry
                }

        # Fallback for unrecognized classes
        return {
            "original_label": clean_label,
            "dataset_origin": dataset_origin,
            "canonical_class": "Unknown",
            "canonical_category": "Unknown",
            "rationale": "Unmapped label",
            "confidence": "low",
            "shared_evaluation": False
        }

    def transform_series(self, series: pd.Series, dataset_origin: str, target_level: str = "class") -> pd.Series:
        """Transforms a pandas Series of raw labels into canonical labels."""
        mapping = self.edge_map if dataset_origin.lower() in ("edge_iiot", "edge") else self.cic_map
        col = "canonical_class" if target_level == "class" else "canonical_category"
        return series.astype(str).str.strip().map(lambda x: mapping.get(x, {}).get(col, "Unknown"))

    def get_manifest_dataframe(self) -> pd.DataFrame:
        """Generates an audit dataframe of all label mappings across both datasets."""
        rows = []
        for raw, val in self.edge_map.items():
            rows.append({
                "source_dataset": "Edge-IIoTset",
                "original_label": raw,
                "canonical_class": val["canonical_class"],
                "canonical_category": val["canonical_category"],
                "confidence": val["confidence"],
                "shared_evaluation": val["shared_evaluation"],
                "rationale": val["rationale"]
            })
        for raw, val in self.cic_map.items():
            rows.append({
                "source_dataset": "CICIoT2023",
                "original_label": raw,
                "canonical_class": val["canonical_class"],
                "canonical_category": val["canonical_category"],
                "confidence": val["confidence"],
                "shared_evaluation": val["shared_evaluation"],
                "rationale": val["rationale"]
            })
        return pd.DataFrame(rows)
