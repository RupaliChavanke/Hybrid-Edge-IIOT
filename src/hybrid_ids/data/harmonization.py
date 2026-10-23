"""
Feature Harmonization Module for Cross-Dataset Compatibility.
Defines semantically compatible common network flow features across Edge-IIoTset and CICIoT2023.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np


# Manifest of harmonized common flow features
HARMONIZED_FEATURE_MANIFEST = [
    {
        "canonical_name": "proto_tcp",
        "definition": "Binary indicator whether flow protocol is Transmission Control Protocol (TCP)",
        "unit": "boolean / binary flag (0 or 1)",
        "edge_source": "derived: tcp.flags / tcp.dstport > 0",
        "cic_source": "TCP",
        "transformation": "Binary cast",
        "status": "exact_match"
    },
    {
        "canonical_name": "proto_udp",
        "definition": "Binary indicator whether flow protocol is User Datagram Protocol (UDP)",
        "unit": "boolean / binary flag (0 or 1)",
        "edge_source": "derived: udp.port > 0 / udp.stream > 0",
        "cic_source": "UDP",
        "transformation": "Binary cast",
        "status": "exact_match"
    },
    {
        "canonical_name": "proto_icmp",
        "definition": "Binary indicator whether flow protocol is Internet Control Message Protocol (ICMP)",
        "unit": "boolean / binary flag (0 or 1)",
        "edge_source": "derived: icmp.checksum > 0",
        "cic_source": "ICMP",
        "transformation": "Binary cast",
        "status": "exact_match"
    },
    {
        "canonical_name": "proto_http",
        "definition": "Binary indicator whether flow carries Hypertext Transfer Protocol (HTTP)",
        "unit": "boolean / binary flag (0 or 1)",
        "edge_source": "derived: http.response > 0 / http.request.method is not empty",
        "cic_source": "HTTP",
        "transformation": "Binary cast",
        "status": "exact_match"
    },
    {
        "canonical_name": "proto_dns",
        "definition": "Binary indicator whether flow carries Domain Name System (DNS) traffic",
        "unit": "boolean / binary flag (0 or 1)",
        "edge_source": "derived: dns.qry.name is not empty / dns.qry.type > 0",
        "cic_source": "DNS",
        "transformation": "Binary cast",
        "status": "exact_match"
    },
    {
        "canonical_name": "proto_arp",
        "definition": "Binary indicator whether flow protocol is Address Resolution Protocol (ARP)",
        "unit": "boolean / binary flag (0 or 1)",
        "edge_source": "derived: arp.opcode > 0",
        "cic_source": "ARP",
        "transformation": "Binary cast",
        "status": "exact_match"
    },
    {
        "canonical_name": "flag_fin",
        "definition": "TCP FIN control flag presence",
        "unit": "binary flag (0 or 1)",
        "edge_source": "tcp.connection.fin",
        "cic_source": "fin_flag_number",
        "transformation": "Numeric float32 cast",
        "status": "exact_match"
    },
    {
        "canonical_name": "flag_syn",
        "definition": "TCP SYN control flag presence",
        "unit": "binary flag (0 or 1)",
        "edge_source": "tcp.connection.syn",
        "cic_source": "syn_flag_number",
        "transformation": "Numeric float32 cast",
        "status": "exact_match"
    },
    {
        "canonical_name": "flag_rst",
        "definition": "TCP RST control flag presence",
        "unit": "binary flag (0 or 1)",
        "edge_source": "tcp.connection.rst",
        "cic_source": "rst_flag_number",
        "transformation": "Numeric float32 cast",
        "status": "exact_match"
    },
    {
        "canonical_name": "flag_ack",
        "definition": "TCP ACK control flag presence",
        "unit": "binary flag (0 or 1)",
        "edge_source": "tcp.flags.ack",
        "cic_source": "ack_flag_number",
        "transformation": "Numeric float32 cast",
        "status": "exact_match"
    },
    {
        "canonical_name": "packet_length",
        "definition": "Observed packet / payload length",
        "unit": "bytes",
        "edge_source": "tcp.len",
        "cic_source": "AVG",
        "transformation": "Standardized non-negative float32",
        "status": "compatible_approximation"
    },
    {
        "canonical_name": "time_delta",
        "definition": "Packet inter-arrival time / flow duration delta",
        "unit": "seconds",
        "edge_source": "udp.time_delta",
        "cic_source": "IAT",
        "transformation": "Clipped non-negative float32",
        "status": "compatible_approximation"
    }
]


class DatasetHarmonizer:
    """Extracts canonical flow features from heterogeneous IoT intrusion datasets."""

    @staticmethod
    def extract_harmonized_features_edge(df: pd.DataFrame) -> pd.DataFrame:
        """Extracts canonical common feature subset from raw Edge-IIoTset records."""
        out = pd.DataFrame(index=df.index)

        # Protocols
        out["proto_tcp"] = (
            (df["tcp.dstport"].fillna(0).astype(float) > 0) |
            (df["tcp.len"].fillna(0).astype(float) > 0)
        ).astype(np.float32)

        out["proto_udp"] = (
            (df["udp.port"].fillna(0).astype(float) > 0)
        ).astype(np.float32)

        out["proto_icmp"] = (
            (df["icmp.checksum"].fillna(0).astype(float) > 0)
        ).astype(np.float32)

        out["proto_http"] = (
            (df["http.response"].fillna(0).astype(float) > 0) |
            (df["http.request.method"].fillna("").astype(str).str.len() > 0)
        ).astype(np.float32)

        out["proto_dns"] = (
            (df["dns.qry.type"].fillna(0).astype(float) > 0)
        ).astype(np.float32)

        out["proto_arp"] = (
            (df["arp.opcode"].fillna(0).astype(float) > 0)
        ).astype(np.float32)

        # TCP Flags
        out["flag_fin"] = df["tcp.connection.fin"].fillna(0).astype(np.float32)
        out["flag_syn"] = df["tcp.connection.syn"].fillna(0).astype(np.float32)
        out["flag_rst"] = df["tcp.connection.rst"].fillna(0).astype(np.float32)
        out["flag_ack"] = df["tcp.flags.ack"].fillna(0).astype(np.float32)

        # Length and timing
        out["packet_length"] = df["tcp.len"].fillna(0).astype(np.float32).clip(lower=0.0)
        out["time_delta"] = df["udp.time_delta"].fillna(0).astype(np.float32).clip(lower=0.0)

        return out

    @staticmethod
    def extract_harmonized_features_cic(df: pd.DataFrame) -> pd.DataFrame:
        """Extracts canonical common feature subset from raw CICIoT2023 records."""
        out = pd.DataFrame(index=df.index)

        # Protocols
        out["proto_tcp"] = df["TCP"].fillna(0).astype(np.float32)
        out["proto_udp"] = df["UDP"].fillna(0).astype(np.float32)
        out["proto_icmp"] = df["ICMP"].fillna(0).astype(np.float32)
        out["proto_http"] = df["HTTP"].fillna(0).astype(np.float32)
        out["proto_dns"] = df["DNS"].fillna(0).astype(np.float32)
        out["proto_arp"] = df["ARP"].fillna(0).astype(np.float32)

        # TCP Flags
        out["flag_fin"] = df["fin_flag_number"].fillna(0).astype(np.float32)
        out["flag_syn"] = df["syn_flag_number"].fillna(0).astype(np.float32)
        out["flag_rst"] = df["rst_flag_number"].fillna(0).astype(np.float32)
        out["flag_ack"] = df["ack_flag_number"].fillna(0).astype(np.float32)

        # Length and timing
        out["packet_length"] = df["AVG"].fillna(0).astype(np.float32).clip(lower=0.0)
        out["time_delta"] = df["IAT"].fillna(0).astype(np.float32).clip(lower=0.0)

        return out

    @staticmethod
    def get_manifest_dataframe() -> pd.DataFrame:
        """Returns the complete feature mapping manifest as a DataFrame."""
        return pd.DataFrame(HARMONIZED_FEATURE_MANIFEST)
