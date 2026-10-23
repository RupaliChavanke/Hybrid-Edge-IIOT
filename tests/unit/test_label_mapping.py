"""
Unit Tests for Canonical Label Harmonization and Threat Taxonomy.
"""

import pytest
import pandas as pd
from src.hybrid_ids.data.label_mapping import LabelHarmonizer


def test_edge_iiot_label_mapping():
    """Verify Edge-IIoTset classes map to expected canonical categories."""
    mapper = LabelHarmonizer()

    normal = mapper.map_label("Normal", "edge_iiot")
    assert normal["canonical_class"] == "Benign"
    assert normal["canonical_category"] == "Benign"

    ddos_udp = mapper.map_label("DDoS_UDP", "edge_iiot")
    assert ddos_udp["canonical_class"] == "DDoS_UDP"
    assert ddos_udp["canonical_category"] == "DDoS"

    ransomware = mapper.map_label("Ransomware", "edge_iiot")
    assert ransomware["canonical_class"] == "Malware"
    assert ransomware["canonical_category"] == "Malware"


def test_cic_iot_2023_label_mapping():
    """Verify CICIoT2023 classes map to expected canonical categories."""
    mapper = LabelHarmonizer()

    benign = mapper.map_label("BenignTraffic", "cic_iot_2023")
    assert benign["canonical_class"] == "Benign"
    assert benign["canonical_category"] == "Benign"

    port_scan = mapper.map_label("Recon-PortScan", "cic_iot_2023")
    assert port_scan["canonical_class"] == "Reconnaissance"
    assert port_scan["canonical_category"] == "Reconnaissance"

    mirai = mapper.map_label("Mirai-udpplain", "cic_iot_2023")
    assert mirai["canonical_class"] == "Malware"
    assert mirai["canonical_category"] == "Malware"


def test_unknown_label_handling():
    """Verify unmapped anomalous label is marked as Unknown without crashing."""
    mapper = LabelHarmonizer()
    unknown = mapper.map_label("NonExistentAttackClass_999", "edge_iiot")
    assert unknown["canonical_class"] == "Unknown"
    assert unknown["shared_evaluation"] is False
