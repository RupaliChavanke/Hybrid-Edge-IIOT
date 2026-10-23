"""
Alerting Engine and Incident Response Routing Module.
Translates model predictions into SOC alert messages with operational severity levels and recommended actions.
"""

from typing import Dict, List, Optional, Any
from src.hybrid_ids.streaming.schemas import SOCAlert, PredictionEvent


# Threat Category Alert Routing Policy
RECOMMENDED_ACTIONS = {
    "DDoS": "Deploy rate-limiting at edge ingress switch; activate upstream BGP Flowspec blackholing.",
    "DoS": "Throttle client connection requests; enable SYN cookies on web gateway.",
    "Malware": "Quarantine host IP; terminate suspicious C2 beacon process and isolate network segment.",
    "Web_Attack": "Block originating source IP on WAF; inspect application server payload logs.",
    "Spoofing": "Inspect ARP cache on local switch; enable Dynamic ARP Inspection (DAI) and DHCP Snooping.",
    "Brute_Force": "Temporarily lock targeted account; enforce multi-factor authentication (MFA).",
    "Reconnaissance": "Log probe signatures; monitor source IP for subsequent lateral movement attempts.",
    "Benign": "No action required; standard operational telemetry logged."
}


class AlertEngine:
    """Generates actionable SOC alerts from prediction events."""

    @staticmethod
    def generate_alert(prediction: PredictionEvent, threat_category: str) -> Optional[SOCAlert]:
        """Generates an alert if the prediction indicates an attack."""
        if not prediction.is_attack:
            return None

        action = RECOMMENDED_ACTIONS.get(threat_category, "Monitor network segment and review security logs.")
        return SOCAlert(
            event_id=prediction.event_id,
            timestamp=prediction.timestamp,
            threat_category=threat_category,
            attack_type=prediction.predicted_label,
            confidence=prediction.confidence,
            severity=prediction.severity,
            recommended_action=action
        )
