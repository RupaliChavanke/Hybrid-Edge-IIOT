"""
Pydantic Data Schemas for Network Telemetry and SOC Alerts.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import time
import uuid


class TelemetryEvent(BaseModel):
    """Network flow telemetry message schema."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    source_dataset: str = "edge_iiot"
    features: Dict[str, Any]


class PredictionEvent(BaseModel):
    """Model inference result message schema."""
    event_id: str
    timestamp: float = Field(default_factory=time.time)
    predicted_label: str
    predicted_class_idx: int
    confidence: float
    is_attack: bool
    severity: str
    latency_ms: float
    model_version: str


class SOCAlert(BaseModel):
    """High-priority Security Operations Center alert message schema."""
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str
    timestamp: float = Field(default_factory=time.time)
    threat_category: str
    attack_type: str
    confidence: float
    severity: str
    recommended_action: str
