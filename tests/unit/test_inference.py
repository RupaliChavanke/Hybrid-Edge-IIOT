"""
Unit Tests for Inference Pipeline, Serialization, and Streaming Telemetry.
"""

import pytest
import numpy as np
import pandas as pd
from src.hybrid_ids.inference.model_registry import ModelRegistry
from src.hybrid_ids.inference.predictor import IntrusionPredictor
from src.hybrid_ids.streaming.schemas import TelemetryEvent, PredictionEvent
from src.hybrid_ids.streaming.alerting import AlertEngine
from src.hybrid_ids.streaming.feature_adapter import StreamingReplayPipeline
from src.hybrid_ids.data.label_mapping import LabelHarmonizer


class MockEnsemble:
    def __init__(self):
        self.class_names = ["Normal", "DDoS_TCP"]
        self.version = "2.0.0"

    def predict_proba(self, X):
        return np.array([[0.1, 0.9]])

    def predict_event(self, X):
        return {
            "predicted_class_idx": 1,
            "predicted_label": "DDoS_TCP",
            "confidence": 0.90,
            "is_attack": True,
            "severity": "CRITICAL",
            "latency_ms": 0.5,
            "model_version": self.version,
            "class_probabilities": {"Normal": 0.1, "DDoS_TCP": 0.9}
        }


class MockCleaner:
    def transform(self, df):
        return df


class MockSelector:
    def transform(self, df):
        return df


class MockEncoder:
    def __init__(self):
        self.idx_to_class = {0: "Normal", 1: "DDoS_TCP"}


def test_model_registry_save_load(tmp_path):
    """Verify artifact serialization and SHA-256 integrity check."""
    registry = ModelRegistry(registry_dir=tmp_path)
    sample_data = {"weights": [0.4, 0.3, 0.3], "version": "test"}

    path, digest = registry.save_artifact(sample_data, "test_model", version="v1")
    assert path.exists()

    loaded = registry.load_artifact("test_model", version="v1", expected_sha256=digest)
    assert loaded == sample_data

    with pytest.raises(ValueError):
        registry.load_artifact("test_model", version="v1", expected_sha256="invalid_hash_value")


def test_streaming_replay_pipeline():
    """Verify telemetry produce and consume flow with alert emission."""
    predictor = IntrusionPredictor(
        ensemble=MockEnsemble(),
        cleaner=MockCleaner(),
        feature_selector=MockSelector(),
        label_encoder=MockEncoder()
    )
    label_mapper = LabelHarmonizer()
    pipeline = StreamingReplayPipeline(predictor=predictor, label_mapper=label_mapper)

    event_id = pipeline.produce_event({"tcp.len": 1500, "tcp.flags.ack": 1.0})
    assert event_id is not None

    preds = pipeline.consume_and_predict(max_events=1)
    assert len(preds) == 1
    assert preds[0].predicted_label == "DDoS_TCP"
    assert preds[0].is_attack is True
    assert len(pipeline.alert_history) == 1
    assert pipeline.alert_history[0].threat_category == "DDoS"
    assert pipeline.alert_history[0].severity == "CRITICAL"
