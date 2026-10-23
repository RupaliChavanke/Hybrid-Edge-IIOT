"""
Integration test for Redpanda Streaming Producer, Consumer, and Threat Detection.
Tests live packet streaming, background consumption, model scoring, and graceful stop.
"""

import time
from app.components.state import (
    start_producer,
    stop_producer,
    start_consumer,
    stop_consumer,
    start_pipeline,
    stop_pipeline,
    get_producer,
    get_consumer,
    reset_pipeline_stats,
    inject_attack_type_sample
)


def test_streaming_producer_consumer_full_duplex():
    """Validates that producer produces, consumer detects threats in real time, and pipeline stops cleanly."""
    reset_pipeline_stats()

    # 1. Start pipeline with Hybrid Ensemble
    prod, cons = start_pipeline(rate=50.0, model_name="Hybrid Ensemble")
    assert prod is not None
    assert cons is not None
    assert prod.is_running
    assert cons.is_running

    # 2. Allow packets to stream for 1.5 seconds
    time.sleep(1.5)

    stats = cons.get_stats()
    assert stats["messages_received"] > 0, "Consumer should receive streaming packets"
    assert stats["messages_processed"] > 0, "Consumer should process packets with ML model"
    assert stats["total_threats"] + stats["total_benign"] == stats["messages_processed"]

    # 3. Test on-demand attack injection
    injected = inject_attack_type_sample("DDoS_UDP")
    assert injected is True
    time.sleep(0.5)

    # 4. Verify predictions deque has entries with required schema
    assert len(cons.recent_predictions) > 0
    latest = cons.recent_predictions[-1]
    assert "event_id" in latest
    assert "predicted_attack" in latest
    assert "confidence" in latest
    assert "latency_breakdown" in latest

    # 5. Stop pipeline cleanly
    stop_pipeline()
    assert not prod.is_running
    assert not cons.is_running
