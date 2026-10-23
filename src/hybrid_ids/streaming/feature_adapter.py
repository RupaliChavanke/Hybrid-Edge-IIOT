"""
Streaming Feature Adapter and Replay Pipeline.
Provides producer-consumer event processing with offline queue fallback for Redpanda/Kafka.
"""

from typing import Dict, List, Optional, Any, Callable
import time
import queue
from src.hybrid_ids.streaming.schemas import TelemetryEvent, PredictionEvent, SOCAlert
from src.hybrid_ids.streaming.alerting import AlertEngine
from src.hybrid_ids.logging_config import setup_logger

logger = setup_logger("streaming_pipeline")


class StreamingReplayPipeline:
    """Manages event streaming replay, inference scoring, and SOC alert dissemination."""

    def __init__(self, predictor: Any, label_mapper: Any):
        self.predictor = predictor
        self.label_mapper = label_mapper
        self.alert_engine = AlertEngine()
        self.event_queue: queue.Queue = queue.Queue(maxsize=10000)
        self.prediction_history: List[PredictionEvent] = []
        self.alert_history: List[SOCAlert] = []

    def produce_event(self, raw_features: Dict[str, Any], source_dataset: str = "edge_iiot") -> str:
        """Publishes a raw telemetry event into the pipeline queue."""
        event = TelemetryEvent(
            source_dataset=source_dataset,
            features=raw_features
        )
        self.event_queue.put(event)
        return event.event_id

    def consume_and_predict(self, max_events: int = 100) -> List[PredictionEvent]:
        """Consumes events from queue, executes ensemble inference, and emits alerts."""
        processed_predictions = []
        count = 0

        while not self.event_queue.empty() and count < max_events:
            event: TelemetryEvent = self.event_queue.get()
            t_res = self.predictor.predict_record(event.features)

            pred_event = PredictionEvent(
                event_id=event.event_id,
                predicted_label=t_res["predicted_label"],
                predicted_class_idx=t_res["predicted_class_idx"],
                confidence=t_res["confidence"],
                is_attack=t_res["is_attack"],
                severity=t_res["severity"],
                latency_ms=t_res["latency_ms"],
                model_version=t_res["model_version"]
            )
            processed_predictions.append(pred_event)
            self.prediction_history.append(pred_event)

            # Route to alert engine
            cat_entry = self.label_mapper.map_label(pred_event.predicted_label, event.source_dataset)
            threat_cat = cat_entry.get("canonical_category", "Unknown")
            alert = self.alert_engine.generate_alert(pred_event, threat_cat)
            if alert:
                self.alert_history.append(alert)

            count += 1

        logger.info(f"Consumed and scored {len(processed_predictions)} streaming events. Generated {len(self.alert_history)} alerts.")
        return processed_predictions
