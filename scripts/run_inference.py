"""
Production Inference and Streaming Telemetry Replay Runner.
Loads serialized models and transformers, ingests test telemetry events,
produces predictions, and outputs SOC incident response alerts.
"""

from pathlib import Path
import sys
import json
import time
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.hybrid_ids.inference.model_registry import ModelRegistry
from src.hybrid_ids.inference.predictor import IntrusionPredictor
from src.hybrid_ids.streaming.feature_adapter import StreamingReplayPipeline
from src.hybrid_ids.data.label_mapping import LabelHarmonizer
from src.hybrid_ids.data.loaders import DataLoader
from src.hybrid_ids.config import get_project_root
from src.hybrid_ids.logging_config import setup_logger

logger = setup_logger("run_inference")


def main():
    root = get_project_root()
    registry = ModelRegistry()
    loader = DataLoader(root)
    label_mapper = LabelHarmonizer()

    logger.info("Loading registered models and transformers...")
    ensemble = registry.load_artifact("hybrid_ensemble", version="v2")
    cleaner = registry.load_artifact("cleaner", version="edge_iiot")
    selector = registry.load_artifact("feature_selector", version="edge_iiot")
    encoder = registry.load_artifact("label_encoder", version="edge_iiot")

    predictor = IntrusionPredictor(
        ensemble=ensemble,
        cleaner=cleaner,
        feature_selector=selector,
        label_encoder=encoder
    )

    pipeline = StreamingReplayPipeline(predictor=predictor, label_mapper=label_mapper)

    logger.info("Loading sample events for streaming replay...")
    df_raw = loader.load_edge_iiot(mode="smoke", sample_size=50)

    # Replay events
    logger.info(f"Replaying {len(df_raw)} network telemetry events through pipeline...")
    for idx, row in df_raw.iterrows():
        event_dict = row.to_dict()
        pipeline.produce_event(event_dict, source_dataset="edge_iiot")

    # Consume and score
    predictions = pipeline.consume_and_predict(max_events=len(df_raw))

    logger.info(f"Generated {len(pipeline.alert_history)} SOC Alerts from {len(predictions)} events.")
    for alert in pipeline.alert_history[:5]:
        logger.info(f"ALERT [{alert.severity}] {alert.attack_type} ({alert.threat_category}): {alert.recommended_action} (Conf: {alert.confidence})")

    # Export recent alerts to artifacts
    alerts_dir = root / "artifacts" / "predictions"
    alerts_dir.mkdir(parents=True, exist_ok=True)
    alerts_json = [a.model_dump() for a in pipeline.alert_history]
    with open(alerts_dir / "recent_soc_alerts.json", "w", encoding="utf-8") as f:
        json.dump(alerts_json, f, indent=2)

    logger.info("Inference replay completed successfully.")


if __name__ == "__main__":
    main()
