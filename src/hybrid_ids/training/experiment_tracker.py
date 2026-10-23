"""
Experiment Tracking and Artifact Logging Module.
Saves reproducible experiment logs, hashes, timestamps, and metric summaries in JSON format.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import time
from datetime import datetime
from src.hybrid_ids.config import get_project_root


class ExperimentTracker:
    """Logs and archives reproducible training and evaluation runs."""

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or (get_project_root() / "artifacts" / "logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def record_experiment(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        dataset_hashes: Dict[str, str],
        model_artifacts: Dict[str, str],
        duration_sec: float
    ) -> Path:
        """Saves a comprehensive experiment run record."""
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        record = {
            "experiment_name": experiment_name,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "duration_sec": duration_sec,
            "config": config,
            "dataset_hashes": dataset_hashes,
            "model_artifacts": model_artifacts,
            "metrics": metrics
        }

        output_path = self.log_dir / f"{experiment_name}_{timestamp_str}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        return output_path
