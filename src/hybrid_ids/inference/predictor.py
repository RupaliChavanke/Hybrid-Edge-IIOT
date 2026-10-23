"""
Real-Time Intrusion Predictor Module.
Provides low-latency scoring for single telemetry records or streaming micro-batches.
"""

from typing import Dict, List, Optional, Tuple, Any
import time
import numpy as np
import pandas as pd
from src.hybrid_ids.models.ensemble import HybridGBDTEnsemble


class IntrusionPredictor:
    """Production predictor wrapping the hybrid gradient-boosted ensemble."""

    def __init__(
        self,
        ensemble: HybridGBDTEnsemble,
        cleaner: Any,
        feature_selector: Any,
        label_encoder: Any
    ):
        self.ensemble = ensemble
        self.cleaner = cleaner
        self.feature_selector = feature_selector
        self.label_encoder = label_encoder

    def predict_record(self, raw_record: Dict[str, Any]) -> Dict[str, Any]:
        """Runs end-to-end cleaning, selection, and ensemble prediction on a raw network telemetry dictionary."""
        df_raw = pd.DataFrame([raw_record])

        # Apply cleaning pipeline
        X_clean = self.cleaner.transform(df_raw)

        # Apply feature selection
        X_selected = self.feature_selector.transform(X_clean)

        # Predict using ensemble
        return self.ensemble.predict_event(X_selected)

    def predict_batch(self, df_batch: pd.DataFrame) -> List[Dict[str, Any]]:
        """Processes a batch of raw telemetry events efficiently."""
        t_start = time.perf_counter()
        X_clean = self.cleaner.transform(df_batch)
        X_selected = self.feature_selector.transform(X_clean)
        probas = self.ensemble.predict_proba(X_selected)
        preds = np.argmax(probas, axis=1)

        t_end = time.perf_counter()
        total_time_ms = (t_end - t_start) * 1000.0
        avg_latency_ms = round(total_time_ms / max(1, len(df_batch)), 4)

        results = []
        for i in range(len(df_batch)):
            c_idx = int(preds[i])
            conf = float(probas[i, c_idx])
            pred_lbl = self.label_encoder.idx_to_class.get(c_idx, str(c_idx))
            is_attack = str(pred_lbl).lower() not in ("benign", "normal", "benigntraffic")

            if not is_attack:
                sev = "INFORMATIONAL"
            elif conf >= 0.85:
                sev = "CRITICAL"
            elif conf >= 0.65:
                sev = "HIGH"
            else:
                sev = "MEDIUM"

            results.append({
                "record_index": i,
                "predicted_class_idx": c_idx,
                "predicted_label": pred_lbl,
                "confidence": round(conf, 4),
                "is_attack": is_attack,
                "severity": sev,
                "avg_batch_latency_ms": avg_latency_ms
            })
        return results
