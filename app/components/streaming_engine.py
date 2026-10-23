"""
Production Live Streaming Ingestion & Inference Engine for Streamlit SOC.
Coordinates live network telemetry streaming from Edge-IIoTset, runs real-time
inference with CatBoost, XGBoost, LightGBM, or the Hybrid Ensemble, and publishes
to Redpanda (or InMemoryStreamingBus fallback).
"""

import sys
import os
import time
import uuid
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hybrid_ids.inference.model_registry import ModelRegistry
from src.hybrid_ids.data.label_mapping import EDGE_IIOT_MAPPING
from src.hybrid_ids.streaming.alerting import RECOMMENDED_ACTIONS
from streaming.health import RedpandaHealthChecker
from streaming.bus import get_streaming_bus

try:
    from confluent_kafka import Producer, Consumer
    HAS_CONFLUENT = True
except ImportError:
    HAS_CONFLUENT = False


CRITICAL_ATTACKS = {"DDoS_UDP", "DDoS_TCP", "DDoS_ICMP", "MITM", "Ransomware"}
HIGH_ATTACKS = {"DDoS_HTTP", "SQL_injection", "Backdoor", "Uploading", "Password"}
MEDIUM_ATTACKS = {"Port_Scanning", "Vulnerability_scanner", "XSS", "Fingerprinting"}


class LiveStreamingManager:
    """Manages real-time streaming telemetry ingestion, model scoring, and Redpanda sync."""

    _instance = None

    @classmethod
    def get_instance(cls) -> "LiveStreamingManager":
        if cls._instance is None:
            cls._instance = LiveStreamingManager()
        return cls._instance

    def __init__(self):
        self.registry = ModelRegistry()
        self.health_checker = RedpandaHealthChecker()
        self.bus = get_streaming_bus()
        self.classes = list(EDGE_IIOT_MAPPING.keys())

        # Load preprocessors
        with open(REPO_ROOT / "models" / "registry" / "cleaner_edge_iiot.pkl", "rb") as f:
            self.cleaner = pickle.load(f)
        with open(REPO_ROOT / "models" / "registry" / "feature_selector_edge_iiot.pkl", "rb") as f:
            self.selector = pickle.load(f)

        # Lazy model cache
        self.models: Dict[str, Any] = {}

        # Load streaming telemetry sample dataset
        sample_path = REPO_ROOT / "data" / "samples" / "edge_iiot_sample.csv"
        if not sample_path.exists():
            sample_path = REPO_ROOT / "data" / "raw" / "edge_iiot" / "ML-EdgeIIoT-dataset.csv"
        
        self.df_sample = pd.read_csv(sample_path, nrows=25000, low_memory=False)
        self.total_rows = len(self.df_sample)
        self.current_offset = 0

    def get_model(self, model_key: str):
        """Loads and caches the requested ML model."""
        if model_key in self.models:
            return self.models[model_key]

        if model_key == "Hybrid Ensemble":
            model = self.registry.load_artifact("hybrid_ensemble", "v2")
        elif model_key == "CatBoost":
            model = self.registry.load_artifact("catboost_model", "v2")
        elif model_key == "XGBoost":
            model = self.registry.load_artifact("xgboost_model", "v2")
        elif model_key == "LightGBM":
            model = self.registry.load_artifact("lightgbm_model", "v2")
        else:
            model = self.registry.load_artifact("hybrid_ensemble", "v2")

        self.models[model_key] = model
        return model

    def check_redpanda_health(self) -> Dict[str, Any]:
        """Probes the Redpanda cluster / in-memory streaming bus."""
        return self.health_checker.check_health()

    def process_streaming_batch(
        self,
        model_name: str = "Hybrid Ensemble",
        batch_size: int = 5
    ) -> List[Dict[str, Any]]:
        """Consumes a burst of network flow packets, scores them, and generates live telemetry events."""
        # Wrap around sample dataset circularly
        start_idx = self.current_offset
        end_idx = start_idx + batch_size
        self.current_offset = (end_idx) % self.total_rows

        if end_idx <= self.total_rows:
            batch_df = self.df_sample.iloc[start_idx:end_idx]
        else:
            batch_df = pd.concat([
                self.df_sample.iloc[start_idx:],
                self.df_sample.iloc[:(end_idx % self.total_rows)]
            ], ignore_index=True)

        # 1. Cleaning & Preprocessing
        t0 = time.perf_counter()
        clean_df = self.cleaner.transform(batch_df)
        X_sel = self.selector.transform(clean_df)
        prep_time = (time.perf_counter() - t0) * 1000.0

        # 2. Model Inference
        model = self.get_model(model_name)
        t_inf0 = time.perf_counter()
        probs = model.predict_proba(X_sel)
        preds = model.predict(X_sel)
        inf_time = (time.perf_counter() - t_inf0) * 1000.0
        per_packet_latency = (prep_time + inf_time) / max(1, len(batch_df))

        # 3. Build Telemetry Events & Publish to Redpanda / Bus
        events = []
        for i in range(len(batch_df)):
            row = batch_df.iloc[i]
            pred_idx = int(preds[i])
            pred_class = self.classes[pred_idx]
            conf = float(probs[i][pred_idx])
            
            # Shannon entropy of probability distribution: -sum(p * log2(p))
            p_vec = probs[i]
            p_vec_safe = p_vec[p_vec > 0]
            entropy = float(-np.sum(p_vec_safe * np.log2(p_vec_safe + 1e-12)))

            # Threat mapping
            meta = EDGE_IIOT_MAPPING.get(pred_class, {})
            threat_cat = meta.get("canonical_category", "Benign" if pred_class == "Normal" else "DDoS")
            gt_label = str(row.get("Attack_type", "Normal"))
            is_threat = (pred_class != "Normal")

            # Risk assignment
            if not is_threat:
                risk = "NORMAL"
            elif pred_class in CRITICAL_ATTACKS:
                risk = "CRITICAL"
            elif pred_class in HIGH_ATTACKS:
                risk = "HIGH"
            else:
                risk = "MEDIUM"

            action = RECOMMENDED_ACTIONS.get(threat_cat, "Monitor network telemetry and inspect firewall logs.")
            event_id = str(uuid.uuid4())[:8]
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")

            # Determine protocol from raw fields if available
            proto_num = row.get("ip.proto", 6)
            proto_map = {6: "TCP", 17: "UDP", 1: "ICMP"}
            proto = proto_map.get(int(proto_num) if pd.notnull(proto_num) else 6, "TCP")

            event_record = {
                "event_id": f"EVT-{event_id}",
                "timestamp": timestamp_str,
                "protocol": proto,
                "predicted_attack": pred_class,
                "threat_category": threat_cat,
                "confidence": conf,
                "entropy": entropy,
                "latency_ms": per_packet_latency,
                "risk_level": risk,
                "ground_truth": gt_label,
                "is_threat": is_threat,
                "model_used": model_name,
                "recommended_action": action,
                "summary": f"Proto={proto} | FrameLen={row.get('frame.len', 64)} | Flags={row.get('tcp.flags', '0x02')}"
            }
            events.append(event_record)

            # Publish raw & predictions to InMemoryStreamingBus / Redpanda topics
            try:
                self.bus.publish("edge-iiot-raw", row.to_dict())
                self.bus.publish("ids-predictions", event_record)
                if is_threat:
                    self.bus.publish("ids-alerts", {
                        "alert_id": f"ALT-{event_id}",
                        "timestamp": timestamp_str,
                        "attack_type": pred_class,
                        "threat_category": threat_cat,
                        "confidence": conf,
                        "severity": risk,
                        "recommended_action": action
                    })
            except Exception:
                pass

        return events

    def inject_attack_sample(
        self,
        attack_type: str = "DDoS_UDP",
        model_name: str = "Hybrid Ensemble"
    ) -> Dict[str, Any]:
        """Injects a single designated attack packet from the real dataset into the live streaming pipeline."""
        matched = self.df_sample[self.df_sample["Attack_type"] == attack_type]
        if len(matched) > 0:
            sample_row = matched.sample(n=1, random_state=int(time.time() * 1000) % 10000)
        else:
            sample_row = self.df_sample.sample(n=1)

        t0 = time.perf_counter()
        clean_df = self.cleaner.transform(sample_row)
        t_clean = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        X_sel = self.selector.transform(clean_df)
        t_sel = (time.perf_counter() - t1) * 1000.0

        model = self.get_model(model_name)
        t2 = time.perf_counter()
        probs = model.predict_proba(X_sel)
        preds = model.predict(X_sel)
        t_inf = (time.perf_counter() - t2) * 1000.0

        pred_idx = int(preds[0])
        pred_class = self.classes[pred_idx]
        conf = float(probs[0][pred_idx])

        p_vec = probs[0]
        p_vec_safe = p_vec[p_vec > 0]
        entropy = float(-np.sum(p_vec_safe * np.log2(p_vec_safe + 1e-12)))

        meta = EDGE_IIOT_MAPPING.get(pred_class, {})
        threat_cat = meta.get("canonical_category", "Benign" if pred_class == "Normal" else "DDoS")
        gt_label = str(sample_row.iloc[0].get("Attack_type", "Normal"))
        is_threat = (pred_class != "Normal")

        if not is_threat:
            risk = "NORMAL"
        elif pred_class in CRITICAL_ATTACKS:
            risk = "CRITICAL"
        elif pred_class in HIGH_ATTACKS:
            risk = "HIGH"
        else:
            risk = "MEDIUM"

        action = RECOMMENDED_ACTIONS.get(threat_cat, "Monitor network telemetry and inspect firewall logs.")
        event_id = str(uuid.uuid4())[:8]
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")

        proto_num = sample_row.iloc[0].get("ip.proto", 6)
        proto_map = {6: "TCP", 17: "UDP", 1: "ICMP"}
        proto = proto_map.get(int(proto_num) if pd.notnull(proto_num) else 6, "TCP")

        event_record = {
            "event_id": f"INJ-{event_id}",
            "timestamp": timestamp_str,
            "protocol": proto,
            "predicted_attack": pred_class,
            "threat_category": threat_cat,
            "confidence": conf,
            "entropy": entropy,
            "latency_ms": t_clean + t_sel + t_inf,
            "latency_breakdown": {
                "clean_ms": t_clean,
                "feature_select_ms": t_sel,
                "inference_ms": t_inf,
                "total_ms": t_clean + t_sel + t_inf
            },
            "risk_level": risk,
            "ground_truth": gt_label,
            "is_threat": is_threat,
            "model_used": model_name,
            "recommended_action": action,
            "summary": f"Injected Packet: Proto={proto} | Target Attack={attack_type} | Ground Truth={gt_label}"
        }

        try:
            self.bus.publish("edge-iiot-raw", sample_row.iloc[0].to_dict())
            self.bus.publish("ids-predictions", event_record)
            if is_threat:
                self.bus.publish("ids-alerts", {
                    "alert_id": f"ALT-{event_id}",
                    "timestamp": timestamp_str,
                    "attack_type": pred_class,
                    "threat_category": threat_cat,
                    "confidence": conf,
                    "severity": risk,
                    "recommended_action": action
                })
        except Exception:
            pass

        return event_record
