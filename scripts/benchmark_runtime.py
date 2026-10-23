"""
Runtime Latency, Throughput, and Memory Profiling Script.
Profiles P50, P95, P99 single-sample inference latency, batch throughput (events/sec),
and peak memory consumption for production intrusion detection.
"""

from pathlib import Path
import sys
import time
import json
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.hybrid_ids.inference.model_registry import ModelRegistry
from src.hybrid_ids.utils.resource_monitor import ResourceMonitor
from src.hybrid_ids.config import get_project_root
from src.hybrid_ids.logging_config import setup_logger

logger = setup_logger("benchmark_runtime")


def main():
    root = get_project_root()
    registry = ModelRegistry()
    monitor = ResourceMonitor()

    data_path = root / "data" / "processed" / "edge_iiot_processed.npz"
    data = np.load(data_path, allow_pickle=True)
    X_test = data["X_test"]

    logger.info("Loading registered models for runtime profiling...")
    cat_model = registry.load_artifact("catboost_model", version="v2")
    xgb_model = registry.load_artifact("xgboost_model", version="v2")
    lgb_model = registry.load_artifact("lightgbm_model", version="v2")
    ensemble = registry.load_artifact("hybrid_ensemble", version="v2")

    models_to_test = [
        ("CatBoost", cat_model),
        ("XGBoost", xgb_model),
        ("LightGBM", lgb_model),
        ("Hybrid_Ensemble", ensemble)
    ]

    profile_results = {}
    warmup_n = 50
    test_n = min(1000, len(X_test))

    for name, m in models_to_test:
        logger.info(f"Benchmarking latency for {name} ({test_n} samples)...")

        # Warm-up phase
        for i in range(warmup_n):
            _ = m.predict(X_test[i:i+1])

        latencies_ms = []
        for i in range(test_n):
            sample = X_test[i:i+1]
            t0 = time.perf_counter()
            _ = m.predict(sample)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        latencies_ms = np.array(latencies_ms)
        p50 = float(np.percentile(latencies_ms, 50))
        p95 = float(np.percentile(latencies_ms, 95))
        p99 = float(np.percentile(latencies_ms, 99))
        mean_lat = float(np.mean(latencies_ms))

        # Batch Throughput
        batch_sizes = [100, 500, 1000]
        throughputs = {}
        for bs in batch_sizes:
            if bs <= len(X_test):
                batch_sample = X_test[:bs]
                t0 = time.perf_counter()
                _ = m.predict(batch_sample)
                t1 = time.perf_counter()
                dur = max(1e-6, t1 - t0)
                throughputs[f"batch_{bs}"] = round(bs / dur, 2)

        mem_info = monitor.get_memory_info()

        profile_results[name] = {
            "p50_latency_ms": round(p50, 4),
            "p95_latency_ms": round(p95, 4),
            "p99_latency_ms": round(p99, 4),
            "mean_latency_ms": round(mean_lat, 4),
            "throughput_events_per_sec": throughputs,
            "process_rss_mb": mem_info["process_rss_mb"]
        }
        logger.info(f"{name} Results -> P50: {p50:.4f}ms, P95: {p95:.4f}ms, P99: {p99:.4f}ms")

    metrics_dir = root / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "runtime_latency_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(profile_results, f, indent=2)

    logger.info("Runtime profiling complete. Saved to artifacts/metrics/runtime_latency_benchmark.json")


if __name__ == "__main__":
    main()
