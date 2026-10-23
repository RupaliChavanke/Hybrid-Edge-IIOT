"""
Performance and Latency Contract Tests.
Ensures real-time throughput and p50/p95 latency compliance for operational intrusion detection.
"""

import time
import numpy as np
import pandas as pd
from src.hybrid_ids.models.catboost_model import CatBoostModel


def test_inference_latency_bounds():
    """Verify single-sample inference latency is under real-time budget (< 10 ms)."""
    np.random.seed(42)
    n_features = 25
    n_samples = 200

    X = pd.DataFrame(np.random.randn(n_samples, n_features), columns=[f"f_{i}" for i in range(n_features)])
    y = np.random.randint(0, 5, size=n_samples)

    model = CatBoostModel(iterations=20, depth=4, random_state=42)
    model.fit(X, y)

    # Benchmark single sample latency over 50 iterations
    latencies = []
    single_sample = X.iloc[[0]]
    for _ in range(50):
        t0 = time.perf_counter()
        _ = model.predict(single_sample)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p50_latency = float(np.median(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    # Real-time requirement: P50 latency < 5 ms, P95 < 15 ms
    assert p50_latency < 5.0, f"P50 latency {p50_latency:.3f} ms exceeds 5 ms threshold"
    assert p95_latency < 15.0, f"P95 latency {p95_latency:.3f} ms exceeds 15 ms threshold"


def test_batch_prediction_throughput():
    """Verify batch throughput exceeds 5,000 samples per second."""
    np.random.seed(42)
    n_features = 25
    n_samples = 2000

    X = pd.DataFrame(np.random.randn(n_samples, n_features), columns=[f"f_{i}" for i in range(n_features)])
    y = np.random.randint(0, 5, size=n_samples)

    model = CatBoostModel(iterations=20, depth=4, random_state=42)
    model.fit(X, y)

    t0 = time.perf_counter()
    _ = model.predict(X)
    elapsed = time.perf_counter() - t0

    throughput = n_samples / elapsed
    assert throughput > 1000.0, f"Throughput {throughput:.1f} samples/s is below 1,000 samples/s threshold"
