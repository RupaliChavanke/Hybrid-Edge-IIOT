# Regression Test Plan & Verification Strategy (Phase 3)

## Objective
Establish a rigorous, automated regression safety net to prevent behavioral regressions, numerical drift, data leakage, and schema violations across the intrusion detection lifecycle.

## Test Matrix Architecture

| Test Suite | Purpose | Target Modules | Coverage Scope |
|---|---|---|---|
| `tests/unit/` | Unit contracts | `data/`, `preprocessing/`, `models/`, `inference/` | Schema validation, zero-leakage splits, label mapping, model training & scoring contracts |
| `tests/integration/` | Cross-module flow | End-to-end inference & alerting | Raw telemetry $\to$ Cleaning $\to$ Feature Selection $\to$ Inference $\to$ Alert Generation |
| `tests/leakage/` | Integrity contracts | `splitting.py`, `cleaning.py`, `feature_selection.py` | Index isolation, zero overlap, target column exclusion, fit isolation on train partition |
| `tests/regression/` | Golden baselines | `models/registry/` | Comparison against `tests/fixtures/golden_predictions.json` within numerical tolerance $10^{-3}$ |
| `tests/performance/` | Operational limits | `catboost_model.py`, `predictor.py` | Single-sample p50 latency $< 5\,\text{ms}$, p95 latency $< 15\,\text{ms}$, throughput $> 1,000\,\text{samples/sec}$ |

## Golden Fixture Specification (`tests/fixtures/golden_predictions.json`)
- **Version:** `2.0.0`
- **Fixture Hash (SHA-256):** `5ad65fe34e03...`
- **Model Baseline:** `catboost_model_v2`
- **Target Schema:** 25 canonical network telemetry features
- **Output Classes:** 15 canonical Edge-IIoTset classes
- **Tolerance Strategy:**
  - Prediction class index: Exact equality (`np.testing.assert_array_equal`)
  - Probability row sum: $1.0 \pm 10^{-4}$ (`np.testing.assert_allclose`)
  - Range: Non-negative and finite $\forall p \in [0.0, 1.0]$

## Continuous Execution Commands
```bash
# Execute entire test suite
pytest

# Execute specific verification suite
pytest tests/unit/
pytest tests/leakage/
pytest tests/regression/
pytest tests/performance/
```
