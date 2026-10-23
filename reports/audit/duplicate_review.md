# Duplicate Detection and Clean-up Audit Review (Phase 1)

**Generated:** 2026-10-22 02:01:11

## Summary
- **Exact Duplicate Groups (Identical SHA-256):** 24
- **Semantic Python Near-Duplicates (Identical AST signature):** 11
- **Quarantine Candidates Identified:** 23

## Canonical Architecture Policy
To ensure backward compatibility and prevent broken dependencies:
1. All canonical data loading, feature transformation, model definitions, and evaluation routines reside under `src/hybrid_ids/`.
2. All operational command-line interfaces reside under `scripts/`.
3. Legacy modules in `preprocessing/`, `training/`, `evaluation/`, and `visualization/` are preserved or safely archived to `archive/` rather than deleted immediately.
4. Compatibility wrappers in root (`train.py`, `app.py`) route directly to the canonical package.
5. Raw datasets (`data/raw/`), pre-calculated evaluation artifacts, and models in `models/registry/` are protected files and marked immutable.

## Exact Duplicate Details
### Group EXACT_001 (SHA-256: `8eb93e131ae4...`)
- `.env` (Size: 542 bytes)
- `.env.example` (Size: 542 bytes)

### Group EXACT_002 (SHA-256: `0a6efb1a0980...`)
- `artifacts/confusion_matrix.png` (Size: 511259 bytes)
- `artifacts/normalized_confusion_matrix.png` (Size: 511259 bytes)

### Group EXACT_003 (SHA-256: `ebaec6640f59...`)
- `artifacts/metrics.json` (Size: 70111 bytes)
- `artifacts/metrics/locked_test_metrics.json` (Size: 70111 bytes)

### Group EXACT_004 (SHA-256: `5682487b43b3...`)
- `artifacts/scaler.pkl` (Size: 2657 bytes)
- `checkpoints/scaler.pkl` (Size: 2657 bytes)

### Group EXACT_005 (SHA-256: `be664554b28f...`)
- `artifacts/label_encoder.pkl` (Size: 457 bytes)
- `models/registry/label_encoder_edge_iiot.pkl` (Size: 457 bytes)

### Group EXACT_006 (SHA-256: `0fc79fadef9b...`)
- `artifacts/hybrid_ensemble.pkl` (Size: 10556806 bytes)
- `models/registry/hybrid_ensemble_v2.pkl` (Size: 10556806 bytes)

### Group EXACT_007 (SHA-256: `326aa4bb2b30...`)
- `artifacts/best_model.pt` (Size: 1509813 bytes)
- `checkpoints/best_model.pt` (Size: 1509813 bytes)

### Group EXACT_008 (SHA-256: `4db30f8580cc...`)
- `artifacts/latency_results.json` (Size: 1210 bytes)
- `artifacts/metrics/runtime_latency_benchmark.json` (Size: 1210 bytes)

### Group EXACT_009 (SHA-256: `e1abb0967959...`)
- `artifacts/model_config.json` (Size: 21353 bytes)
- `artifacts/metrics/validation_summary.json` (Size: 21353 bytes)

### Group EXACT_010 (SHA-256: `11a4232000cb...`)
- `artifacts/cleaner.pkl` (Size: 2142 bytes)
- `models/registry/cleaner_edge_iiot.pkl` (Size: 2142 bytes)

