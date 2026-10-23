# Post-Refactor Comprehensive Test Results (Phase 13)

**Test Execution Date:** 2026-10-22 02:16:51  
**Target Environment:** macOS ARM64, Python 3.14.7, Pytest 9.1.1  
**Total Tests Run:** 23  
**Passing Tests:** 23 / 23 (100% Pass Rate)  
**Failed / Skipped:** 0  
**Suite Runtime:** 1.58 seconds  

## Test Inventory and Status

| Suite | Test Module | Test Name | Status | Duration |
|---|---|---|---|---|
| Integration | `tests/integration/test_end_to_end_pipeline.py` | `test_end_to_end_train_predict_alert` | PASSED | 0.08s |
| Leakage | `tests/leakage/test_leakage_contracts.py` | `test_zero_index_overlap_in_splits` | PASSED | 0.04s |
| Leakage | `tests/leakage/test_leakage_contracts.py` | `test_no_target_columns_in_features` | PASSED | 0.02s |
| Leakage | `tests/leakage/test_leakage_contracts.py` | `test_feature_selector_fit_isolation` | PASSED | 0.03s |
| Performance | `tests/performance/test_latency_and_throughput.py` | `test_inference_latency_bounds` | PASSED | 0.12s |
| Performance | `tests/performance/test_latency_and_throughput.py` | `test_batch_prediction_throughput` | PASSED | 0.15s |
| Regression | `tests/regression/test_golden_regression.py` | `test_golden_prediction_fixture` | PASSED | 0.09s |
| Unit | `tests/unit/test_ensemble.py` | `test_ensemble_uniform_averaging` | PASSED | 0.02s |
| Unit | `tests/unit/test_ensemble.py` | `test_ensemble_weight_optimization` | PASSED | 0.04s |
| Unit | `tests/unit/test_ensemble.py` | `test_probability_calibrator_simplex` | PASSED | 0.03s |
| Unit | `tests/unit/test_inference.py` | `test_model_registry_save_load` | PASSED | 0.02s |
| Unit | `tests/unit/test_inference.py` | `test_streaming_replay_pipeline` | PASSED | 0.05s |
| Unit | `tests/unit/test_label_mapping.py` | `test_edge_iiot_label_mapping` | PASSED | 0.02s |
| Unit | `tests/unit/test_label_mapping.py` | `test_cic_iot_2023_label_mapping` | PASSED | 0.02s |
| Unit | `tests/unit/test_label_mapping.py` | `test_unknown_label_handling` | PASSED | 0.01s |
| Unit | `tests/unit/test_models.py` | `test_catboost_fit_predict` | PASSED | 0.14s |
| Unit | `tests/unit/test_models.py` | `test_xgboost_fit_predict` | PASSED | 0.16s |
| Unit | `tests/unit/test_models.py` | `test_lightgbm_fit_predict` | PASSED | 0.15s |
| Unit | `tests/unit/test_preprocessing.py` | `test_pipeline_cleaner_imputation_and_leakage_dropping` | PASSED | 0.03s |
| Unit | `tests/unit/test_preprocessing.py` | `test_feature_selector_mutual_info` | PASSED | 0.05s |
| Unit | `tests/unit/test_schema.py` | `test_data_validator_detects_nulls_infs_constants` | PASSED | 0.02s |
| Unit | `tests/unit/test_schema.py` | `test_loader_instantiation` | PASSED | 0.01s |
| Unit | `tests/unit/test_splitting.py` | `test_zero_leakage_split_isolation` | PASSED | 0.04s |

## Acceptance Verification
- Zero test regressions detected across modularized packages.
- Zero numerical drift against the golden prediction contract fixture (`tests/fixtures/golden_predictions.json`).
- All data leakage and model schema invariant checks satisfied.
