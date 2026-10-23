# Pre-Refactor Test Baseline & Regression Results (Phase 3)

**Timestamp:** 2026-10-22 02:06:18  
**Environment:** macOS arm64, Python 3.14.7, Pytest 9.1.1  
**Total Tests Executed:** 23  
**Passing Tests:** 23  
**Failed Tests:** 0  
**Test Duration:** 1.54 seconds  

## Execution Breakdown

```
tests/integration/test_end_to_end_pipeline.py::test_end_to_end_train_predict_alert PASSED [  4%]
tests/leakage/test_leakage_contracts.py::test_zero_index_overlap_in_splits PASSED         [  8%]
tests/leakage/test_leakage_contracts.py::test_no_target_columns_in_features PASSED       [ 13%]
tests/leakage/test_leakage_contracts.py::test_feature_selector_fit_isolation PASSED      [ 17%]
tests/performance/test_latency_and_throughput.py::test_inference_latency_bounds PASSED   [ 21%]
tests/performance/test_latency_and_throughput.py::test_batch_prediction_throughput PASSED [ 26%]
tests/regression/test_golden_regression.py::test_golden_prediction_fixture PASSED        [ 30%]
tests/unit/test_ensemble.py::test_ensemble_uniform_averaging PASSED                      [ 34%]
tests/unit/test_ensemble.py::test_ensemble_weight_optimization PASSED                    [ 39%]
tests/unit/test_ensemble.py::test_probability_calibrator_simplex PASSED                  [ 43%]
tests/unit/test_inference.py::test_model_registry_save_load PASSED                       [ 47%]
tests/unit/test_inference.py::test_streaming_replay_pipeline PASSED                      [ 52%]
tests/unit/test_label_mapping.py::test_edge_iiot_label_mapping PASSED                    [ 56%]
tests/unit/test_label_mapping.py::test_cic_iot_2023_label_mapping PASSED                 [ 60%]
tests/unit/test_label_mapping.py::test_unknown_label_handling PASSED                     [ 65%]
tests/unit/test_models.py::test_catboost_fit_predict PASSED                              [ 69%]
tests/unit/test_models.py::test_xgboost_fit_predict PASSED                               [ 73%]
tests/unit/test_models.py::test_lightgbm_fit_predict PASSED                              [ 78%]
tests/unit/test_preprocessing.py::test_pipeline_cleaner_imputation_and_leakage_dropping PASSED [ 82%]
tests/unit/test_preprocessing.py::test_feature_selector_mutual_info PASSED               [ 86%]
tests/unit/test_schema.py::test_data_validator_detects_nulls_infs_constants PASSED        [ 91%]
tests/unit/test_schema.py::test_loader_instantiation PASSED                              [ 95%]
tests/unit/test_splitting.py::test_zero_leakage_split_isolation PASSED                   [100%]
```

## Key Verification Findings
1. **Zero Data Leakage:** Partition isolation between Train ($70\%$), Validation ($15\%$), and Locked-Test ($15\%$) confirms $0$ overlapping indices.
2. **Deterministic Output:** Golden fixture predictions match bitwise for class indices and within $10^{-3}$ tolerance for probability values.
3. **Operational Readiness:** Single-sample inference latency benchmarked well below the $5\,\text{ms}$ budget, and batch throughput verified above $1,000$ samples/sec.
