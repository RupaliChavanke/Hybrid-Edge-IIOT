# Pipeline Reproduction & Execution Guide (Phase 15)

## 1. System Requirements & Setup
- **Operating System:** macOS (Apple Silicon / Intel) or Linux (x86_64 / arm64)
- **Python Version:** $\ge 3.10$ (tested on Python 3.14.7)
- **Hardware Budget:** $\ge 8\,\text{GB}$ RAM, $\ge 10\,\text{GB}$ free disk space

### Environment Installation
```bash
# Clone and enter directory
cd /Volumes/New\ Storage/Edge-IIOT

# Activate Python virtual environment
source .venv/bin/activate

# Install editable package with development dependencies
pip install -e .
```

---

## 2. Step-by-Step Reproduction Workflow

### Step 1: Audit Repository & Checkpoints (Phase 0 & 1)
```bash
python3 scripts/audit_repository.py
```
*Generates repository inventory, cryptographic hashes, and duplicate cleanup plan in `reports/audit/`.*

### Step 2: Audit Raw Datasets (Phase 4)
```bash
python3 scripts/audit_datasets.py
```
*Performs independent data quality profiling for Edge-IIoTset and CICIoT2023.*

### Step 3: Zero-Leakage Data Preparation & Partitioning (Phase 5)
```bash
python3 scripts/prepare_data.py --k-features 25
```
*Generates 70/15/15 zero-leakage numpy split archives in `data/processed/edge_iiot_processed.npz`.*

### Step 4: Fair Baseline Evaluation (Phase 6)
```bash
python3 scripts/train_baselines.py
```
*Trains 8 baseline models (Majority, Stratified, Logistic Regression, Random Forest, Extra Trees, GBDTs) on the locked test set.*

### Step 5: Hyperparameter Tuning (Phase 7)
```bash
python3 scripts/tune_models.py
```
*Conducts bounded random search on validation data and logs trials in `artifacts/metrics/*_tuning.csv`.*

### Step 6: Hybrid Ensemble Training & Out-of-Fold Stacking (Phase 8)
```bash
python3 scripts/train_ensemble.py
```
*Optimizes Simplex SLSQP ensemble weights and trains regularized out-of-fold stacking meta-model.*

### Step 7: Cross-Dataset Generalization Benchmark (Phase 9)
```bash
python3 scripts/cross_dataset_evaluation.py
```
*Executes Experiments A through F across Edge-IIoTset and CICIoT2023.*

### Step 8: Ablations, Robustness, and Statistical Significance (Phase 10)
```bash
python3 scripts/run_ablations_and_statistics.py
```
*Evaluates noise injection, feature dropout, paired McNemar tests, and 95% bootstrap confidence intervals.*

### Step 9: Runtime Latency and Memory Profiling (Phase 11)
```bash
python3 scripts/benchmark_runtime.py
```
*Measures P50, P95, P99 single-sample latency and throughput (events/sec).*

### Step 10: Real-Time Telemetry Replay & SOC Alerting (Phase 12)
```bash
python3 scripts/run_inference.py
```
*Replays test streams through the production predictor and emits structured SOC incidents.*

### Step 11: Launch Interactive Streamlit SOC Dashboard
```bash
streamlit run app.py
```
*Opens the comprehensive 10-tab SOC evaluation dashboard in browser.*

---

## 3. Automated Regression Verification
```bash
# Run complete test suite (23 tests)
pytest

# Run golden regression contract test specifically
pytest tests/regression/test_golden_regression.py -v
```
